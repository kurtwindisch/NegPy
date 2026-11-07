//! The picture in a full film scan. A `Format` reads the strip's own marks for tilt, scale and
//! film center; the stages after it read the rebate tone and find the frame's sides, and never
//! ask which format ran.

use crate::components::connected_components;
use crate::filter::{self, box_filter};
use crate::par_zip;
use crate::stats::{median_f64, np_median, np_percentiles};
use crate::warp::{self, Image, Interp, rotation_matrix, warp_affine};

pub type Roi = (i64, i64, i64, i64);

pub struct StripFrame {
    /// The largest straight rectangle in the frame, in the input's coordinates.
    pub roi: Roi,
    /// The film box Rebate Trim scales between, in the input's coordinates.
    pub film: Roi,
    /// The strip's tilt in degrees: `apply_fine_rotation` by this straightens it.
    pub angle: f64,
}

/// What a strip reader measures, on the plane it straightened.
struct StripGeometry {
    straight: Vec<f32>,
    angle: f64,
    px_per_mm: f64,
    center: f64,
    /// One row of holes, so `center` is assumed from the format.
    lone_row: bool,
}

/// One frame size of a format, in mm. `along` runs with the film.
struct FrameSpec {
    along: f64,
    across: f64,
    /// Limits of a frame's width as a share of `along`.
    negative: (f64, f64),
    slide: (f64, f64),
    /// A frame's width as a share of `along`, for a frame that shows one side.
    frame_width: f64,
    /// Distance from the film center to the margin that holds the rebate tone.
    margin: f64,
    film_across: f64,
    gap: f64,
}

struct Format {
    read: fn(&[f32], usize, usize) -> Option<StripGeometry>,
    frames: &'static [FrameSpec],
}

const FORMATS: &[Format] = &[Format {
    read: read_sprockets,
    frames: &[FrameSpec {
        along: 36.0,
        across: 24.3,
        negative: (0.98, 1.02),
        slide: (0.98, 1.06),
        frame_width: 1.005,
        margin: 12.35,
        film_across: 35.0,
        gap: 2.0,
    }],
}];

const SPROCKET_PITCH_MM: f64 = 4.75;
const ROW_TO_CENTER_MM: f64 = 14.0;
const ROW_DISTANCE_MM: (f64, f64) = (26.5, 29.5);
const HOLE_LEVEL: f64 = 0.72;
const HOLE_AREA: (f64, f64) = (0.0006, 0.012);
const HOLE_ASPECT: (f64, f64) = (0.45, 1.15);
const HOLE_FILL: f64 = 0.8;
/// How far the early screen loosens each hole limit.
const SCREEN_SLACK: f64 = 0.8;
const SCREEN_STEP: usize = 3;
const ROW_BAND: f64 = 0.03;
const FIRST_ROW_HOLES: usize = 4;
const SECOND_ROW_HOLES: usize = 3;
const EVEN_SPACING: (f64, f64) = (0.10, 0.80);
const PITCH_OF_LONG_EDGE: (f64, f64) = (1.0 / 36.0, 1.0 / 4.5);

const DENSITY_FLOOR: f32 = 1e-3;
/// Low first: a high threshold passes a scene edge inside the frame.
const NEGATIVE_LEVELS: [f32; 3] = [0.03, 0.05, 0.08];
const SLIDE_LEVELS: [f32; 2] = [0.15, 0.25];
const CLEAN_COLUMN: f64 = 0.03;
const CLEAN_RUN: f64 = 0.012;
const MIN_CLEAN_RUN: usize = 5;
const SIDE_ROWS_INSET: f64 = 0.12;
/// Two half frames read as one frame with a clean gutter inside it.
const INNER_GAP_INSET: f64 = 0.10;
const MARGIN_BAND_MM: f64 = 0.171;
const FAR_SIDE_MM: f64 = 1.2;
const FAR_SIDE_LEVEL: f32 = 0.08;
const LINE_SPAN: usize = 8;
const LINE_STEP: f32 = 0.02;
const LINE_ROWS: f64 = 0.8;
const GUTTER_INSET: f64 = 0.10;
const GUTTER_MM: f64 = 3.0;
const GUTTER_WALL: usize = 8;
const GUTTER_EXPOSED: f64 = 0.2;
/// How far from its place, and over what width, a frame's edge must show when a lone row sets it.
const ROW_EDGE_REACH_MM: f64 = 1.0;
const ROW_EDGE_SPAN_MM: f64 = 0.4;
const ROW_EDGE_SHARE: f64 = 0.3;

#[derive(Clone, Copy)]
struct Hole {
    x: f64,
    y: f64,
}

/// Sprocket holes: compact bright shapes clear of the scan's edge. The shape limits stay wide,
/// because cine and still perforations differ.
fn holes(lum: &[f32], h: usize, w: usize) -> Vec<Hole> {
    let Some(q) = np_percentiles(lum, &[99.5]) else {
        return Vec::new();
    };
    let level = (HOLE_LEVEL as f32) * q[0];
    let mask: Vec<u8> = lum.iter().map(|&v| u8::from(v > level)).collect();
    let c = connected_components(&mask, h, w, 8);
    let frame = (h * w) as f64;
    (1..c.count)
        .filter_map(|i| {
            let [x, y, bw, bh, area] = c.stats[i];
            let (bw_f, bh_f, area_f) = (bw as f64, bh as f64, area as f64);
            let clear = x > 0 && y > 0 && x + bw < w as i32 && y + bh < h as i32;
            let sized = HOLE_AREA.0 * frame < area_f && area_f < HOLE_AREA.1 * frame;
            let shaped = HOLE_ASPECT.0 < bw_f / bh_f
                && bw_f / bh_f < HOLE_ASPECT.1
                && area_f / (bw_f * bh_f) > HOLE_FILL;
            (clear && sized && shaped).then(|| Hole {
                x: c.centroids[i][0],
                y: c.centroids[i][1],
            })
        })
        .collect()
}

/// The largest band of holes on one line across the scan: the first of equal size.
fn largest_row(holes: &[Hole], band: f64) -> Vec<Hole> {
    let near = |y: f64| holes.iter().filter(move |o| (o.y - y).abs() < band);
    let mut best: Option<(f64, usize)> = None;
    for hole in holes {
        let n = near(hole.y).count();
        if best.is_none_or(|(_, m)| n > m) {
            best = Some((hole.y, n));
        }
    }
    best.map_or_else(Vec::new, |(y, _)| near(y).copied().collect())
}

fn gaps(row: &[Hole]) -> Vec<f64> {
    let mut x: Vec<f64> = row.iter().map(|o| o.x).collect();
    x.sort_by(f64::total_cmp);
    x.windows(2).map(|p| p[1] - p[0]).collect()
}

/// A sprocket row keeps one pitch; a row of bright shapes in a scene does not.
fn evenly_spaced(row: &[Hole]) -> bool {
    let mut dx = gaps(row);
    if dx.len() < 2 {
        return false;
    }
    let pitch = median_f64(&mut dx.clone());
    let even = dx
        .iter_mut()
        .filter(|d| (**d / pitch - 1.0).abs() < EVEN_SPACING.0);
    even.count() as f64 / dx.len() as f64 >= EVEN_SPACING.1
}

fn hole_rows(lum: &[f32], h: usize, w: usize) -> Option<(Vec<Hole>, Option<Vec<Hole>>)> {
    let found = holes(lum, h, w);
    if found.len() < FIRST_ROW_HOLES {
        return None;
    }
    let band = ROW_BAND * h as f64;
    let first = largest_row(&found, band);
    if !evenly_spaced(&first) {
        return None;
    }
    let rest: Vec<Hole> = found
        .iter()
        .filter(|o| !first.iter().any(|f| f.x == o.x && f.y == o.y))
        .copied()
        .collect();
    let second = (rest.len() >= SECOND_ROW_HOLES)
        .then(|| largest_row(&rest, band))
        .filter(|r| r.len() >= SECOND_ROW_HOLES && evenly_spaced(r));
    Some((first, second))
}

fn row_slope(row: &[Hole]) -> f64 {
    let n = row.len() as f64;
    let (mx, my) = (
        row.iter().map(|o| o.x).sum::<f64>() / n,
        row.iter().map(|o| o.y).sum::<f64>() / n,
    );
    let sxy: f64 = row.iter().map(|o| (o.x - mx) * (o.y - my)).sum();
    let sxx: f64 = row.iter().map(|o| (o.x - mx) * (o.x - mx)).sum();
    sxy / sxx
}

fn mean_y(row: &[Hole]) -> f64 {
    row.iter().map(|o| o.y).sum::<f64>() / row.len() as f64
}

/// Bilinear, edge pixels repeated: the same turn as `apply_fine_rotation`.
fn rotated(lum: &[f32], h: usize, w: usize, angle: f64) -> Vec<f32> {
    let mut out = vec![0f32; lum.len()];
    let src = Image {
        data: lum,
        h,
        w,
        cn: 1,
    };
    let m = rotation_matrix((w as f32 / 2.0, h as f32 / 2.0), angle, 1.0);
    let mode = (Interp::Linear, warp::Border::Replicate);
    warp_affine(&src, mode, m, false, w, &mut out);
    out
}

fn read_sprockets(lum: &[f32], h: usize, w: usize) -> Option<StripGeometry> {
    let (first, second) = hole_rows(lum, h, w)?;
    let slopes: Vec<f64> = [Some(&first), second.as_ref()]
        .into_iter()
        .flatten()
        .filter(|r| r.len() >= SECOND_ROW_HOLES)
        .map(|r| row_slope(r))
        .collect();
    let angle = libm::atan(slopes.iter().sum::<f64>() / slopes.len() as f64).to_degrees();
    let straight = rotated(lum, h, w, angle);
    let (first, second) = hole_rows(&straight, h, w)?;
    let pitch = median_f64(&mut gaps(&first));
    let long = h.max(w) as f64;
    if !(PITCH_OF_LONG_EDGE.0 * long < pitch && pitch < PITCH_OF_LONG_EDGE.1 * long) {
        return None;
    }
    let px_per_mm = pitch / SPROCKET_PITCH_MM;
    let center = match &second {
        Some(second) => {
            let (a, b) = (mean_y(&first), mean_y(second));
            let apart = (a - b).abs() / px_per_mm;
            if !(ROW_DISTANCE_MM.0 < apart && apart < ROW_DISTANCE_MM.1) {
                return None;
            }
            (a + b) / 2.0
        }
        None => {
            let y = mean_y(&first);
            let reach = ROW_TO_CENTER_MM * px_per_mm;
            if y < h as f64 / 2.0 {
                y + reach
            } else {
                y - reach
            }
        }
    };
    Some(StripGeometry {
        straight,
        angle,
        px_per_mm,
        center,
        lone_row: second.is_none(),
    })
}

fn density(lum: &[f32], h: usize, w: usize) -> Option<Vec<f32>> {
    let mut log = vec![0f32; lum.len()];
    par_zip(&mut log, lum, move |d, &v| {
        *d = -crate::mathf::log10f(v.max(DENSITY_FLOOR))
    });
    let k = ((w as f64 / 360.0).round() as usize).max(1) | 1;
    let mut out = vec![0f32; lum.len()];
    box_filter(
        &log,
        &mut out,
        (h, w, 1),
        (k, k),
        true,
        filter::Border::Reflect101,
    )?;
    Some(out)
}

/// Blocks of the margin between the holes and the picture, on both sides of the film center.
fn margin_blocks(d: &[f32], h: usize, w: usize, g: &StripGeometry, spec: &FrameSpec) -> Vec<f32> {
    let half_h = ((MARGIN_BAND_MM * g.px_per_mm) as usize).max(2);
    let half_w = (w / 60).max(2);
    let mut blocks = Vec::new();
    for sign in [-1.0, 1.0] {
        let y = (g.center + sign * spec.margin * g.px_per_mm) as i64;
        if y < half_h as i64 || y as usize + half_h >= h {
            continue;
        }
        let y = y as usize;
        let mut x = half_w;
        while x + half_w <= w - half_w {
            let mut block: Vec<f32> = (y - half_h..y + half_h)
                .flat_map(|r| d[r * w + x - half_w..r * w + x + half_w].iter().copied())
                .collect();
            blocks.push(np_median(&mut block));
            x += 2 * half_w;
        }
    }
    blocks
}

/// Starts and ends of the exposed runs that have a clean run of rebate outside them.
fn sides(exposed: &[f64], min_run: usize) -> (Vec<usize>, Vec<usize>) {
    let mut runs: Vec<(usize, usize, bool)> = Vec::new();
    let mut i = 0;
    while i < exposed.len() {
        let on = exposed[i] >= CLEAN_COLUMN;
        let j = (i..exposed.len())
            .find(|&j| (exposed[j] >= CLEAN_COLUMN) != on)
            .unwrap_or(exposed.len());
        runs.push((i, j, on));
        i = j;
    }
    let (mut left, mut right) = (Vec::new(), Vec::new());
    for p in runs.windows(2) {
        if !p[0].2 && p[1].2 && p[0].1 - p[0].0 >= min_run {
            left.push(p[1].0);
        }
        if p[0].2 && !p[1].2 && p[1].1 - p[1].0 >= min_run {
            right.push(p[1].0);
        }
    }
    (left, right)
}

fn has_inner_gap(exposed: &[f64], l: usize, r: usize, min_run: usize) -> bool {
    let inset = ((r - l) as f64 * INNER_GAP_INSET) as usize;
    longest_clean_run(&exposed[l + inset..r - inset]) >= min_run
}

/// The frame's rows and columns on the straightened plane, nearest the scan's center.
fn find_frame(
    g: &StripGeometry,
    h: usize,
    w: usize,
    spec: &FrameSpec,
    slide: bool,
) -> Option<(i64, i64, usize, usize)> {
    let d = density(&g.straight, h, w)?;
    let blocks = margin_blocks(&d, h, w, g, spec);
    let tone = if slide {
        np_percentiles(&blocks, &[90.0])?[0]
    } else {
        np_median(&mut blocks.clone())
    };
    let (long, across) = (spec.along * g.px_per_mm, spec.across * g.px_per_mm);
    let (y0, y1) = (
        (g.center - across / 2.0) as i64,
        (g.center + across / 2.0) as i64,
    );
    let inset = (SIDE_ROWS_INSET * across) as i64;
    let (a, b) = (
        (y0 + inset).max(0) as usize,
        ((y1 - inset).min(h as i64)).max(0) as usize,
    );
    if b <= a {
        return None;
    }
    let limits = if slide { spec.slide } else { spec.negative };
    let levels: &[f32] = if slide {
        &SLIDE_LEVELS
    } else {
        &NEGATIVE_LEVELS
    };
    let min_run = ((CLEAN_RUN * long) as usize).max(MIN_CLEAN_RUN);
    let band = &d[a * w..b * w];
    let edge_level = levels[levels.len() - 1];
    let frame_ends = |l: usize, r: usize| {
        !g.lone_row
            || [y0, y1]
                .iter()
                .all(|&edge| has_row_edge(&d, (h, w), edge, (l, r), edge_level, g.px_per_mm))
    };
    for &level in levels {
        let exposed = exposed_columns(band, w, tone, level);
        let (left, right) = sides(&exposed, min_run);
        let mut best: Option<(f64, usize, usize)> = None;
        for &l in &left {
            for &r in right.iter().filter(|&&r| r > l) {
                let share = (r - l) as f64 / long;
                if !(limits.0 < share && share < limits.1) || has_inner_gap(&exposed, l, r, min_run)
                {
                    continue;
                }
                let off = ((l + r) as f64 / 2.0 - w as f64 / 2.0).abs();
                if best.is_none_or(|(o, _, _)| off < o) {
                    best = Some((off, l, r));
                }
            }
        }
        if let Some((_, l, r)) = best {
            return frame_ends(l, r).then_some((y0, y1, l, r));
        }
    }
    let width = spec.frame_width * long;
    let (l, r) = one_sided(band, w, tone, levels[0], width, g.px_per_mm, min_run)?;
    frame_ends(l, r).then_some((y0, y1, l, r))
}

/// A density step along the frame's rows near `edge`, or an edge off the plane. A lone row places
/// the frame by assuming the format, and a larger format's picture runs past the far edge with no step.
fn has_row_edge(
    d: &[f32],
    (h, w): (usize, usize),
    edge: i64,
    (l, r): (usize, usize),
    level: f32,
    px_per_mm: f64,
) -> bool {
    if edge < 0 || edge >= h as i64 {
        return true;
    }
    let reach = (ROW_EDGE_REACH_MM * px_per_mm) as i64;
    let span = ((ROW_EDGE_SPAN_MM * px_per_mm) as i64).max(2);
    let (lo, hi) = (
        (edge - reach).max(span),
        (edge + reach).min(h as i64 - span - 1),
    );
    (lo..=hi).any(|y| {
        let inner = &d[(y + span) as usize * w + l..(y + span) as usize * w + r];
        let outer = &d[(y - span) as usize * w + l..(y - span) as usize * w + r];
        let steps = inner
            .iter()
            .zip(outer)
            .filter(|(a, b)| (**a - **b).abs() > level)
            .count();
        steps as f64 >= ROW_EDGE_SHARE * (r - l) as f64
    })
}

fn exposed_columns(band: &[f32], w: usize, tone: f32, level: f32) -> Vec<f64> {
    let rows = (band.len() / w) as f64;
    let mut counts = vec![0u32; w];
    for row in band.chunks_exact(w) {
        for (n, &v) in counts.iter_mut().zip(row) {
            *n += u32::from((v - tone).abs() > level);
        }
    }
    counts.iter().map(|&n| n as f64 / rows).collect()
}

fn longest_clean_run(exposed: &[f64]) -> usize {
    let (mut run, mut longest) = (0, 0);
    for &v in exposed {
        run = if v < CLEAN_COLUMN { run + 1 } else { 0 };
        longest = longest.max(run);
    }
    longest
}

/// A narrow clean run inside the frame, with exposed film on both sides: a gutter or a frame gap.
fn has_gutter(exposed: &[f64], l: usize, r: usize, px_per_mm: f64) -> bool {
    let inset = ((r - l) as f64 * GUTTER_INSET) as usize;
    let mut i = l + inset;
    while i < r - inset {
        if exposed[i] >= CLEAN_COLUMN {
            i += 1;
            continue;
        }
        let end = (i..exposed.len())
            .find(|&j| exposed[j] >= CLEAN_COLUMN)
            .unwrap_or(exposed.len());
        let start = (0..i)
            .rev()
            .find(|&j| exposed[j] >= CLEAN_COLUMN)
            .map_or(0, |j| j + 1);
        let mean = |part: &[f64]| part.iter().sum::<f64>() / part.len().max(1) as f64;
        let before = mean(&exposed[start.saturating_sub(GUTTER_WALL)..start]);
        let after = mean(&exposed[end..(end + GUTTER_WALL).min(exposed.len())]);
        if ((end - start) as f64) < GUTTER_MM * px_per_mm
            && before > GUTTER_EXPOSED
            && after > GUTTER_EXPOSED
        {
            return true;
        }
        i = end;
    }
    false
}

/// A straight edge through the frame's rows, in either direction.
fn has_line(band: &[f32], w: usize, lo: usize, hi: usize) -> bool {
    if hi < lo + LINE_SPAN + 1 {
        return false;
    }
    let rows = (band.len() / w) as f64;
    (lo..hi - LINE_SPAN).any(|x| {
        let (mut up, mut down) = (0u32, 0u32);
        for row in band.chunks_exact(w) {
            let step = row[x + LINE_SPAN] - row[x];
            up += u32::from(step > LINE_STEP);
            down += u32::from(step < -LINE_STEP);
        }
        up.max(down) as f64 / rows >= LINE_ROWS
    })
}

/// A frame placed from the one side it shows, when the other fades into the rebate. The found side
/// comes from the lowest level alone, so a scene edge has no clean run outside it, and the placed
/// side must meet clean film or a straight edge.
fn one_sided(
    band: &[f32],
    w: usize,
    tone: f32,
    level: f32,
    width: f64,
    px_per_mm: f64,
    min_run: usize,
) -> Option<(usize, usize)> {
    let exposed = exposed_columns(band, w, tone, level);
    let looser = exposed_columns(band, w, tone, FAR_SIDE_LEVEL);
    let (left, right) = sides(&exposed, min_run);
    let centre = |l: usize, r: usize| ((l + r) as f64 / 2.0 - w as f64 / 2.0).abs();
    let from_left = left
        .iter()
        .map(|&l| (l, (l as f64 + width) as usize, false))
        .filter(|&(_, r, _)| r <= w);
    let from_right = right
        .iter()
        .filter(|&&r| r as f64 >= width)
        .map(|&r| ((r as f64 - width) as usize, r, true));
    let mut found: Vec<(usize, usize, bool)> = from_left.chain(from_right).collect();
    found.sort_by(|a, b| centre(a.0, a.1).total_cmp(&centre(b.0, b.1)));
    let reach = (FAR_SIDE_MM * px_per_mm) as usize;
    found.into_iter().find_map(|(l, r, placed_left)| {
        let far = if placed_left { l } else { r };
        let (lo, hi) = (far.saturating_sub(reach), (far + reach).min(w));
        let met = longest_clean_run(&exposed[lo..hi]) >= min_run
            || longest_clean_run(&looser[lo..hi]) >= min_run
            || has_line(band, w, lo, hi);
        (met && !has_gutter(&exposed, l, r, px_per_mm)).then_some((l, r))
    })
}

/// A straightened point in the coordinates the plane had before the turn.
fn unrotated((x, y): (f64, f64), h: usize, w: usize, angle: f64) -> (f64, f64) {
    let (s, c) = (libm::sin(angle.to_radians()), libm::cos(angle.to_radians()));
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    let (dx, dy) = (x - cx, y - cy);
    (c * dx - s * dy + cx, s * dx + c * dy + cy)
}

/// The straight rectangle inside a tilted box. At no tilt it is the box itself.
fn inside(box_: (f64, f64, f64, f64), h: usize, w: usize, angle: f64) -> Roi {
    let (y0, y1, x0, x1) = box_;
    let p = |x, y| unrotated((x, y), h, w, angle);
    let (tl, tr, bl, br) = (p(x0, y0), p(x1, y0), p(x0, y1), p(x1, y1));
    (
        tl.1.max(tr.1).ceil() as i64,
        bl.1.min(br.1).floor() as i64,
        tl.0.max(bl.0).ceil() as i64,
        tr.0.min(br.0).floor() as i64,
    )
}

/// The straight rectangle around a tilted box.
fn around(box_: (f64, f64, f64, f64), h: usize, w: usize, angle: f64) -> Roi {
    let (y0, y1, x0, x1) = box_;
    let p = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)].map(|c| unrotated(c, h, w, angle));
    let lo = |f: fn(&(f64, f64)) -> f64| p.iter().map(f).fold(f64::INFINITY, f64::min);
    let hi = |f: fn(&(f64, f64)) -> f64| p.iter().map(f).fold(f64::NEG_INFINITY, f64::max);
    (
        lo(|c| c.1).floor() as i64,
        hi(|c| c.1).ceil() as i64,
        lo(|c| c.0).floor() as i64,
        hi(|c| c.0).ceil() as i64,
    )
}

fn clamp((y1, y2, x1, x2): Roi, h: usize, w: usize) -> Roi {
    (y1.max(0), y2.min(h as i64), x1.max(0), x2.min(w as i64))
}

/// The plane turned a quarter counter-clockwise, as `np.rot90`.
fn quarter_turn(lum: &[f32], h: usize, w: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(lum.len());
    for x in (0..w).rev() {
        out.extend((0..h).map(|y| lum[y * w + x]));
    }
    out
}

/// A rectangle on the quarter-turned plane, in the coordinates of a plane `w` wide.
fn from_quarter((y1, y2, x1, x2): Roi, w: usize) -> Roi {
    (x1, x2, w as i64 - y2, w as i64 - y1)
}

fn mean_plane(img: &[f32], h: usize, w: usize, cn: usize) -> Vec<f32> {
    if cn < 3 {
        return img.iter().step_by(cn.max(1)).copied().collect();
    }
    let mut out = vec![0f32; h * w];
    out.iter_mut()
        .zip(img.chunks_exact(cn))
        .for_each(|(d, p)| *d = (p[0] + p[1] + p[2]) / 3.0);
    out
}

/// A sparse, looser `holes` on either axis: a scan with no strip marks stops here, so it does
/// not pay for a search per axis. The level stays exact: a lower one joins a thin rebate to its holes.
fn may_show_holes(img: &[f32], h: usize, w: usize, cn: usize) -> bool {
    let (sh, sw) = (h / SCREEN_STEP, w / SCREEN_STEP);
    let channels = cn.min(3);
    let mut plane = Vec::with_capacity(sh * sw);
    for y in 0..sh {
        let row = &img[(SCREEN_STEP * y) * w * cn..(SCREEN_STEP * y + 1) * w * cn];
        plane.extend(
            row.chunks_exact(cn)
                .step_by(SCREEN_STEP)
                .take(sw)
                .map(|p| p[..channels].iter().sum::<f32>() / channels as f32),
        );
    }
    let Some(q) = np_percentiles(&plane, &[99.5]) else {
        return false;
    };
    let level = (HOLE_LEVEL as f32) * q[0];
    let mask: Vec<u8> = plane.iter().map(|&v| u8::from(v > level)).collect();
    let c = connected_components(&mask, sh, sw, 8);
    let frame = (sh * sw) as f64;
    let shapes = (1..c.count).filter(|&i| {
        let [_, _, bw, bh, area] = c.stats[i];
        let (bw, bh, area) = (bw as f64, bh as f64, area as f64);
        let aspect = bw.min(bh) / bw.max(bh);
        HOLE_AREA.0 * SCREEN_SLACK * frame < area
            && area < HOLE_AREA.1 / SCREEN_SLACK * frame
            && aspect > HOLE_ASPECT.0 * SCREEN_SLACK
            && area / (bw * bh) > HOLE_FILL * SCREEN_SLACK
    });
    shapes.take(FIRST_ROW_HOLES).count() == FIRST_ROW_HOLES
}

fn frame_on(
    lum: &[f32],
    h: usize,
    w: usize,
    format: &Format,
    slide: bool,
) -> Option<Option<StripFrame>> {
    let g = (format.read)(lum, h, w)?;
    let frame = format.frames.iter().find_map(|spec| {
        let (y0, y1, l, r) = find_frame(&g, h, w, spec, slide)?;
        let (y0, y1) = (y0 as f64, y1 as f64);
        let (l, r) = (l as f64, r as f64);
        let half_gap = spec.gap / 2.0 * g.px_per_mm;
        let half_film = spec.film_across / 2.0 * g.px_per_mm;
        let film = (
            g.center - half_film,
            g.center + half_film,
            l - half_gap,
            r + half_gap,
        );
        Some(StripFrame {
            roi: clamp(inside((y0, y1, l, r), h, w, g.angle), h, w),
            film: clamp(around(film, h, w, g.angle), h, w),
            angle: g.angle,
        })
    });
    Some(frame)
}

/// The picture in a scan that shows the strip's own marks, or None to leave the frame to the
/// film-box path. A strip found with no valid frame is final: the other axis is not tried.
pub fn find_strip_frame(
    img: &[f32],
    h: usize,
    w: usize,
    cn: usize,
    slide: bool,
) -> Option<StripFrame> {
    if h < 16 || w < 16 || cn == 0 || img.len() != h * w * cn || !may_show_holes(img, h, w, cn) {
        return None;
    }
    let lum = mean_plane(img, h, w, cn);
    for format in FORMATS {
        if let Some(found) = frame_on(&lum, h, w, format, slide) {
            return found.filter(usable);
        }
        let turned = quarter_turn(&lum, h, w);
        if let Some(found) = frame_on(&turned, w, h, format, slide) {
            return found
                .map(|f| StripFrame {
                    roi: from_quarter(f.roi, w),
                    film: from_quarter(f.film, w),
                    angle: f.angle,
                })
                .filter(usable);
        }
    }
    None
}

fn usable(f: &StripFrame) -> bool {
    f.roi.1 > f.roi.0 && f.roi.3 > f.roi.2
}

#[cfg(test)]
mod tests {
    use super::*;

    const H: usize = 1000;
    const W: usize = 1500;
    const PITCH: usize = 150;
    const MM: f64 = PITCH as f64 / SPROCKET_PITCH_MM;

    /// A strip across the scan: a dark picture on a thin rebate, with holes in both rails.
    fn strip(gutter: bool) -> Vec<f32> {
        let mut img = vec![0.45f32; H * W];
        let fw = (36.0 * MM) as usize;
        let (x0, y0) = ((W - fw) / 2, 500 - (12.15 * MM) as usize);
        let y1 = 1000 - y0;
        for y in y0..y1 {
            for x in x0..x0 + fw {
                img[y * W + x] = 0.1;
            }
        }
        if gutter {
            for y in y0..y1 {
                for x in 735..765 {
                    img[y * W + x] = 0.45;
                }
            }
        }
        for row in [500 - (14.0 * MM) as usize, 500 + (14.0 * MM) as usize] {
            let mut x = 40;
            while x + 50 < W {
                for y in row - 45..row + 45 {
                    for c in x..x + 60 {
                        img[y * W + c] = 1.0;
                    }
                }
                x += PITCH;
            }
        }
        img
    }

    #[test]
    fn finds_the_picture_between_the_rails() {
        let f = find_strip_frame(&strip(false), H, W, 1, false).expect("a frame");
        let fw = 36.0 * MM;
        assert!(((f.roi.3 - f.roi.2) as f64 - fw).abs() < 6.0);
        assert!(f.angle.abs() < 0.01);
    }

    #[test]
    fn a_quarter_turned_strip_returns_input_coordinates() {
        let lum = strip(false);
        let flat = find_strip_frame(&lum, H, W, 1, false).expect("a frame");
        let turned = quarter_turn(&lum, H, W);
        let f = find_strip_frame(&turned, W, H, 1, false).expect("a frame");
        assert!(((f.roi.1 - f.roi.0) - (flat.roi.3 - flat.roi.2)).abs() <= 2);
        assert!(((f.roi.3 - f.roi.2) - (flat.roi.1 - flat.roi.0)).abs() <= 2);
    }

    #[test]
    fn two_half_frames_step_aside() {
        assert!(find_strip_frame(&strip(true), H, W, 1, false).is_none());
    }

    #[test]
    fn unevenly_spaced_shapes_are_not_sprockets() {
        let hole = |x| Hole { x, y: 0.0 };
        let row = |xs: &[f64]| xs.iter().map(|&x| hole(x)).collect::<Vec<_>>();
        assert!(evenly_spaced(&row(&[0.0, 150.0, 300.0, 450.0, 600.0])));
        assert!(!evenly_spaced(&row(&[
            0.0, 150.0, 240.0, 450.0, 600.0, 700.0, 900.0
        ])));
    }

    #[test]
    fn a_tilted_frame_gets_a_rectangle_inside_it() {
        let inner = inside((100.0, 900.0, 100.0, 1400.0), H, W, 2.0);
        assert!(inner.0 > 100 && inner.1 < 900 && inner.2 > 100 && inner.3 < 1400);
        assert_eq!(
            inside((100.0, 900.0, 100.0, 1400.0), H, W, 0.0),
            (100, 900, 100, 1400)
        );
    }
}
