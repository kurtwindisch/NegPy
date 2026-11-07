"""Auto Crop in a full film scan: a strip's own marks locate the picture, and a frame they cannot
place stays on the film-box path."""

from dataclasses import replace
from unittest.mock import patch

import cv2
import negpy_core
import numpy as np
import pytest

from negpy.domain.models import WorkspaceConfig
from negpy.features.geometry.batch_autocrop import detect_crop_candidate, resolve_roll_crops
from negpy.features.geometry.logic import (
    _detect_strip_frame,
    apply_fine_rotation,
    autocrop_detection_key,
    get_autocrop_coords,
    resolve_meter_rect,
    strip_tilt,
)
from negpy.features.geometry.models import AutocropMode, straightens_on_arm
from negpy.features.process.models import ProcessMode, with_process_mode
from negpy.features.stitch.models import StitchConfig
from negpy.services.rendering.image_processor import _resolve_armed_autocrop, strip_slide_of

H, W, PITCH = 1200, 1800, 150
MM = PITCH / 4.75
FRAME_W, FRAME_H = 36 * MM, 24.3 * MM
LEFT = int((W - FRAME_W) / 2)
RIGHT = int(LEFT + FRAME_W)
TOP = int(H / 2 - FRAME_H / 2)
BOTTOM = int(TOP + FRAME_H)


def _texture(seed: int, shape: tuple) -> np.ndarray:
    n = cv2.GaussianBlur(np.random.default_rng(seed).random(shape).astype(np.float32), (0, 0), 14)
    return (n - n.min()) / (n.max() - n.min())


def _strip(
    *, slide=False, surround=1.0, hole_w=2.0, tilt=0.0, gutter=0.0, wall=False, thin=False, hole_xs=None, fade=None, no_gap=False
) -> np.ndarray:
    """A strip across the scan. A negative has a thin rebate and a denser picture; a slide the reverse."""
    rebate, low, high = (0.012, 0.15, 0.6) if slide else (0.55, 0.08, 0.3)
    if thin:
        low, high = 0.40, 0.46
    img = np.full((H, W), surround, np.float32)
    film_top, film_bottom = int(H / 2 - 17.5 * MM), int(H / 2 + 17.5 * MM)
    img[film_top:film_bottom] = rebate
    for i in (-1, 0, 1):
        x0 = LEFT + i * int(38 * MM)
        a, b = max(0, x0), min(W, x0 + int(FRAME_W))
        img[TOP:BOTTOM, a:b] = (low + (high - low) * _texture(i + 2, (H, W)))[TOP:BOTTOM, a:b]
    if fade == "right":
        img[TOP:BOTTOM, LEFT + int(0.55 * FRAME_W) : RIGHT] = rebate
    if fade == "left":
        img[TOP:BOTTOM, LEFT : LEFT + int(0.45 * FRAME_W)] = rebate
    if no_gap:
        img[TOP:BOTTOM, RIGHT : RIGHT + int(4 * MM)] = (low + (high - low) * _texture(9, (H, W)))[TOP:BOTTOM, RIGHT : RIGHT + int(4 * MM)]
    if wall:
        img[TOP + 10 : BOTTOM - 120, LEFT + 70 : RIGHT - 70] = 0.1
    if gutter:
        half = int(gutter * MM / 2)
        img[TOP:BOTTOM, W // 2 - half : W // 2 + half] = rebate
    for centre in (H / 2 - 14 * MM, H / 2 + 14 * MM):
        half_h, half_w = int(1.4 * MM), int(hole_w / 2 * MM)
        for x in np.arange(40, W - 60, PITCH) if hole_xs is None else hole_xs:
            y = int(centre)
            img[y - half_h : y + half_h, int(x) - half_w : int(x) + half_w] = 1.0
    img = np.stack([img] * 3, axis=-1).astype(np.float32)
    return np.asarray(apply_fine_rotation(img, -tilt)) if tilt else img


def _masked(img: np.ndarray, *, show: float, soft: float = 0.0, slant: float = 0.0, value: float = 0.004) -> np.ndarray:
    """A holder over the top rail that shows `show` of each hole, and over the bottom rail."""
    row = H / 2 - 14 * MM
    cut = row + 1.4 * MM - show * 2.8 * MM
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    edge = cut + np.tan(np.radians(slant)) * (x - W / 2)
    cover = np.maximum(y < edge, y > BOTTOM + 15).astype(np.float32)
    if soft:
        cover = cv2.GaussianBlur(cover, (0, 0), soft)
    return (img * (1 - cover[..., None]) + value * cover[..., None]).astype(np.float32)


def _near(roi, expected, slack=8):
    return all(abs(a - b) <= slack for a, b in zip(roi, expected))


@pytest.mark.parametrize("hole_w", [2.0, 1.87], ids=["still", "cine"])
@pytest.mark.parametrize("surround", [1.0, 0.004], ids=["white", "black"])
def test_the_picture_is_found_between_the_rails(hole_w, surround):
    found = _detect_strip_frame(_strip(hole_w=hole_w, surround=surround), False)
    assert found is not None and _near(found.roi, (TOP, BOTTOM, LEFT, RIGHT))


def test_a_slide_reads_the_black_rebate():
    found = _detect_strip_frame(_strip(slide=True), True)
    assert found is not None and _near(found.roi, (TOP, BOTTOM, LEFT, RIGHT))


def test_the_tilt_is_the_rotation_that_straightens_the_strip():
    img = _strip(tilt=2.5)
    found = _detect_strip_frame(img, False)
    assert found is not None and found.angle == pytest.approx(2.5, abs=0.05)
    straight = _detect_strip_frame(np.asarray(apply_fine_rotation(img, found.angle)), False)
    assert straight is not None and abs(straight.angle) < 0.05


def test_a_tilted_frame_gets_a_crop_inside_the_frame():
    found = _detect_strip_frame(_strip(tilt=2.5), False)
    assert found is not None
    y1, y2, x1, x2 = found.roi
    assert y2 - y1 < FRAME_H - 0.8 * FRAME_W * np.sin(np.radians(2.5))
    assert x2 - x1 < FRAME_W - 0.8 * FRAME_H * np.sin(np.radians(2.5))


def test_a_strip_running_top_to_bottom_returns_input_coordinates():
    flat = _detect_strip_frame(_strip(), False)
    found = _detect_strip_frame(np.ascontiguousarray(np.rot90(_strip())), False)
    assert flat is not None and found is not None
    y1, y2, x1, x2 = flat.roi
    assert _near(found.roi, (W - x2, W - x1, y1, y2))


def _other_format(
    *, frame_w: float, frame_h: float, film_h: float, rows: tuple, seed: int, slide=False, line_at: float = 0.0
) -> np.ndarray:
    """Frames across a scan of another format that shares the 35mm hole pitch. Sizes in mm."""
    rebate, low, high = (0.012, 0.15, 0.6) if slide else (0.55, 0.08, 0.3)
    mm = 1600 / 100
    scan_h, scan_w, half = int(70 * mm), int(100 * mm), frame_h / 2 * mm
    img = np.full((scan_h, scan_w), 1.0, np.float32)
    middle = scan_h / 2
    img[max(0, int(middle - film_h / 2 * mm)) : int(middle + film_h / 2 * mm)] = rebate
    picture = low + (high - low) * _texture(seed, (scan_h, scan_w))
    start = np.random.default_rng(seed).uniform(0, (frame_w + 2.5) * mm)
    for x0 in np.arange(start - scan_w, scan_w, (frame_w + 2.5) * mm):
        a, b = max(0, int(x0)), max(0, int(x0 + frame_w * mm))
        img[int(middle - half) : int(middle + half), a:b] = picture[int(middle - half) : int(middle + half), a:b]
        if line_at:
            img[int(middle - half) : int(middle + half), max(0, int(x0 + line_at * mm)) : b] *= 0.5
    for row in rows:
        y = int(middle + row * mm)
        for x in np.arange(2.0, 98.0, 4.75):
            img[y - int(1.4 * mm) : y + int(1.4 * mm), int((x - 1) * mm) : int((x + 1) * mm)] = 1.0
    return np.stack([img] * 3, axis=-1).astype(np.float32)


@pytest.mark.parametrize("slide", [False, True])
@pytest.mark.parametrize("frame_w", [42.0, 56.0, 70.0])
@pytest.mark.parametrize("seed", range(6))
def test_a_larger_format_with_one_row_steps_aside(frame_w, seed, slide):
    img = _other_format(frame_w=frame_w, frame_h=56.0, film_h=61.5, rows=(29.0,), seed=seed, slide=slide)
    assert _detect_strip_frame(img, slide) is None


@pytest.mark.parametrize("seed", range(6))
def test_a_lone_row_still_places_a_35mm_frame(seed):
    img = _other_format(frame_w=36.0, frame_h=24.3, film_h=35.0, rows=(14.0,), seed=seed)
    found = _detect_strip_frame(img, False)
    assert found is not None
    y1, y2, x1, x2 = found.roi
    assert (y2 - y1) / 16 == pytest.approx(24.3, abs=0.3) and (x2 - x1) / 16 == pytest.approx(36.0, abs=0.5)


@pytest.mark.parametrize("frame_w", [23.0, 24.0])
@pytest.mark.parametrize("seed", range(6))
def test_a_short_frame_between_neighbors_steps_aside(frame_w, seed):
    """A straight scene edge in the neighbor sits where a 36 mm frame would end."""
    line_at = 36.2 - frame_w - 2.5
    img = _other_format(frame_w=frame_w, frame_h=24.0, film_h=35.0, rows=(-14.0, 14.0), seed=seed, line_at=line_at)
    assert _detect_strip_frame(img, False) is None


@pytest.mark.parametrize("gutter", [0.6, 1.0, 2.0])
def test_two_half_frames_step_aside(gutter):
    assert _detect_strip_frame(_strip(gutter=gutter), False) is None


def test_a_row_of_uneven_bright_shapes_is_not_a_sprocket_row():
    """The median gap reads as the true pitch, so only the spacing test stops this frame."""
    gaps = (PITCH, 90, 210, PITCH, 100, 200, PITCH, 95, 205, PITCH)
    assert _detect_strip_frame(_strip(), False) is not None
    assert _detect_strip_frame(_strip(hole_xs=np.cumsum((60, *gaps))), False) is None


def test_a_scene_wall_inside_the_frame_is_not_a_side():
    found = _detect_strip_frame(_strip(wall=True), False)
    assert found is None or _near(found.roi, (TOP, BOTTOM, LEFT, RIGHT))


def test_a_thin_picture_lands_on_the_frame_or_steps_aside():
    found = _detect_strip_frame(_strip(thin=True), False)
    assert found is None or _near(found.roi, (TOP, BOTTOM, LEFT, RIGHT))


@pytest.mark.parametrize("value", [0.004, 1.0], ids=["black", "white"])
@pytest.mark.parametrize("soft", [0.0, 6.0, 14.0])
@pytest.mark.parametrize("slant", [0.0, 0.6])
@pytest.mark.parametrize("show", [0.1, 0.25, 0.5, 0.75])
def test_a_holder_sliver_never_shifts_the_crop(show, slant, soft, value):
    found = _detect_strip_frame(_masked(_strip(), show=show, soft=soft, slant=slant, value=value), False)
    assert found is None or _near(found.roi, (TOP, BOTTOM, LEFT, RIGHT), slack=12)


@pytest.mark.parametrize("fade", ["left", "right"])
def test_a_frame_that_fades_into_the_rebate_is_placed_from_its_one_side(fade):
    found = _detect_strip_frame(_strip(fade=fade), False)
    assert found is not None and _near(found.roi, (TOP, BOTTOM, LEFT, RIGHT), slack=12)


def test_a_placed_side_needs_clean_film_or_an_edge():
    """The right side fades, and picture runs on past where the frame should end."""
    assert _detect_strip_frame(_strip(no_gap=True), False) is None


def test_a_film_with_no_marks_stays_on_the_film_box_path():
    assert _detect_strip_frame(np.full((H, W, 3), 0.3, np.float32), False) is None


def test_auto_crop_only_searches_when_asked():
    with patch("negpy.features.geometry.logic._detect_strip_frame") as find:
        get_autocrop_coords(_strip())
    find.assert_not_called()


def test_film_edge_mode_keeps_its_own_path():
    with patch("negpy.features.geometry.logic._detect_strip_frame") as find:
        get_autocrop_coords(_strip(), mode=AutocropMode.FILM, strip_slide=False)
    find.assert_not_called()


def test_a_full_trim_crops_to_the_picture_and_no_trim_keeps_the_rebate_but_no_neighbor():
    img = _strip()
    y1, y2, x1, x2 = get_autocrop_coords(img, target_ratio_str="Free", strip_slide=False)
    assert x2 - x1 < FRAME_W + 8 and y2 - y1 < FRAME_H + 8
    y1, y2, x1, x2 = get_autocrop_coords(img, target_ratio_str="Free", strip_slide=False, rebate_trim=0.0)
    assert x1 > LEFT - 2.5 * MM and x2 < RIGHT + 2.5 * MM
    assert y2 - y1 > FRAME_H + 5 * MM


def test_a_strip_frame_counts_as_a_film_box_for_the_meters():
    img = _strip()
    box = resolve_meter_rect(img, WorkspaceConfig().geometry, 1600.0, strip_slide=False)
    assert box is not None
    assert box[0] == pytest.approx(LEFT / W, abs=0.01) and box[2] == pytest.approx(RIGHT / W, abs=0.01)


def _armed(**geometry) -> WorkspaceConfig:
    base = WorkspaceConfig()
    return replace(base, geometry=replace(base.geometry, crop_from_auto=True, crop_rect=None, **geometry))


def test_an_arm_straightens_the_strip_and_crops_the_whole_frame():
    settings, resolved = _resolve_armed_autocrop(_strip(tilt=2.0), _armed())
    assert settings.geometry.fine_rotation == pytest.approx(2.0, abs=0.05)
    assert settings.geometry.autocrop_squared
    assert resolved is not None and resolved.squared and resolved.skew[0] == settings.geometry.fine_rotation
    assert resolved.key == autocrop_detection_key(settings.geometry)
    x1, y1, x2, y2 = resolved.rect
    assert (x2 - x1) * W > 0.98 * FRAME_W and (y2 - y1) * H > 0.9 * FRAME_H


def test_a_reset_rotation_stays_reset():
    """The arm is marked, so the crop re-detects on the tilted frame and sits inside it."""
    settings, resolved = _resolve_armed_autocrop(_strip(tilt=2.0), _armed(autocrop_squared=True))
    assert settings.geometry.fine_rotation == 0.0
    assert resolved is not None and resolved.skew is None and resolved.rect is not None


def test_a_frame_with_its_own_rotation_keeps_it():
    settings, resolved = _resolve_armed_autocrop(_strip(tilt=2.0), _armed(fine_rotation=0.5))
    assert settings.geometry.fine_rotation == 0.5
    assert resolved is not None and resolved.skew is None


def test_a_tilt_past_the_limit_is_not_straightened():
    with patch("negpy.services.rendering.image_processor.strip_tilt", return_value=9.0):
        settings, resolved = _resolve_armed_autocrop(_strip(), _armed())
    assert settings.geometry.fine_rotation == 0.0
    assert resolved is not None and resolved.skew is None


def test_square_first_does_not_run_on_a_strip():
    with patch("negpy.services.rendering.image_processor.squared_geometry") as fit:
        _resolve_armed_autocrop(_strip(tilt=1.0), _armed(autocrop_square=True))
    fit.assert_not_called()


def test_square_first_still_runs_with_no_strip():
    img = np.full((H, W, 3), 0.3, np.float32)
    with patch("negpy.services.rendering.image_processor.squared_geometry", return_value=None) as fit:
        _resolve_armed_autocrop(img, _armed(autocrop_square=True))
    fit.assert_called_once()


def test_a_frame_with_no_strip_searches_once():
    img = np.full((H, W, 3), 0.3, np.float32)
    with patch("negpy.features.geometry.logic._detect_strip_frame", return_value=None) as find:
        _resolve_armed_autocrop(img, _armed())
    find.assert_called_once()


def test_the_arm_mark_blocks_a_second_rotation():
    g = _armed().geometry
    assert straightens_on_arm(g)
    assert not straightens_on_arm(replace(g, autocrop_squared=True))
    assert not straightens_on_arm(replace(g, fine_rotation=0.3))
    assert not straightens_on_arm(replace(g, autocrop_mode=AutocropMode.FILM))


def test_the_film_mode_sets_the_polarity():
    config = WorkspaceConfig()
    assert strip_slide_of(config) is False
    assert strip_slide_of(with_process_mode(config, ProcessMode.E6)) is True


def test_a_stitch_skips_the_search():
    config = replace(WorkspaceConfig(), stitch=StitchConfig(stitch_enabled=True, stitch_paths=("a", "b")))
    assert strip_slide_of(config) is None


@pytest.mark.parametrize("portrait", [False, True])
def test_crop_all_frames_gives_a_strip_frame_its_own_crop(portrait):
    img = _strip(tilt=1.5)
    img = np.ascontiguousarray(np.rot90(img)) if portrait else img
    evidence = detect_crop_candidate("a", img, target_ratio="3:2", strip_slide=False)
    assert evidence.reason == "strip_frame" and evidence.roi is None
    assert evidence.correction_angle == pytest.approx(1.5, abs=0.05)
    crops = resolve_roll_crops([evidence])
    assert len(crops) == 1 and crops[0].straightened and crops[0].correction_angle == evidence.correction_angle


def test_crop_all_frames_leaves_a_frames_own_skew_alone():
    evidence = detect_crop_candidate("a", _strip(tilt=1.5), target_ratio="3:2", strip_slide=False, own_skew=True)
    assert evidence.correction_angle == 0.0
    assert evidence.reason == "strip_frame"


def test_a_strip_frame_stays_out_of_the_roll_template():
    strip = detect_crop_candidate("strip", _strip(), target_ratio="3:2", strip_slide=False)
    with patch("negpy.features.geometry.batch_autocrop.build_roll_template", return_value=None) as build:
        crops = resolve_roll_crops([strip])
    build.assert_called_once_with([])
    assert [c.key for c in crops] == ["strip"]


def test_strip_tilt_is_none_with_no_marks():
    assert strip_tilt(np.full((H, W, 3), 0.3, np.float32), WorkspaceConfig().geometry, False) is None


def test_the_binding_reads_gray_and_color():
    img = _strip()
    gray = negpy_core.find_strip_frame(np.ascontiguousarray(img[..., 0]), False)
    color = negpy_core.find_strip_frame(img, False)
    assert gray is not None and gray == color
