"""Trichromatic R/G/B triplet capture orchestration.

Sets the Scanlight to one narrowband channel, fires the camera, repeats for red,
green, blue, optionally shoots one IR frame — then turns the light off. The three RAWs are handed to NegPy's
RGB-Scan mode, which aligns/merges/inverts them. Hardware is injected, so the
whole flow is unit-testable with fakes.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
from typing import Callable, Optional

from negpy.infrastructure.capture.base import (
    CAPTURE_ORDER,
    Camera,
    CaptureResult,
    CaptureSettings,
    Channel,
    IrExposure,
    LightSource,
)
from negpy.infrastructure.loaders.ir_companion import ir_companion_paths
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

#: Placeholder suffix on the path handed to `Camera.capture`. The camera replaces it with
#: its own (.ARW, .CR3, .NEF) and returns where the file actually landed, so this is only
#: ever a stem separator.
_RAW_SUFFIX = ".raw"

#: Stem suffix of the IR frame, after the primary exposure's stem (`Frame007_R_IR`, `Frame007_IR`).
IR_SUFFIX = "_IR"
#: The `on_channel` letter reported for the IR frame.
IR_CHANNEL = "IR"

ProgressCb = Callable[[float], None]


class CaptureError(RuntimeError):
    """Raised when a triplet capture cannot complete."""


def verify_raw_size(path: str, min_bytes: int, max_bytes: int) -> None:
    """Raise CaptureError if the captured RAW is missing or implausibly sized."""
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise CaptureError(f"captured file is missing: {path}") from exc
    if not min_bytes <= size <= max_bytes:
        lo, hi = min_bytes // 1024 // 1024, max_bytes // 1024 // 1024
        raise CaptureError(
            f"{os.path.basename(path)} is {size / 1024 / 1024:.1f} MB, outside the plausible "
            f"RAW range ({lo}-{hi} MB) — did the film move or the camera misfire?"
        )


Shot = Callable[[str], str]


def _capture_staged(output_folder: str, frame_name: str, shots: list[Shot], cancel: Optional[threading.Event] = None) -> list[str]:
    """Run `shots` into a staging dir beside the destination, then promote every file at once.

    Each shot takes the staged frame stem and returns the file the camera wrote. Nothing reaches
    `output_folder` unless every shot succeeded, so a frame never mixes old and new exposures.
    """
    staging_dir = tempfile.mkdtemp(prefix=f".{frame_name}-", suffix=".capture", dir=output_folder)
    try:
        staged_stem = os.path.join(staging_dir, frame_name)
        staged = []
        for shot in shots:
            if cancel is not None and cancel.is_set():
                raise CaptureError("capture cancelled")
            staged.append(shot(staged_stem))
        if cancel is not None and cancel.is_set():
            raise CaptureError("capture cancelled")
        return _promote_files(staged, output_folder)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def _drop_stale_ir(primary_path: str, ir_path: Optional[str]) -> None:
    """Remove IR companions of `primary_path` other than `ir_path`, so a retake never pairs
    a new color exposure with an old IR frame."""
    for path in ir_companion_paths(primary_path):
        if ir_path is not None and os.path.normcase(path) == os.path.normcase(ir_path):
            continue
        try:
            os.remove(path)
            logger.info("removed stale IR companion %s", path)
        except OSError:
            logger.exception("could not remove stale IR companion %s", path)


def _capture_validated_single(
    camera: Camera,
    *,
    final_stem: str,
    shutter: Optional[str],
    min_raw_bytes: int,
    max_raw_bytes: int,
    cancel: Optional[threading.Event] = None,
    iso: Optional[str] = None,
    aperture: Optional[str] = None,
) -> str:
    """Capture beside the destination and replace it only after size validation."""

    def shoot(stem: str) -> str:
        path = camera.capture(stem + _RAW_SUFFIX, shutter=shutter, iso=iso, aperture=aperture)
        verify_raw_size(path, min_raw_bytes, max_raw_bytes)
        return path

    (path,) = _capture_staged(os.path.dirname(final_stem), os.path.basename(final_stem), [shoot], cancel)
    _drop_stale_ir(path, None)
    return path


def capture_single(
    camera: Camera,
    *,
    roll_name: str,
    frame_number: int,
    output_folder: str,
    shutter: Optional[str] = None,
    min_raw_bytes: int = 8 * 1024 * 1024,
    max_raw_bytes: int = 200 * 1024 * 1024,
    cancel: Optional[threading.Event] = None,
) -> str:
    """One camera exposure with NO light control — for normal white-light scanning without the
    Scanlight (the operator's own high-CRI light stays on). One file, no R/G/B split; imported
    into NegPy as an ordinary single RAW (no triplet merge)."""
    os.makedirs(output_folder, exist_ok=True)
    stem = os.path.join(output_folder, f"{roll_name}_Frame{frame_number:03d}")
    logger.info("capturing single (white-light) frame → %s", stem)
    return _capture_validated_single(
        camera,
        final_stem=stem,
        shutter=shutter,
        min_raw_bytes=min_raw_bytes,
        max_raw_bytes=max_raw_bytes,
        cancel=cancel,
    )


def _promote_files(staged_paths: list[str], output_folder: str) -> list[str]:
    """Replace a frame's files as one recoverable operation.

    All captures have already passed validation before this runs. Existing files are
    backed up inside the staging directory so an I/O failure during promotion can roll
    every file back to the prior complete frame.
    """
    final_paths = [os.path.join(output_folder, os.path.basename(path)) for path in staged_paths]
    staging_dir = os.path.dirname(staged_paths[0])
    backups: dict[int, str] = {}
    promoted: list[int] = []

    try:
        for i, final_path in enumerate(final_paths):
            if os.path.exists(final_path):
                backup_path = os.path.join(staging_dir, f".previous-{os.path.basename(final_path)}")
                try:
                    os.link(final_path, backup_path)
                except OSError:
                    shutil.copy2(final_path, backup_path)
                backups[i] = backup_path

        for i, staged in enumerate(staged_paths):
            os.replace(staged, final_paths[i])
            promoted.append(i)
    except OSError as exc:
        for i in reversed(promoted):
            final_path = final_paths[i]
            backup_path = backups.get(i)
            try:
                if backup_path is not None and os.path.exists(backup_path):
                    os.replace(backup_path, final_path)
                else:
                    os.remove(final_path)
            except OSError:
                logger.exception("could not roll back %s after promotion failure", os.path.basename(final_path))
        raise CaptureError(f"could not promote completed frame: {exc}") from exc

    return final_paths


class CaptureService:
    """Orchestrates one R/G/B triplet capture from an injected light + camera."""

    def __init__(
        self,
        light: LightSource,
        camera: Camera,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._light = light
        self._camera = camera
        self._sleep = sleep

    def set_light(self, r: int, g: int, b: int) -> None:
        """Live light control for the RGB sliders (no capture)."""
        self._light.set_color(r=r, g=g, b=b)

    def _ir_shot(self, ir: IrExposure, settle_s: float, iso: Optional[str], aperture: Optional[str], sizes: tuple[int, int]) -> Shot:
        """The IR frame, named after the primary exposure's stem: IR lit alone, then one capture."""

        def shoot(primary_stem: str) -> str:
            self._light.set_color(ir=ir.level)
            self._sleep(settle_s)
            path = self._camera.capture(primary_stem + IR_SUFFIX + _RAW_SUFFIX, shutter=ir.shutter, iso=iso, aperture=aperture)
            verify_raw_size(path, *sizes)
            logger.info("IR frame → %s", os.path.basename(path))
            return path

        return shoot

    def capture_triplet(
        self,
        settings: CaptureSettings,
        progress: Optional[ProgressCb] = None,
        cancel: Optional[threading.Event] = None,
        on_channel: Optional[Callable[[str], None]] = None,
    ) -> CaptureResult:
        """Capture red, green, blue exposures of the current frame, then the IR frame when
        `settings.ir` is set.

        Each channel: light that channel only → settle → capture → size-check.
        The light is always turned off when done (even on error). Raises
        `CaptureError` on cancel or an implausibly sized file.
        """
        os.makedirs(settings.output_folder, exist_ok=True)
        frame_name = f"{settings.roll_name}_Frame{settings.frame_number:03d}"
        level = {Channel.RED: settings.levels[0], Channel.GREEN: settings.levels[1], Channel.BLUE: settings.levels[2]}
        total = len(CAPTURE_ORDER) + (1 if settings.ir is not None else 0)

        def channel_shot(i: int, ch: Channel) -> Shot:
            def shoot(frame_stem: str) -> str:
                if on_channel is not None:
                    on_channel(ch.letter)
                r, g, b = ch.rgb(level[ch])
                _t0 = time.perf_counter()
                self._light.set_color(r=r, g=g, b=b)
                self._sleep(settings.settle_s)
                _t1 = time.perf_counter()
                shutter = settings.shutters[i] if settings.shutters is not None else None
                out_path = self._camera.capture(
                    f"{frame_stem}_{ch.letter}{_RAW_SUFFIX}", shutter=shutter, iso=settings.iso, aperture=settings.aperture
                )
                self._verify_size(out_path, settings)
                # Per-channel timing, so the scan-speed bottleneck is visible in the log: settle is a
                # fixed wait, while capture and download are transport-bound.
                logger.info(
                    "channel %s: settle %.0f ms, capture+download %.0f ms → %s",
                    ch.letter,
                    (_t1 - _t0) * 1000,
                    (time.perf_counter() - _t1) * 1000,
                    os.path.basename(out_path),
                )
                if progress is not None:
                    progress((i + 1) / total)
                if i < total - 1:
                    self._sleep(settings.inter_exposure_delay_s)
                return out_path

            return shoot

        shots = [channel_shot(i, ch) for i, ch in enumerate(CAPTURE_ORDER)]
        if settings.ir is not None:
            ir_shoot = self._ir_shot(
                settings.ir, settings.settle_s, settings.iso, settings.aperture, (settings.min_raw_bytes, settings.max_raw_bytes)
            )

            def ir_after_red(frame_stem: str) -> str:
                if on_channel is not None:
                    on_channel(IR_CHANNEL)
                path = ir_shoot(f"{frame_stem}_{CAPTURE_ORDER[0].letter}")
                if progress is not None:
                    progress(1.0)
                return path

            shots.append(ir_after_red)

        try:
            paths = _capture_staged(settings.output_folder, frame_name, shots, cancel)
        finally:
            self._light_off()

        red, green, blue = paths[:3]
        ir_path = paths[3] if settings.ir is not None else None
        _drop_stale_ir(red, ir_path)
        return CaptureResult(frame_number=settings.frame_number, red_path=red, green_path=green, blue_path=blue, ir_path=ir_path)

    def capture_rgb_single(self, settings: CaptureSettings, cancel: Optional[threading.Event] = None) -> str:
        """One exposure with R, G and B lit together at the preset's levels (one file, no
        R/G/B split), then the IR frame when `settings.ir` is set. The shared shutter and the
        preset's ISO and aperture are forced, as for a triplet."""
        r, g, b = settings.levels
        frame_name = f"{settings.roll_name}_Frame{settings.frame_number:03d}"
        logger.info("capturing single RGB frame → %s", frame_name)
        return self._capture_one_file(
            output_folder=settings.output_folder,
            frame_name=frame_name,
            light=lambda: self._light.set_color(r=r, g=g, b=b),
            settle_s=settings.settle_s,
            shutter=settings.shutters[0] if settings.shutters is not None else None,
            iso=settings.iso,
            aperture=settings.aperture,
            ir=settings.ir,
            sizes=(settings.min_raw_bytes, settings.max_raw_bytes),
            cancel=cancel,
        )

    def capture_white(
        self,
        *,
        roll_name: str,
        frame_number: int,
        output_folder: str,
        w_level: int,
        shutter: Optional[str] = None,
        settle_s: float = 0.4,
        min_raw_bytes: int = 8 * 1024 * 1024,
        max_raw_bytes: int = 200 * 1024 * 1024,
        cancel: Optional[threading.Event] = None,
        ir: Optional[IrExposure] = None,
    ) -> str:
        """Single white-light exposure for slide / E-6 film (one file, no R/G/B split), then the
        IR frame when `ir` is set."""
        frame_name = f"{roll_name}_Frame{frame_number:03d}"
        logger.info("capturing white frame → %s", frame_name)
        return self._capture_one_file(
            output_folder=output_folder,
            frame_name=frame_name,
            light=lambda: self._light.set_color(w=w_level),
            settle_s=settle_s,
            shutter=shutter,
            iso=None,
            aperture=None,
            ir=ir,
            sizes=(min_raw_bytes, max_raw_bytes),
            cancel=cancel,
        )

    def _capture_one_file(
        self,
        *,
        output_folder: str,
        frame_name: str,
        light: Callable[[], None],
        settle_s: float,
        shutter: Optional[str],
        iso: Optional[str],
        aperture: Optional[str],
        ir: Optional[IrExposure],
        sizes: tuple[int, int],
        cancel: Optional[threading.Event],
    ) -> str:
        os.makedirs(output_folder, exist_ok=True)

        def shoot(stem: str) -> str:
            light()
            self._sleep(settle_s)
            path = self._camera.capture(stem + _RAW_SUFFIX, shutter=shutter, iso=iso, aperture=aperture)
            verify_raw_size(path, *sizes)
            return path

        shots = [shoot]
        if ir is not None:
            shots.append(self._ir_shot(ir, settle_s, iso, aperture, sizes))
        try:
            paths = _capture_staged(output_folder, frame_name, shots, cancel)
        finally:
            self._light_off()
        _drop_stale_ir(paths[0], paths[1] if ir is not None else None)
        return paths[0]

    def _light_off(self) -> None:
        try:
            self._light.off()
        except Exception:
            logger.exception("failed to turn the Scanlight off after capture")

    def _verify_size(self, path: str, settings: CaptureSettings) -> None:
        verify_raw_size(path, settings.min_raw_bytes, settings.max_raw_bytes)
