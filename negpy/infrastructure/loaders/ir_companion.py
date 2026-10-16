"""IR companion files: `<stem>_IR.<ext>` and an optional `<stem>_IR_VALID.<ext>` beside a main file.

A film scanner writes them as TIFF; a camera scan with an IR-capable Scanlight writes the IR
frame as the camera's RAW, named after the primary (red) exposure's stem.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from negpy.infrastructure.loaders.constants import IR_COMPANION_EXTENSIONS, SUPPORTED_TIFF_EXTENSIONS
from negpy.infrastructure.loaders.ir_planes import normalize_ir_to_float32
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)


def _matches(entries: list[str], stem_lower: str, token: str) -> list[str]:
    """Entries named `stem + token + <companion ext>`, TIFF first."""
    found = []
    for name in entries:
        entry_stem, ext = os.path.splitext(name)
        if entry_stem.lower() == stem_lower + token and ext.lower() in IR_COMPANION_EXTENSIONS:
            found.append(name)
    return sorted(found, key=lambda n: (os.path.splitext(n)[1].lower() not in SUPPORTED_TIFF_EXTENSIONS, n))


def _listing(main_path: str) -> tuple[str, str, list[str]]:
    dirname = os.path.dirname(main_path) or "."
    stem_lower = os.path.basename(os.path.splitext(main_path)[0]).lower()
    try:
        entries = os.listdir(dirname)
    except OSError:
        entries = []
    return dirname, stem_lower, entries


def find_ir_companion(main_path: str) -> tuple[Optional[str], Optional[str]]:
    """(IR path, validity-mask path) beside `main_path`; either is None when absent."""
    dirname, stem_lower, entries = _listing(main_path)
    ir = _matches(entries, stem_lower, "_ir")
    valid = _matches(entries, stem_lower, "_ir_valid")
    return (
        os.path.join(dirname, ir[0]) if ir else None,
        os.path.join(dirname, valid[0]) if valid else None,
    )


def ir_companion_paths(main_path: str) -> list[str]:
    """Every IR companion file of `main_path`, for actions that move or delete the frame's files."""
    dirname, stem_lower, entries = _listing(main_path)
    return [os.path.join(dirname, n) for token in ("_ir", "_ir_valid") for n in _matches(entries, stem_lower, token)]


def _is_tiff(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_TIFF_EXTENSIONS


def _decode_raw_ir(path: str) -> np.ndarray:
    """A camera IR frame as one linear plane: a half-size decode (2x2 binning, no demosaic),
    then the sensor channel that saw the most IR. The caller
    scales it to the main decode's size."""
    import rawpy

    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            gamma=(1, 1),
            no_auto_bright=True,
            adjust_maximum_thr=0.0,
            use_camera_wb=False,
            user_wb=[1, 1, 1, 1],
            output_bps=16,
            output_color=rawpy.ColorSpace.raw,
            user_flip=0,
            half_size=True,
        )
    channel = int(np.argmax(np.median(rgb[::8, ::8].reshape(-1, rgb.shape[-1]), axis=0)))
    return normalize_ir_to_float32(np.ascontiguousarray(rgb[..., channel]))


def _read_valid_mask(path: str, shape: tuple[int, ...]) -> np.ndarray:
    import tifffile

    valid = np.asarray(tifffile.imread(path))
    if valid.shape != shape:
        raise ValueError(f"mask shape {valid.shape} does not match IR shape {shape}")
    if valid.dtype not in (np.dtype(np.bool_), np.dtype(np.uint8)):
        raise ValueError(f"mask dtype {valid.dtype} is not bool or uint8")
    if valid.dtype == np.uint8:
        rows_per_chunk = max(1, (1 << 20) // max(1, valid.shape[1]))
        for start in range(0, valid.shape[0], rows_per_chunk):
            chunk = valid[start : start + rows_per_chunk]
            if not np.all((chunk == 0) | (chunk == 1) | (chunk == 255)):
                raise ValueError("uint8 mask values must be 0, 1, or 255")
    return valid.astype(np.bool_, copy=False)


def read_ir_companion(main_path: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """(IR plane in [0, 1], validity mask) from `main_path`'s companion files, fail-closed.

    Pixels the mask marks invalid read as clean film (1.0). A mask that cannot be read drops
    the IR as well, since its invalid pixels would read as defects.
    """
    ir_path, valid_path = find_ir_companion(main_path)
    if ir_path is None:
        return None, None
    try:
        if _is_tiff(ir_path):
            import tifffile

            ir = normalize_ir_to_float32(np.asarray(tifffile.imread(ir_path)))
        else:
            ir = _decode_raw_ir(ir_path)
    except Exception as e:
        logger.warning(f"Failed to read IR sidecar {ir_path}: {e}")
        return None, None
    if valid_path is None or not _is_tiff(valid_path):
        return ir, None
    try:
        valid = _read_valid_mask(valid_path, ir.shape)
    except Exception as e:
        logger.warning(f"Failed to read IR validity mask {valid_path}; ignoring IR sidecar {ir_path}: {e}")
        return None, None
    ir = ir.copy()
    ir[~valid] = 1.0
    return ir, valid
