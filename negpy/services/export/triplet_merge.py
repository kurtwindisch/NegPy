"""Merge a Trichrome triplet into one linear RGB TIFF that replaces its three exposures.

The buffer comes from the render decode (`ImageProcessor._decode_oriented_f32`), not
Linear Output's own decode, so the file holds exactly the source the pipeline inverted
for the triplet and the frame's edit carries over unchanged. Nothing that the edit
applies itself is baked: no flat-field, no user geometry.
"""

import hashlib
import os
import tempfile
from dataclasses import replace
from typing import Optional

import numpy as np
import tifffile

from negpy.domain.models import TiffCompression, WorkspaceConfig
from negpy.features.flatfield.models import FlatFieldConfig
from negpy.features.hdr.models import HdrConfig
from negpy.features.metadata.fsdate import sync_export_filesystem_dates
from negpy.features.rgbscan.models import is_rgb_triplet
from negpy.features.stitch.models import StitchConfig
from negpy.kernel.image.logic import _to_uint16_jit
from negpy.services.export.linear_output import (
    _is_camera_raw,
    _linear_resolution,
    _read_source_meta_tiff,
    _write_tiff,
)

MERGED_SUFFIX = "_RGB"
MERGED_EXT = ".tif"
_MERGED_FORMAT = "camera RAW (RGB triplet)"


class MergeVerifyError(RuntimeError):
    """The file on disk does not hold the buffer that was written."""


def can_merge(red_path: str, green_path: str, blue_path: str) -> bool:
    """Whether a triplet is camera RAW with all three exposures on disk."""
    return all(os.path.exists(p) for p in (red_path, green_path, blue_path)) and _is_camera_raw(red_path)


def merged_path_for(red_path: str, taken: frozenset = frozenset()) -> str:
    """`<red stem>_RGB.tif` next to the red exposure, numbered past any existing file."""
    folder = os.path.dirname(red_path)
    stem = os.path.splitext(os.path.basename(red_path))[0] + MERGED_SUFFIX
    path = os.path.join(folder, stem + MERGED_EXT)
    counter = 2
    while path in taken or os.path.exists(path):
        path = os.path.join(folder, f"{stem}_{counter}{MERGED_EXT}")
        counter += 1
    return path


def decode_params(params: WorkspaceConfig) -> WorkspaceConfig:
    """The frame's params with every correction its edit re-applies taken out of the decode."""
    if not is_rgb_triplet(params.rgbscan):
        raise ValueError("Not a Trichrome triplet")
    return replace(params, flatfield=FlatFieldConfig(), stitch=StitchConfig(), hdr=HdrConfig())


def is_merged_triplet(path: str) -> bool:
    """Whether *path* is a TIFF NegPy wrote from a triplet, which never takes part in grouping."""
    if not path.lower().endswith((".tif", ".tiff")):
        return False
    try:
        with tifffile.TiffFile(path) as tif:
            description = tif.pages[0].description or ""
    except Exception:
        return False
    return description.startswith("NegPy Linear Output") and _MERGED_FORMAT in description


def _digest(u16: np.ndarray) -> str:
    return hashlib.blake2b(np.ascontiguousarray(u16).tobytes(), digest_size=16).hexdigest()


def write_merged_triplet(
    f32: np.ndarray,
    red_path: str,
    out_path: str,
    compression: TiffCompression = TiffCompression.ZIP,
) -> None:
    """Write *f32* as an untagged 16-bit TIFF at *out_path* and prove it reads back identical.

    The file is written and verified under a temporary name, then renamed, so *out_path*
    exists only when it holds the whole buffer. An existing *out_path* is never replaced.
    """
    expected = _digest(_to_uint16_jit(np.ascontiguousarray(f32, dtype=np.float32)))
    folder = os.path.dirname(out_path) or "."
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(dir=folder, delete=False, suffix=".part") as tmp:
            tmp_path = tmp.name
            _write_tiff(
                f32,
                tmp,
                os.path.basename(red_path),
                source_path=red_path,
                source_meta=_read_source_meta_tiff(red_path),
                source_format=_MERGED_FORMAT,
                resolution=_linear_resolution(red_path),
                compression=compression,
            )
        written = tifffile.imread(tmp_path)
        if written.dtype != np.uint16 or written.shape != f32.shape or _digest(written) != expected:
            raise MergeVerifyError(f"{os.path.basename(out_path)} did not read back as written")
        if os.path.exists(out_path):
            raise FileExistsError(out_path)
        os.replace(tmp_path, out_path)
        tmp_path = None
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.unlink(tmp_path)
    sync_export_filesystem_dates(out_path, red_path)
