"""A camera scan's IR frame beside its primary RAW: lookup, discovery hiding, and decode."""

import os

import numpy as np
import pytest

from negpy.infrastructure.loaders.constants import is_ir_sidecar_path
from negpy.infrastructure.loaders.ir_companion import find_ir_companion, ir_companion_paths
from negpy.services.rendering.lens import source_ir


def _touch(tmp_path, *names: str) -> None:
    for name in names:
        (tmp_path / name).write_bytes(b"")


@pytest.mark.parametrize(
    "name,hidden",
    [
        ("Roll001_Frame001_R_IR.ARW", True),  # trichrome: named after the red exposure
        ("Roll001_Frame002_IR.cr3", True),  # single: case-insensitive against Frame002.CR3
        ("Roll001_Frame002_IR_VALID.tif", True),
        ("Roll001_Frame001_IR.ARW", False),  # no Frame001.* main: the trichrome's parts are _R/_G/_B
        ("Roll001_Frame009_IR.ARW", False),  # orphan
        ("Roll001_Frame002_IR.jpg", False),  # an IR companion is linear data, never a JPEG
    ],
)
def test_companion_predicate_covers_camera_raws(tmp_path, name, hidden):
    _touch(
        tmp_path,
        "Roll001_Frame001_R.ARW",
        "Roll001_Frame001_G.ARW",
        "Roll001_Frame001_B.ARW",
        "Roll001_Frame002.CR3",
        name,
    )
    assert is_ir_sidecar_path(str(tmp_path / name)) is hidden


def test_lookup_prefers_a_tiff_and_lists_every_companion(tmp_path):
    _touch(tmp_path, "f.ARW", "f_IR.ARW", "f_ir.tif", "f_IR_VALID.tif", "f_G.ARW")
    ir, valid = find_ir_companion(str(tmp_path / "f.ARW"))
    assert os.path.basename(ir) == "f_ir.tif" and os.path.basename(valid) == "f_IR_VALID.tif"
    assert sorted(os.path.basename(p) for p in ir_companion_paths(str(tmp_path / "f.ARW"))) == ["f_IR.ARW", "f_IR_VALID.tif", "f_ir.tif"]
    assert find_ir_companion(str(tmp_path / "f_G.ARW")) == (None, None)


def _sim_dng(path, color):
    from negpy.infrastructure.simulated import gphoto, scanlight

    scanlight._color[:] = color
    try:
        path.write_bytes(gphoto.SimGphoto().still_dng())
    finally:
        scanlight._color[:] = [0, 0, 0, 0, 0]


def test_raw_companion_decodes_to_one_plane_scaled_to_the_main_buffer(tmp_path):
    from negpy.infrastructure.loaders.rawpy_loader import RawpyLoader
    from negpy.infrastructure.simulated.gphoto import _SENSOR_HW

    main, ir_path = tmp_path / "Roll_Frame001.dng", tmp_path / "Roll_Frame001_IR.dng"
    _sim_dng(main, [200, 0, 0, 0, 0])
    _sim_dng(ir_path, [0, 0, 0, 0, 250])

    ctx, metadata = RawpyLoader().load(str(main))
    ctx.close()
    ir = metadata["ir"]
    assert metadata["ir_companion"] is True
    assert ir.ndim == 2 and ir.dtype == np.float32 and 0.0 <= ir.min() and ir.max() <= 1.0
    full = source_ir(metadata, (*_SENSOR_HW, 3))
    assert full.shape == _SENSOR_HW
    # Dust reads dark in IR against clear film.
    assert np.percentile(full, 0.05) < 0.5 * np.median(full)


def test_another_carriers_mismatched_plane_is_dropped():
    metadata = {"ir": np.ones((10, 10), np.float32), "orientation": 1}
    assert source_ir(metadata, (20, 20, 3)) is None
    assert source_ir({**metadata, "ir_companion": True}, (20, 20, 3)).shape == (20, 20)
