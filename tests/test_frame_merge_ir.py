"""Merge to TIFF keeps the frame's IR as the merged TIFF's `_IR` companion."""

import os

import numpy as np

from negpy.infrastructure.loaders.constants import is_ir_sidecar_path
from negpy.infrastructure.loaders.ir_companion import read_ir_companion
from negpy.services.export.frame_merge import merged_ir_path, write_merged_ir


def test_merged_ir_reads_back_as_the_merged_tiffs_companion(tmp_path):
    out = tmp_path / "Roll_Frame001_R_RGB.tif"
    out.write_bytes(b"")
    ir = np.linspace(0.0, 1.0, 24, dtype=np.float32).reshape(4, 6)

    write_merged_ir(ir, str(out))

    assert os.path.basename(merged_ir_path(str(out))) == "Roll_Frame001_R_RGB_IR.tif"
    assert is_ir_sidecar_path(merged_ir_path(str(out)))
    back, valid = read_ir_companion(str(out))
    assert valid is None
    np.testing.assert_allclose(back, ir, atol=1 / 65535)
    assert sorted(os.listdir(tmp_path)) == ["Roll_Frame001_R_RGB.tif", "Roll_Frame001_R_RGB_IR.tif"]
