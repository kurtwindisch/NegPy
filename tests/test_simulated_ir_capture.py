"""Capture IR end to end on the simulated rig: calibrate, scan a triplet with its IR frame, load it."""

import os

import numpy as np
import pytest

from negpy.infrastructure.capture.base import CaptureSettings, IrExposure
from negpy.infrastructure.capture.gphoto import GphotoCamera
from negpy.infrastructure.capture.raw_demosaic import linear_demosaic
from negpy.infrastructure.capture.scanlight import Scanlight
from negpy.infrastructure.loaders.constants import is_ir_sidecar_path
from negpy.infrastructure.loaders.rawpy_loader import RawpyLoader
from negpy.infrastructure.simulated.gphoto import _SHUTTERS, SimGphoto
from negpy.services.capture.calibration import CalibrationService, Roi, scan_ladder, target_signal
from negpy.services.capture.service import CaptureService
from negpy.services.rendering.lens import source_ir

_REBATE = Roi(0.3, 0.07, 0.4, 0.04)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setenv("NEGPY_SIMULATE_HARDWARE", "1")
    cam = GphotoCamera(gp_module=SimGphoto(), jpeg_path=str(tmp_path / "live.jpg"), settings_path=str(tmp_path / "live.json"))
    cam.open()
    light = Scanlight()
    yield light, cam
    light.close()
    cam.close()


def test_calibrated_ir_frame_lands_beside_the_triplet_and_loads_registered(rig, tmp_path):
    light, cam = rig
    calibration = CalibrationService(light, cam, lambda p: linear_demosaic(p, half_size=True), settle_s=0.0)
    result = calibration.calibrate(_REBATE, str(tmp_path / "cal" / "c.raw"), candidates=scan_ladder(_SHUTTERS), ir=True)
    assert result.ir is not None and not result.ir_error
    assert result.ir.signal == pytest.approx(target_signal(), rel=0.2)

    out = tmp_path / "roll"
    capture = CaptureService(light, cam, sleep=lambda _s: None)
    shot = capture.capture_triplet(
        CaptureSettings(
            roll_name="Roll001",
            frame_number=1,
            output_folder=str(out),
            levels=result.levels,
            settle_s=0.0,
            shutters=result.shutters,
            min_raw_bytes=1024,
            ir=IrExposure(result.ir.level, result.ir.shutter),
        )
    )
    assert os.path.basename(shot.ir_path).startswith("Roll001_Frame001_R_IR.")
    assert is_ir_sidecar_path(shot.ir_path)

    ctx, metadata = RawpyLoader().load(shot.red_path)
    ctx.close()
    red = linear_demosaic(shot.red_path)[..., 0]
    ir = source_ir(metadata, red.shape)
    assert ir is not None and ir.shape == red.shape
    # The dust the IR frame sees sits where the red exposure is dark too: one film frame, registered.
    h, w = red.shape
    film = np.zeros(red.shape, bool)
    film[h // 8 : -h // 8, w // 8 : -w // 8] = True  # inside the holder
    dust = film & (ir < 0.3 * np.median(ir[film]))
    assert dust.any()
    assert np.median(red[dust]) < 0.3 * np.median(red[film])
