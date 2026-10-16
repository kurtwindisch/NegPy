"""IR frame calibration on an IR-converted camera: one level and shutter for the IR LED."""

import os
import threading

import numpy as np
import pytest

from negpy.services.capture.calibration import (
    SHUTTER_CANDIDATES,
    CalibrationExposureError,
    CalibrationResult,
    CalibrationService,
    Roi,
    target_signal,
    true_seconds,
)


class FakeLight:
    def __init__(self):
        self.ir = 0
        self.rgb = (0, 0, 0)

    def set_color(self, r=0, g=0, b=0, w=0, ir=0, save=False):
        assert not (ir and (r or g or b or w))
        self.ir, self.rgb = ir, (r, g, b)

    def off(self):
        self.ir, self.rgb = 0, (0, 0, 0)

    def close(self):
        pass


class FakeCamera:
    def __init__(self):
        self.last_shutter = "0.4"

    def capture(self, out_path, shutter=None, iso=None, aperture=None):
        if shutter:
            self.last_shutter = shutter
        return os.path.splitext(out_path)[0] + ".ARW"

    def close(self):
        pass


_K = np.array([2000.0, 4000.0, 6000.0])  # color response per sensor channel to its own LED


def _service(ir_response):
    """`ir_response`: per sensor channel counts per IR level per second."""
    light, cam = FakeLight(), FakeCamera()

    def demosaic(_path):
        sec = true_seconds(cam.last_shutter, SHUTTER_CANDIDATES)
        img = np.empty((64, 64, 3))
        img[...] = (_K * np.asarray(light.rgb, float) + np.asarray(ir_response, float) * light.ir) * sec
        return np.clip(img, 0, 65535)

    return CalibrationService(light, cam, demosaic, source_clip=lambda *_a: (0.0, 0.0), sleep=lambda _s: None), light


_ROI = Roi(0, 0, 1, 1)
_SCRATCH = "/tmp/_negpy_ir_cal.raw"


def test_converted_camera_lands_on_target():
    svc, light = _service((300.0, 280.0, 260.0))
    cal = svc.calibrate_ir(_ROI, _SCRATCH)
    assert cal.signal == pytest.approx(target_signal(), rel=0.06)
    assert cal.shutter in SHUTTER_CANDIDATES
    assert light.ir == 0  # off at the end


@pytest.mark.parametrize("response,status", [((0.5, 0.1, 0.1), "under"), ((1e7, 1e7, 1e7), "over")])
def test_out_of_reach_raises_like_a_color_channel(response, status):
    svc, _light = _service(response)
    with pytest.raises(CalibrationExposureError) as err:
        svc.calibrate_ir(_ROI, _SCRATCH)
    assert (err.value.status, err.value.channel) == (status, "IR")


def test_full_run_adds_the_ir_phase_after_color():
    svc, _light = _service((300.0, 280.0, 260.0))
    progress = []
    result = svc.calibrate(_ROI, _SCRATCH, ir=True, progress=lambda f, _m: progress.append(f))
    assert result.ir is not None and not result.ir_error
    assert progress == sorted(progress) and progress[-1] == 1.0
    assert svc.calibrate(_ROI, _SCRATCH).ir is None


@pytest.mark.parametrize(
    "response,reason",
    [
        ((0.0, 0.0, 0.0), "the camera records no IR"),
        ((0.5, 0.1, 0.1), "the IR frame stays too dim even at the slowest shutter"),
    ],
)
def test_an_ir_failure_keeps_the_color_calibration(response, reason):
    svc, _light = _service(response)
    result = svc.calibrate(_ROI, _SCRATCH, ir=True)
    assert set(result.channels) == {"R", "G", "B"}
    assert result.ir is None and result.ir_error == reason


def test_cancel_during_the_ir_phase_still_cancels():
    cancel = threading.Event()
    svc, light = _service((300.0, 280.0, 260.0))
    orig = light.set_color

    def set_color(*a, **kw):
        if kw.get("ir"):
            cancel.set()
        orig(*a, **kw)

    light.set_color = set_color
    svc._calibrate_color = lambda *_a: CalibrationResult(channels={})
    with pytest.raises(RuntimeError, match="cancelled"):
        svc.calibrate(_ROI, _SCRATCH, ir=True, cancel=cancel)
