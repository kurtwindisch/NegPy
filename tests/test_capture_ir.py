"""IR frame capture after the color exposure(s), for IR dust removal."""

import os
import threading

import pytest

from negpy.infrastructure.capture.base import CaptureSettings, IrExposure
from negpy.services.capture.service import CaptureError, CaptureService


class FakeLight:
    def __init__(self):
        self.colors: list[tuple[int, int, int, int, int]] = []

    def set_color(self, r=0, g=0, b=0, w=0, ir=0, save=False):
        self.colors.append((r, g, b, w, ir))

    def off(self):
        self.colors.append((0, 0, 0, 0, 0))

    def close(self):
        pass


class FakeCamera:
    def __init__(self, size=21 * 1024 * 1024, fail_on=None):
        self.size = size
        self.fail_on = fail_on
        self.shots: list[tuple[str, object]] = []

    def capture(self, out_path, shutter=None, iso=None, aperture=None):
        out_path = os.path.splitext(out_path)[0] + ".ARW"
        self.shots.append((os.path.basename(out_path), shutter))
        size = 10 if self.fail_on and out_path.endswith(self.fail_on) else self.size
        with open(out_path, "wb") as f:
            f.write(b"\0" * size)
        return out_path

    def close(self):
        pass


_IR = IrExposure(level=250, shutter="2")


def _settings(tmp_path, **kw):
    base = dict(roll_name="Roll001", frame_number=1, output_folder=str(tmp_path), levels=(200, 180, 255), ir=_IR)
    base.update(kw)
    return CaptureSettings(**base)


def _lit(light):
    return [c for c in light.colors if any(c)]


def test_triplet_shoots_ir_last_named_after_red(tmp_path):
    light, cam = FakeLight(), FakeCamera()
    channels, progress = [], []
    result = CaptureService(light, cam, sleep=lambda _s: None).capture_triplet(
        _settings(tmp_path), progress=progress.append, on_channel=channels.append
    )

    assert sorted(os.listdir(tmp_path)) == [
        "Roll001_Frame001_B.ARW",
        "Roll001_Frame001_G.ARW",
        "Roll001_Frame001_R.ARW",
        "Roll001_Frame001_R_IR.ARW",
    ]
    assert os.path.basename(result.ir_path) == "Roll001_Frame001_R_IR.ARW"
    assert len(result.paths) == 3  # the IR frame is a companion, not a color file
    assert _lit(light)[-1] == (0, 0, 0, 0, 250)  # IR lit alone, after blue
    assert channels == ["R", "G", "B", "IR"]
    assert cam.shots[-1][1] == "2"
    assert progress[-1] == pytest.approx(1.0)
    assert light.colors[-1] == (0, 0, 0, 0, 0)


def test_single_and_white_shoot_ir_after_the_frame(tmp_path):
    svc = CaptureService(FakeLight(), FakeCamera(), sleep=lambda _s: None)
    single = tmp_path / "single"
    svc.capture_rgb_single(_settings(single))
    assert sorted(os.listdir(single)) == ["Roll001_Frame001.ARW", "Roll001_Frame001_IR.ARW"]

    white = tmp_path / "white"
    light = FakeLight()
    CaptureService(light, FakeCamera(), sleep=lambda _s: None).capture_white(
        roll_name="Slide01", frame_number=2, output_folder=str(white), w_level=200, ir=_IR
    )
    assert sorted(os.listdir(white)) == ["Slide01_Frame002.ARW", "Slide01_Frame002_IR.ARW"]
    assert _lit(light) == [(0, 0, 0, 200, 0), (0, 0, 0, 0, 250)]


def test_failed_ir_shot_promotes_nothing(tmp_path):
    svc = CaptureService(FakeLight(), FakeCamera(fail_on="_IR.ARW"), sleep=lambda _s: None)
    with pytest.raises(CaptureError):
        svc.capture_triplet(_settings(tmp_path))
    assert os.listdir(tmp_path) == []


def test_cancel_before_ir_promotes_nothing(tmp_path):
    cancel = threading.Event()

    def on_channel(letter):
        if letter == "IR":
            cancel.set()

    cam = FakeCamera()
    with pytest.raises(CaptureError):
        CaptureService(FakeLight(), cam, sleep=lambda _s: None).capture_triplet(_settings(tmp_path), cancel=cancel, on_channel=on_channel)
    assert os.listdir(tmp_path) == []


def test_retake_without_ir_drops_the_stale_ir_frame(tmp_path):
    svc = CaptureService(FakeLight(), FakeCamera(), sleep=lambda _s: None)
    svc.capture_triplet(_settings(tmp_path))
    (tmp_path / "Roll001_Frame001_R_IR_VALID.tif").write_bytes(b"")

    svc.capture_triplet(_settings(tmp_path, ir=None))

    assert sorted(os.listdir(tmp_path)) == ["Roll001_Frame001_B.ARW", "Roll001_Frame001_G.ARW", "Roll001_Frame001_R.ARW"]


def test_retake_with_ir_keeps_only_the_new_ir_frame(tmp_path):
    (tmp_path / "Roll001_Frame001.ARW").write_bytes(b"")
    (tmp_path / "Roll001_Frame001_IR.tif").write_bytes(b"")

    CaptureService(FakeLight(), FakeCamera(), sleep=lambda _s: None).capture_rgb_single(_settings(tmp_path))

    assert sorted(os.listdir(tmp_path)) == ["Roll001_Frame001.ARW", "Roll001_Frame001_IR.ARW"]
