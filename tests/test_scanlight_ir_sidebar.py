"""The Scanlight panel's IR frame controls (Big Scanlight only)."""

import sys
import types
from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from PyQt6.QtWidgets import QApplication

from negpy.desktop.view.sidebar.scanlight import ScanlightSidebar
from negpy.services.capture.presets import ScanlightPreset

if not QApplication.instance():
    _app = QApplication(sys.argv)

_WHITE = "White Light (B&W or Slide Film)"


def _sidebar(has_ir=True):
    ctrl = MagicMock()
    ctrl.session.repo.get_global_setting.return_value = {}
    w = ScanlightSidebar(ctrl)
    w._light_has_ir = has_ir
    return w


def _poll(has_ir):
    return {"usb_ok": False, "usb_model": "", "light_ok": True, "light_detail": "fw", "light_has_white": True, "light_has_ir": has_ir}


def _select(w, monkeypatch, preset):
    monkeypatch.setattr(w._presets, "get", lambda _n: preset)
    w.preset_combo.addItem("TestStock", "TestStock")
    idx = w.preset_combo.findData("TestStock")
    w.preset_combo.setCurrentIndex(idx)
    w._on_preset_selected(idx)


_PRESET = ScanlightPreset(r_level=200, shutter_r="1/5", ir_level=250, shutter_ir="4")


def test_ir_controls_show_only_for_a_light_with_an_ir_led_and_a_camera_the_user_says_records_ir():
    w = _sidebar(has_ir=False)
    w._on_poll_status(_poll(False))
    assert w.capture_ir_btn.isHidden() and w.ir_slider.isHidden() and w._ir_shutter_row.isHidden()
    w._on_poll_status(_poll(True))
    # Off by default: only an IR-converted camera records the IR LED.
    assert not w.capture_ir_btn.isHidden() and w.capture_ir_btn.isEnabled() and not w.capture_ir_btn.isChecked()
    assert w.ir_slider.isHidden() and w._ir_shutter_row.isHidden()
    w.capture_ir_btn.setChecked(True)
    assert not w.ir_slider.isHidden() and not w._ir_shutter_row.isHidden()


def test_an_uncalibrated_preset_says_scans_skip_the_ir_frame(monkeypatch):
    w = _sidebar()
    w.capture_ir_btn.setChecked(True)
    _select(w, monkeypatch, ScanlightPreset(r_level=200, shutter_r="1/5"))
    assert "calibrate this preset with Capture IR on" in w._ir_hint.text()
    _select(w, monkeypatch, _PRESET)
    assert w._ir_hint.isHidden()
    assert (w.ir_slider.value(), w.ir_shutter_stepper.currentText()) == (250, "4")
    w.capture_ir_btn.setChecked(False)
    _select(w, monkeypatch, ScanlightPreset(r_level=200, shutter_r="1/5"))
    assert w._ir_hint.isHidden()


@pytest.mark.parametrize("has_ir,checked,expected", [(True, True, 250), (True, False, 0), (False, True, 0)])
def test_scan_request_carries_the_ir_exposure(tmp_path, monkeypatch, has_ir, checked, expected):
    w = _sidebar(has_ir)
    _select(w, monkeypatch, _PRESET)
    w.capture_ir_btn.setChecked(checked)
    w.output.folder_edit.setText(str(tmp_path))
    w.output.roll_edit.setText("Roll001")
    w._start_capture(retake=False)
    req = w.controller.start_capture.call_args[0][0]
    assert req.ir_level == expected and req.shutter_ir == "4"


def test_calibration_bakes_the_ir_exposure_into_the_preset(monkeypatch):
    w = _sidebar()
    saved: dict = {}
    monkeypatch.setattr(w._presets, "save", lambda _n, preset: saved.update(preset=preset))
    monkeypatch.setattr(w._presets, "get", lambda _n: None)
    monkeypatch.setattr(w, "_reload_presets", lambda **_k: None)
    w._calibrating_preset = "Portra 400"
    w._on_calibration_finished(
        types.SimpleNamespace(
            levels=(200, 180, 90),
            shutters=("1/5", "1/5", "1/5"),
            single_capture=False,
            sensor_matrix=None,
            ir=types.SimpleNamespace(level=240, shutter="1/3"),
        )
    )
    assert (saved["preset"].ir_level, saved["preset"].shutter_ir) == (240, "1/3")
    assert (w._settings.last_ir_level, w._settings.last_shutter_ir) == (240, "1/3")


def test_white_light_preset_uses_the_last_calibrated_ir_exposure():
    w = _sidebar()
    w._settings = replace(w._settings, last_ir_level=230, last_shutter_ir="2")
    idx = w.preset_combo.findData(_WHITE)
    w.preset_combo.setCurrentIndex(idx)
    w._on_preset_selected(idx)
    assert (w._settings.ir_level, w._settings.shutter_ir) == (230, "2")
    assert w.capture_ir_btn.isEnabled()


@pytest.mark.parametrize("has_ir,capture_ir,asked", [(True, True, True), (True, False, False), (False, True, False)])
def test_calibration_measures_ir_only_when_the_user_turned_capture_ir_on(monkeypatch, has_ir, capture_ir, asked):
    w = _sidebar(has_ir)
    w.capture_ir_btn.setChecked(capture_ir)
    monkeypatch.setattr(w.calib_window.image, "roi", lambda: (0.4, 0.4, 0.2, 0.2))
    monkeypatch.setattr(w, "_available_shutters", lambda info=None, scanlight=True: ("1/5",))
    w._on_calibrate_new_preset("Portra 400")
    req = w.controller.start_calibration.call_args[0][0]
    assert req.calibrate_ir is asked


def test_a_failed_ir_phase_still_saves_the_color_preset(monkeypatch):
    w = _sidebar()
    saved: dict = {}
    monkeypatch.setattr(w._presets, "save", lambda _n, preset: saved.update(preset=preset))
    monkeypatch.setattr(w._presets, "get", lambda _n: None)
    monkeypatch.setattr(w, "_reload_presets", lambda **_k: None)
    w._calibrating_preset = "Portra 400"
    w._on_calibration_finished(
        types.SimpleNamespace(
            levels=(200, 180, 90),
            shutters=("1/5", "1/5", "1/5"),
            single_capture=False,
            sensor_matrix=None,
            ir=None,
            ir_error="the camera records no IR",
        )
    )
    assert saved["preset"].r_level == 200 and saved["preset"].ir_level == 0
    assert "with no IR exposure: the camera records no IR" in w.status_strip.message()
