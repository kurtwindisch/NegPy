"""Merge Roll to TIFF: a Trichrome triplet becomes one linear TIFF that renders the same
frame, carries the triplet's edit, and never costs an exposure it could not replace."""

import os
from dataclasses import replace

import numpy as np
import pytest
import tifffile

from negpy.domain.models import WorkspaceConfig
from negpy.features.process.sensor import effective_sensor_matrix
from negpy.features.rgbscan.models import RgbScanConfig
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.assets import rolls
from negpy.services.assets.sidecar import load_sidecar, sidecar_path_for, write_sidecar
from negpy.services.assets.triplet_merge import carry_edit, carry_sidecar, merged_edit
from negpy.services.export import triplet_merge
from negpy.services.export.triplet_merge import (
    MergeVerifyError,
    can_merge,
    decode_params,
    is_merged_triplet,
    merged_path_for,
    write_merged_triplet,
)
from negpy.services.rendering.image_processor import ImageProcessor

_MATRIX = (1.0, -0.1, 0.0, -0.05, 1.0, -0.1, 0.0, -0.2, 1.0)


def _triplet(tmp_path) -> tuple[str, str, str]:
    paths = tuple(str(tmp_path / f"IMG_{i}.ARW") for i in (1, 2, 3))
    for p in paths:
        open(p, "wb").close()
    return paths  # type: ignore[return-value]


def _exposures(red: str, green: str, blue: str) -> dict:
    rng = np.random.default_rng(7)
    return {p: rng.integers(0, 65535, size=(24, 36, 3), dtype=np.uint16) for p in (red, green, blue)}


def _fake_sensor_decode(buffers: dict, orientation: int = 1):
    def decode(path, linear_raw, fast=False, wb_override=None, demosaic="Auto", positive_source=False, highlight_mode=0, **_kw):
        return buffers[path], {"orientation": orientation, "cam_xyz": None, "camera_wb": [1.0, 1.0, 1.0]}

    return decode


def _triplet_config(green: str, blue: str) -> WorkspaceConfig:
    return replace(WorkspaceConfig(), rgbscan=RgbScanConfig(enabled=True, green_path=green, blue_path=blue, align=False))


def _repo(tmp_path) -> StorageRepository:
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    return repo


@pytest.mark.parametrize("orientation", [1, 6])
def test_merged_tiff_decodes_to_the_source_the_triplet_renders_from(tmp_path, orientation):
    red, green, blue = _triplet(tmp_path)
    cfg = _triplet_config(green, blue)
    triplet = ImageProcessor()
    triplet._decode_sensor_rgb = _fake_sensor_decode(_exposures(red, green, blue), orientation)
    expected, _ir, _cs = triplet._decode_oriented_f32(red, cfg)

    merged, _ir, _cs = triplet._decode_oriented_f32(red, decode_params(cfg))
    out = merged_path_for(red)
    write_merged_triplet(merged, red, out)

    reread, _ir, _cs = ImageProcessor()._decode_oriented_f32(out, merged_edit(cfg))
    assert np.array_equal(reread, expected)


def test_decode_params_leave_flat_field_to_the_edit():
    cfg = replace(WorkspaceConfig(), rgbscan=RgbScanConfig(enabled=True, green_path="g", blue_path="b"))
    cfg = replace(cfg, flatfield=replace(cfg.flatfield, apply=True, profile_id="abc"))
    assert decode_params(cfg).flatfield.apply is False
    with pytest.raises(ValueError):
        decode_params(WorkspaceConfig())


def test_merged_edit_drops_the_triplet_and_the_sensor_unmix():
    cfg = _triplet_config("g", "b")
    cfg = replace(cfg, process=replace(cfg.process, linear_raw=True, sensor_matrix=_MATRIX, sensor_profile="rig"))
    edit = merged_edit(cfg)
    assert edit.rgbscan == RgbScanConfig()
    assert effective_sensor_matrix(edit.process) is None
    assert edit.process.sensor_profile == "None"


def test_write_is_atomic_and_recognized_as_a_merge(tmp_path):
    red, _g, _b = _triplet(tmp_path)
    out = merged_path_for(red)
    f32 = np.full((8, 10, 3), 0.25, dtype=np.float32)
    write_merged_triplet(f32, red, out)

    assert out.endswith("IMG_1_RGB.tif")
    assert tifffile.imread(out).dtype == np.uint16
    assert is_merged_triplet(out)
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".part")]


def test_an_ordinary_tiff_is_not_a_merge(tmp_path):
    path = str(tmp_path / "scan.tif")
    tifffile.imwrite(path, np.zeros((4, 4, 3), dtype=np.uint16), description="scanner")
    assert not is_merged_triplet(path)
    assert not is_merged_triplet(str(tmp_path / "IMG_1.ARW"))


def test_a_file_that_reads_back_wrong_is_never_kept(tmp_path, monkeypatch):
    red, _g, _b = _triplet(tmp_path)
    out = merged_path_for(red)
    monkeypatch.setattr(triplet_merge.tifffile, "imread", lambda _p: np.zeros((8, 10, 3), dtype=np.uint16))

    with pytest.raises(MergeVerifyError):
        write_merged_triplet(np.full((8, 10, 3), 0.5, dtype=np.float32), red, out)
    assert not os.path.exists(out)
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".part")]


def test_an_existing_file_is_never_replaced(tmp_path):
    red, _g, _b = _triplet(tmp_path)
    out = str(tmp_path / "IMG_1_RGB.tif")
    open(out, "wb").close()
    with pytest.raises(FileExistsError):
        write_merged_triplet(np.zeros((4, 4, 3), dtype=np.float32), red, out)
    assert os.path.getsize(out) == 0


def test_merged_names_skip_files_on_disk_and_names_taken_in_the_batch(tmp_path):
    red, _g, _b = _triplet(tmp_path)
    open(tmp_path / "IMG_1_RGB.tif", "wb").close()
    second = merged_path_for(red)
    assert second.endswith("IMG_1_RGB_2.tif")
    assert merged_path_for(red, frozenset({second})).endswith("IMG_1_RGB_3.tif")


def test_can_merge_needs_three_camera_raws_on_disk(tmp_path):
    red, green, blue = _triplet(tmp_path)
    assert can_merge(red, green, blue)
    os.remove(blue)
    assert not can_merge(red, green, blue)
    tif = str(tmp_path / "a.tif")
    open(tif, "wb").close()
    assert not can_merge(tif, green, red)


def test_carry_edit_copies_everything_and_keeps_the_original(tmp_path):
    repo = _repo(tmp_path)
    red, green, blue = _triplet(tmp_path)
    new_path = str(tmp_path / "IMG_1_RGB.tif")
    cfg = _triplet_config(green, blue)
    repo.save_file_settings("red", cfg, file_path=red)
    repo.save_history_step("red", 0, cfg)
    repo.save_work_print("red", "Warm", cfg)
    repo.save_file_mark("red", "keeper", file_path=red)
    roll_id = rolls.create_virtual_roll(repo, "Roll", [red, green, blue, "/other.ARW"])
    rolls.fork_edit(repo, roll_id, "red", red, cfg)
    rolls.set_frame_override(repo, roll_id, "red", "film", True)
    scene = rolls.create_scene(repo, roll_id, "Beach", ["red"])

    carry_edit(repo, "red", red, "new", new_path, [green, blue], cfg)

    assert repo.load_file_settings("new").rgbscan == RgbScanConfig()
    assert repo.load_all_history("new")[0][1].rgbscan == RgbScanConfig()
    assert repo.load_work_print("new", "Warm").rgbscan == RgbScanConfig()
    assert repo.load_file_marks()["new"] == "keeper"
    assert repo.load_file_marks_by_path()[new_path] == "keeper"
    assert repo.load_file_settings(rolls.roll_edit_hash("new", roll_id)).rgbscan == RgbScanConfig()
    entry = rolls.roll_for_id(repo, roll_id)
    assert entry["member_paths"] == [new_path, "/other.ARW"]
    assert rolls.is_forked(repo, roll_id, "new")
    assert rolls.frame_override_cards(repo, roll_id, "new") == {"film"}
    assert "new" in dict(rolls.roll_scenes(repo, roll_id))[scene]["member_hashes"]
    assert repo.load_file_settings("red").rgbscan == cfg.rgbscan
    assert repo.load_file_marks()["red"] == "keeper"


def test_carry_edit_never_overwrites_an_edit_the_new_file_already_has(tmp_path):
    repo = _repo(tmp_path)
    repo.save_file_settings("red", _triplet_config("g", "b"))
    own = replace(WorkspaceConfig(), process=replace(WorkspaceConfig().process, crosstalk_strength=0.5))
    repo.save_file_settings("new", own)
    assert repo.copy_file_edits("red", "new", "/x.tif", merged_edit) is False
    assert repo.load_file_settings("new").process.crosstalk_strength == 0.5


def test_a_roll_sensor_matrix_is_locked_out_of_the_merged_frame(tmp_path):
    repo = _repo(tmp_path)
    folder = tmp_path / "roll"
    folder.mkdir()
    red, green, blue = _triplet(folder)
    new_path = str(folder / "IMG_1_RGB.tif")
    roll_id = rolls.recognize_folder(repo, str(folder))
    rolls.set_roll_defaults(repo, roll_id, sensor_matrix=_MATRIX, linear_raw=True, narrowband_scan=True)
    cfg = _triplet_config(green, blue)

    carry_edit(repo, "red", red, "new", new_path, [green, blue], cfg)

    assert "sensor" in rolls.frame_override_cards(repo, roll_id, "new")
    resolved = rolls.resolve_roll_config(repo, roll_id, "new", repo.load_file_settings("new"))
    assert resolved.process.sensor_matrix is None
    assert resolved.process.narrowband_scan is True


def test_a_sidecar_moves_only_when_the_red_exposure_has_one(tmp_path):
    red, green, blue = _triplet(tmp_path)
    new_path = str(tmp_path / "IMG_1_RGB.tif")
    cfg = _triplet_config(green, blue)
    assert carry_sidecar(red, new_path, cfg) is False
    assert not os.path.exists(sidecar_path_for(new_path))

    write_sidecar(red, cfg)
    assert carry_sidecar(red, new_path, cfg) is True
    assert load_sidecar(new_path).rgbscan == RgbScanConfig()


def _finish(tmp_path, monkeypatch, results, trash=True):
    from unittest.mock import MagicMock

    from negpy.desktop.controller import AppController

    trashed: list = []
    monkeypatch.setattr("negpy.desktop.controller._move_to_trash", lambda p: trashed.append(p) or True)
    ctrl = MagicMock()
    ctrl.session.repo = _repo(tmp_path)
    ctrl._triplet_merge_trash = trash
    ctrl._apply_roll_forks = lambda assets: None
    return ctrl, trashed, AppController._on_triplet_merge_finished


def _asset(red, green, blue, file_hash="red"):
    return {"name": "IMG_1 (RGB)", "path": red, "hash": file_hash, "green_path": green, "blue_path": blue}


def test_finish_swaps_the_frame_and_trashes_its_exposures(tmp_path, monkeypatch):
    from negpy.desktop.workers.triplet_merge import TripletMergeResult

    red, green, blue = _triplet(tmp_path)
    out = str(tmp_path / "IMG_1_RGB.tif")
    open(out, "wb").close()
    write_sidecar(red, _triplet_config(green, blue))
    asset = _asset(red, green, blue)
    results = [TripletMergeResult(asset, out, new_hash="new")]
    ctrl, trashed, finish = _finish(tmp_path, monkeypatch, results)
    ctrl.state.uploaded_files = [{"path": "/a.ARW", "hash": "a"}, asset]
    ctrl.session.repo.save_file_settings("red", _triplet_config(green, blue), file_path=red)

    finish(ctrl, results, False)

    swapped = ctrl.session.replace_assets.call_args.args[0]
    assert list(swapped) == [1]
    assert swapped[1]["path"] == out and swapped[1]["hash"] == "new"
    assert ctrl.session.repo.load_file_settings("new") is not None
    assert trashed == [red, green, blue, sidecar_path_for(red)]


def test_finish_keeps_the_exposures_of_a_frame_that_failed(tmp_path, monkeypatch):
    from negpy.desktop.workers.triplet_merge import TripletMergeResult

    red, green, blue = _triplet(tmp_path)
    asset = _asset(red, green, blue)
    results = [TripletMergeResult(asset, str(tmp_path / "IMG_1_RGB.tif"), error="disk full")]
    ctrl, trashed, finish = _finish(tmp_path, monkeypatch, results)
    ctrl.state.uploaded_files = [asset]

    finish(ctrl, results, False)

    assert trashed == []
    assert ctrl.session.replace_assets.call_args.args[0] == {}
    assert "1 failed" in ctrl.set_status.call_args.args[0]


def test_finish_without_trash_leaves_every_exposure(tmp_path, monkeypatch):
    from negpy.desktop.workers.triplet_merge import TripletMergeResult

    red, green, blue = _triplet(tmp_path)
    out = str(tmp_path / "IMG_1_RGB.tif")
    open(out, "wb").close()
    asset = _asset(red, green, blue)
    results = [TripletMergeResult(asset, out, new_hash="new")]
    ctrl, trashed, finish = _finish(tmp_path, monkeypatch, results, trash=False)
    ctrl.state.uploaded_files = [asset]

    finish(ctrl, results, False)

    assert trashed == []
    assert list(ctrl.session.replace_assets.call_args.args[0]) == [0]


def test_finish_copies_a_forked_frames_shared_edit(tmp_path, monkeypatch):
    from negpy.desktop.workers.triplet_merge import TripletMergeResult

    red, green, blue = _triplet(tmp_path)
    out = str(tmp_path / "IMG_1_RGB.tif")
    open(out, "wb").close()
    asset = _asset(red, green, blue, file_hash=rolls.roll_edit_hash("red", "r1"))
    results = [TripletMergeResult(asset, out, new_hash="new")]
    ctrl, _trashed, finish = _finish(tmp_path, monkeypatch, results)
    ctrl.state.uploaded_files = [asset]
    ctrl.session.repo.save_file_settings("red", _triplet_config(green, blue), file_path=red)

    finish(ctrl, results, False)

    assert ctrl.session.repo.load_file_settings("new") is not None


def test_plan_skips_stitches_and_missing_exposures(tmp_path):
    from unittest.mock import MagicMock

    from negpy.desktop.controller import AppController

    red, green, blue = _triplet(tmp_path)
    ctrl = MagicMock()
    ctrl.state.uploaded_files = [
        {"name": "plain.ARW", "path": str(tmp_path / "plain.ARW")},
        _asset(red, green, blue),
        {**_asset(red, green, blue), "name": "stitch", "stitch_paths": ("/p.ARW",)},
        {**_asset(red, green, str(tmp_path / "gone.ARW")), "name": "gone"},
    ]
    indices, skipped = AppController.triplet_merge_plan(ctrl)
    assert indices == [1]
    assert skipped == ["stitch: part of a stitch", "gone: an exposure is missing or not a camera RAW"]
