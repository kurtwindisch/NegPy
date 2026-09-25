"""Carries a Trichrome triplet's edit and roll records onto the TIFF merged from it."""

import os
from dataclasses import replace
from typing import Any, List

from negpy.domain.models import WorkspaceConfig
from negpy.features.rgbscan.models import RgbScanConfig
from negpy.services.assets import rolls
from negpy.services.assets.sidecar import sidecar_path_for, write_sidecar


def merged_edit(config: WorkspaceConfig) -> WorkspaceConfig:
    """A triplet's edit as it applies to its merged TIFF.

    The TIFF is one source, so the triplet paths go. The sensor unmix goes too: a
    triplet skips it, and a single file would apply it.
    """
    return replace(
        config,
        rgbscan=RgbScanConfig(),
        process=replace(config.process, sensor_matrix=None, sensor_profile="None"),
    )


def carry_edit(
    repo: Any, red_hash: str, red_path: str, new_hash: str, new_path: str, dropped_paths: List[str], config: WorkspaceConfig
) -> None:
    """Copy the triplet's edit, forks, card locks, scene and roll membership to the merged file.

    *config* is the triplet's hydrated edit. Nothing is taken from the red exposure. A
    roll whose Calibration card holds a sensor matrix gets that card locked on the new
    frame at the roll's other values, so the roll cannot put the unmix back.
    """
    repo.copy_file_edits(red_hash, new_hash, new_path, merged_edit)
    for roll_id in rolls.adopt_replacement(repo, red_hash, new_hash, red_path, new_path, dropped_paths):
        repo.copy_file_edits(rolls.roll_edit_hash(red_hash, roll_id), rolls.roll_edit_hash(new_hash, roll_id), new_path, merged_edit)
    for roll_id in rolls.rolls_containing_path(repo, new_path):
        if rolls.roll_defaults(repo, roll_id).get("sensor_matrix") is None:
            continue
        if repo.load_file_settings(new_hash) is None:
            resolved = rolls.resolve_roll_config(repo, roll_id, red_hash, config)
            repo.save_file_settings(new_hash, merged_edit(resolved), file_path=new_path)
        rolls.set_frame_override(repo, roll_id, new_hash, "sensor", True)


def carry_sidecar(red_path: str, new_path: str, config: WorkspaceConfig) -> bool:
    """Write the merged file's sidecar when the red exposure has one. Returns whether it did."""
    if not os.path.exists(sidecar_path_for(red_path)):
        return False
    write_sidecar(new_path, merged_edit(config))
    return True
