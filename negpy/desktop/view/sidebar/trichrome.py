import os

from negpy.desktop.view.confirm import confirm_assembly_mode, confirm_triplet_merge
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import hint_label, wrap_tooltip
from negpy.desktop.view.widgets.rgb_triplet_dialog import open_triplet_dialog
from negpy.features.rgbscan.models import is_rgb_triplet


def merge_roll_triplets(parent, controller) -> None:
    """Merge Roll to TIFF: plan, confirm, then hand the triplets to the controller."""
    indices, skipped = controller.triplet_merge_plan()
    if not indices:
        controller.set_status("No triplet in this roll can be merged", 4000)
        return
    trash = confirm_triplet_merge(parent, len(indices), skipped)
    if trash is not None:
        controller.request_triplet_merge(indices, trash)


class TrichromeSidebar(BaseSidebar):
    """
    Trichromatic capture: whether a folder groups into red/green/blue triplets, and
    which three exposures the open frame is assembled from. The mode is a rig fact, not
    a roll's — one flag for every roll — since a copy stand either shoots three
    exposures per frame or it does not.
    """

    def _init_ui(self) -> None:
        self.enable_btn = self._small_toggle(
            "mdi.google-circles-communities",
            "Trichrome Mode",
            bool(self.controller.session.repo.get_global_setting("rgbscan_mode", False)),
            wrap_tooltip(
                "Assemble each frame from its red, green and blue exposures. A folder is grouped into "
                "triplets in capture order on load; the mode applies to every roll you open."
            ),
        )
        self.layout.addWidget(self.enable_btn)

        self.edit_btn = self._labeled_action(
            "fa5s.pen",
            " Edit Triplet…",
            wrap_tooltip("Assign this frame's three exposures by hand, for a frame the grouping got wrong"),
        )
        self.layout.addWidget(self.edit_btn)

        self.merge_btn = self._labeled_action(
            "fa5s.object-group",
            " Merge Roll to TIFF…",
            wrap_tooltip(
                "Write each triplet as one linear RGB TIFF next to its red exposure, carry the frame's "
                "edit over, and optionally move the three exposures to the Trash"
            ),
        )
        self.layout.addWidget(self.merge_btn)

        self.hint = hint_label("")
        self.layout.addWidget(self.hint)

    def _connect_signals(self) -> None:
        self.enable_btn.toggled.connect(self._on_toggled)
        self.controller.rgb_scan_mode_changed.connect(self._follow_mode)
        self.edit_btn.clicked.connect(lambda: open_triplet_dialog(self, self.controller.session))
        self.merge_btn.clicked.connect(lambda: merge_roll_triplets(self, self.controller))
        self.controller.session.files_changed.connect(self.sync_ui)
        self.sync_ui()

    def _on_toggled(self, checked: bool) -> None:
        """Turning the mode on regroups the loaded roll into triplets, which reads and
        thumbnails every frame again, so it asks first. With nothing loaded there is
        nothing to regroup and nothing to confirm."""
        loaded = len(self.state.uploaded_files)
        if checked and loaded and not confirm_assembly_mode(self, "Trichrome", loaded):
            self._follow_mode(False)
            return
        self.controller.set_rgb_scan_mode(checked)

    def _follow_mode(self, enabled: bool) -> None:
        """Follow a mode change this card did not make. Signals are blocked because the
        controller has already applied it; letting toggled through would ask for it a
        second time and re-run discovery."""
        self.enable_btn.blockSignals(True)
        self.enable_btn.setChecked(enabled)
        self.enable_btn.blockSignals(False)

    def sync_ui(self) -> None:
        conf = self.state.config.rgbscan
        self.block_signals(True)
        try:
            self.enable_btn.setChecked(bool(self.controller.session.repo.get_global_setting("rgbscan_mode", False)))
            self.edit_btn.setEnabled(bool(self.state.uploaded_files))
            self.merge_btn.setEnabled(any(f.get("green_path") and f.get("blue_path") for f in self.state.uploaded_files))
            if is_rgb_triplet(conf):
                green, blue = os.path.basename(conf.green_path), os.path.basename(conf.blue_path)
                self.hint.setText(f"This frame is assembled with {green} and {blue}.")
            else:
                self.hint.setText("This frame is a single exposure.")
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        for w in (self.enable_btn, self.edit_btn, self.merge_btn):
            w.blockSignals(blocked)
