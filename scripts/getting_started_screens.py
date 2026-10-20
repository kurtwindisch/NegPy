"""Capture the screens of docs/GETTING_STARTED.md from the in-app tour.

Opens the real window on a demo roll in a throwaway user directory, goes to each tour step,
rings the step's target and writes docs/media/getting_started/<slug>.webp. Needs a display:
on Linux, run it under `xvfb-run -a -s "-screen 0 1920x1080x24"`.
"""

import os
import sys
import tempfile
from pathlib import Path

os.environ["NEGPY_USER_DIR"] = tempfile.mkdtemp(prefix="negpy_screens_")
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "media" / "getting_started"
WINDOW_SIZE = (1600, 1000)
IMAGE_WIDTH = 1400
WEBP_QUALITY = 78
DEMO_SEEDS = (7, 11, 23, 42, 5, 64)
# The first render and the Film Strip thumbnails of the demo roll must finish before the first grab.
LOAD_MS = 20000
STEP_MS = 2500
TIMEOUT_MS = 300000

# Tour step title (English) -> image name. A step missing here gets no image and a warning.
SLUGS = {
    "Welcome to NegPy": "welcome",
    "The Screen": "screen",
    "Find Anything": "find",
    "The Library: Rolls": "library",
    "Keep and Reject": "keep-reject",
    "Light Table": "light-table",
    "Frame or Roll": "frame-or-roll",
    "Setup": "setup",
    "Your Scanning Setup": "scanning-setup",
    "Metering": "metering",
    "Roll Analysis": "roll-analysis",
    "Crop": "crop",
    "Straighten": "straighten",
    "Print Stock": "print-stock",
    "Print Density": "print-density",
    "ISO-R Grade": "grade",
    "Test Strip": "test-strip",
    "Shadows and Highlights": "shadows-highlights",
    "Filtration": "filtration",
    "Dodge and Burn": "dodge-burn",
    "Pick Better": "pick-better",
    "Lab": "lab",
    "Toning": "toning",
    "Dust and Scratches": "dust",
    "Finishing": "finishing",
    "The Analysis Readout": "analysis",
    "Before and After": "before-after",
    "Soft Proof": "soft-proof",
    "History and Work Prints": "history",
    "Metadata": "metadata",
    "Export": "export",
    "Scanning": "scanning",
}
SKIPPED = {"You're All Set"}


def main() -> int:
    import numpy as np
    import tifffile
    from PIL import Image

    from negpy.desktop import main as app_main
    from negpy.features.flatfield.logic import set_gain_provider
    from negpy.infrastructure.simulated.images import negative
    from negpy.infrastructure.storage.repository import StorageRepository
    from negpy.kernel.system.config import APP_CONFIG, BASE_USER_DIR
    from negpy.kernel.system.i18n import configure
    from negpy.services.assets.flatfield import FlatFieldProfiles

    app_main._bootstrap_environment()
    repo = StorageRepository(APP_CONFIG.edits_db_path, APP_CONFIG.settings_db_path)
    repo.initialize()
    repo.save_global_setting("tutorial_seen", True)
    # Before the desktop imports, which translate their labels on import.
    configure("en")
    set_gain_provider(FlatFieldProfiles.load_gain)

    from PyQt6.QtCore import QTimer
    from PyQt6.QtGui import QColor, QPainter, QPen
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setStyle(app_main._AppStyle("Fusion"))
    app.installEventFilter(app_main.AppEventFilter(app))

    from negpy.desktop.controller import AppController
    from negpy.desktop.session import DesktopSessionManager
    from negpy.desktop.view.main_window import MainWindow
    from negpy.desktop.view.styles.templates import load_stylesheet
    from negpy.desktop.view.styles.theme import THEME
    from negpy.desktop.view.widgets import tutorial_steps

    app.setStyleSheet(load_stylesheet())
    window = MainWindow(AppController(DesktopSessionManager(repo)))
    window.resize(*WINDOW_SIZE)
    window.move(40, 40)
    window.show()
    steps = tutorial_steps.build(window)
    overlay = window.tutorial_overlay
    scratch = Path(BASE_USER_DIR)
    OUT.mkdir(parents=True, exist_ok=True)

    def act(action_id: str) -> None:
        action = window.shortcut_manager.action_for(action_id)
        if action is not None:
            action()

    def open_palette() -> None:
        def fill() -> None:
            palette = QApplication.activeModalWidget()
            if palette is not None and hasattr(palette, "query"):
                palette.query.setText("grade")

        # The palette runs a modal loop, so it opens from the event loop and the grab closes it.
        QTimer.singleShot(0, lambda: act("command_palette"))
        QTimer.singleShot(800, fill)

    # The state a step's screen shows, beyond the step's own target.
    stage = {
        "Find Anything": open_palette,
        "Light Table": lambda: act("toggle_light_table"),
        "Test Strip": lambda: act("toggle_test_strip"),
        "Before and After": lambda: act("toggle_compare"),
    }

    def grab(idx: int, slug: str) -> None:
        overlay.hide()
        app.processEvents()
        pixmap = window.grab()
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        hole, _also, focus = overlay._rects(steps[idx])
        for rect, width in ((hole, 3), (focus, 2)):
            if rect is not None and rect.width() < window.width() * 0.95:
                painter.setPen(QPen(QColor(THEME.accent_primary), width))
                painter.drawRoundedRect(rect.adjusted(2, 2, -2, -2), 6, 6)
        modal = QApplication.activeModalWidget()
        if modal is not None and modal is not window:
            painter.drawPixmap(window.mapFromGlobal(modal.mapToGlobal(modal.rect().topLeft())), modal.grab())
            modal.close()
        painter.end()
        png = scratch / f"{slug}.png"
        pixmap.save(str(png))
        image = Image.open(png).convert("RGB")
        image = image.resize((IMAGE_WIDTH, round(image.height * IMAGE_WIDTH / image.width)), Image.Resampling.LANCZOS)
        image.save(OUT / f"{slug}.webp", quality=WEBP_QUALITY, method=6)
        print(f"{slug}.webp", flush=True)

    pos = {"idx": -1}
    started: list[bool] = []

    def next_step() -> None:
        idx = pos["idx"] = pos["idx"] + 1
        if idx >= len(steps):
            app.quit()
            return
        title = steps[idx].title
        slug = SLUGS.get(title)
        if slug is None:
            if title not in SKIPPED:
                print(f"warning: tour step {title!r} has no image name in SLUGS", file=sys.stderr)
            next_step()
            return
        if started:
            overlay.goto(idx)
        else:
            started.append(True)
            overlay.start(steps, idx)
        overlay.hide()
        if title in stage:
            stage[title]()
        QTimer.singleShot(STEP_MS, lambda: (grab(idx, slug), next_step()))

    def load_demo_roll() -> None:
        folder = scratch / "Demo Roll"
        folder.mkdir(exist_ok=True)
        for n, seed in enumerate(DEMO_SEEDS, 1):
            rgb, _ir = negative(1600, 2400, seed=seed)
            tifffile.imwrite(folder / f"frame_{n:02d}.tif", np.ascontiguousarray(rgb), photometric="rgb")
        window.controller.open_library_folder(str(folder))
        window.right_panel.library_tree.reload()
        window.right_panel.library_tree.rolls_changed.emit()
        QTimer.singleShot(LOAD_MS, next_step)

    def timeout() -> None:
        print("timed out", file=sys.stderr)
        app.exit(2)

    QTimer.singleShot(1500, load_demo_roll)
    QTimer.singleShot(TIMEOUT_MS, timeout)
    code = app.exec()
    window.controller.cleanup()
    return code


if __name__ == "__main__":
    sys.exit(main())
