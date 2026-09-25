import gc
import os
import threading
from dataclasses import dataclass
from typing import List

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from negpy.domain.models import TiffCompression, WorkspaceConfig
from negpy.kernel.image.logic import calculate_file_hash
from negpy.services.export.triplet_merge import decode_params, write_merged_triplet
from negpy.services.rendering.image_processor import ImageProcessor


@dataclass(frozen=True)
class TripletMergeTask:
    """One triplet to merge. ``asset`` is a copy of its Film Strip entry."""

    asset: dict
    params: WorkspaceConfig
    out_path: str
    compression: TiffCompression


@dataclass(frozen=True)
class TripletMergeResult:
    asset: dict
    out_path: str
    new_hash: str = ""
    error: str = ""


class TripletMergeWorker(QObject):
    """Writes and verifies each merged TIFF. It deletes nothing: the controller moves the
    exposures to the Trash once the edits have followed the frames."""

    progress = pyqtSignal(int, int, str)  # current, total, label
    finished = pyqtSignal(list, bool)  # [TripletMergeResult], aborted

    def __init__(self) -> None:
        super().__init__()
        self._processor = ImageProcessor()
        self._cancel = threading.Event()

    @pyqtSlot()
    def cancel(self) -> None:
        self._cancel.set()

    @pyqtSlot(list)
    def run(self, tasks: List[TripletMergeTask]) -> None:
        self._cancel.clear()
        results: List[TripletMergeResult] = []
        aborted = False
        try:
            for i, task in enumerate(tasks):
                if self._cancel.is_set():
                    aborted = True
                    break
                self.progress.emit(i + 1, len(tasks), os.path.basename(task.out_path))
                try:
                    f32, _ir, _cs = self._processor._decode_oriented_f32(task.asset["path"], decode_params(task.params))
                    write_merged_triplet(f32, task.asset["path"], task.out_path, task.compression)
                    del f32
                    new_hash = calculate_file_hash(task.out_path)
                    if new_hash.startswith("err_"):
                        raise OSError(f"Could not read {os.path.basename(task.out_path)} back")
                    results.append(TripletMergeResult(task.asset, task.out_path, new_hash=new_hash))
                except Exception as e:
                    results.append(TripletMergeResult(task.asset, task.out_path, error=str(e)))
                gc.collect()
        finally:
            self.finished.emit(results, aborted)
