from typing import TYPE_CHECKING, Optional

import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.features.flatfield.logic import apply_flatfield, flatfield_token
from negpy.features.flatfield.models import FlatFieldConfig
from negpy.features.lens.logic import apply_lens
from negpy.kernel.image.logic import apply_exif_orientation
from negpy.features.lens.models import LensCorrections, LensMetadata
from negpy.features.hdr.models import hdr_active
from negpy.features.retouch.logic import downsample_ir
from negpy.features.rgbscan.models import is_rgb_triplet
from negpy.features.stitch.models import stitch_active
from negpy.services.assets.half_frame import slice_chain

if TYPE_CHECKING:
    from negpy.domain.models import WorkspaceConfig


def metadata_lens_corrections(config: "WorkspaceConfig") -> LensCorrections:
    """Composite registrations refer to the unwarped component images."""
    if stitch_active(config.stitch) or hdr_active(config.hdr) or is_rgb_triplet(config.rgbscan):
        return LensCorrections()
    return LensCorrections(config.geometry.lens_distortion_from_metadata, config.geometry.lens_ca_from_metadata)


def flatfield_source(
    img: ImageBuffer,
    config: "WorkspaceConfig",
    gain_slices: tuple = (),
    *,
    orientation: Optional[int],
) -> ImageBuffer:
    """*img* flat-fielded as the render's source is. ``gain_slices`` are the ``slice_half`` cuts
    that took *img* out of the decoded frame. A stitch decode corrects its parts, and the
    embedded-lens decode applies the flat field before its warp."""
    if config.stitch.stitch_enabled or metadata_lens_corrections(config):
        return img
    gain_crop = (lambda g: slice_chain(g, gain_slices)) if gain_slices else None
    return apply_flatfield(img, config.flatfield, gain_crop, orientation=orientation)


def lens_decode_token(corrections: LensCorrections, flatfield: FlatFieldConfig) -> str:
    return f"|embedded-lens-v3-d{int(corrections.distortion)}-ca{int(corrections.ca)}" + flatfield_token(flatfield) if corrections else ""


def prepare_lens_source(
    img: ImageBuffer,
    metadata: dict,
    flatfield: FlatFieldConfig,
    corrections: LensCorrections = LensCorrections(True, True),
) -> ImageBuffer:
    """Flat-field in sensor positions before a lens warp moves the samples."""
    img = apply_flatfield(img, flatfield, orientation=metadata.get("orientation", 1))
    lens = metadata.get("lens_correction")
    if isinstance(lens, LensMetadata):
        img = apply_lens(img, lens, metadata.get("orientation", 1), corrections)
    return img


def prepare_lens_ir(ir: Optional[np.ndarray], metadata: dict, corrections: LensCorrections) -> Optional[np.ndarray]:
    """The IR plane warped with the source (see `prepare_lens_source`), so it stays registered."""
    lens = metadata.get("lens_correction")
    if ir is None or not isinstance(lens, LensMetadata):
        return ir
    return apply_lens(ir, lens, metadata.get("orientation", 1), corrections)


def source_ir(metadata: dict, shape: tuple[int, ...]) -> Optional[np.ndarray]:
    """The decode's IR plane, oriented, at the oriented buffer's `shape`. A camera IR frame
    decodes at half size and is scaled here; a plane of another size from any other carrier
    belongs to another frame and is dropped."""
    ir = metadata.get("ir")
    if ir is None:
        return None
    ir = apply_exif_orientation(ir, metadata.get("orientation", 1))
    h, w = shape[:2]
    if ir.shape[:2] == (h, w):
        return ir
    if metadata.get("ir_companion"):
        return downsample_ir(ir, max(h, w), dims=(w, h))
    return None
