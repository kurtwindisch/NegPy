use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::strip::{self, Roi};

/// `find_strip_frame(img, slide)` on an HxW or HxWxC float32 image: (roi, film, angle), each roi
/// (y1, y2, x1, x2) in the image's coordinates, or None when the scan shows no strip marks.
#[pyfunction]
fn find_strip_frame(
    py: Python<'_>,
    img: PyReadonlyArrayDyn<'_, f32>,
    slide: bool,
) -> PyResult<Option<(Roi, Roi, f64)>> {
    let shape = img.shape();
    let (h, w, cn) = match *shape {
        [h, w] => (h, w, 1),
        [h, w, cn] => (h, w, cn),
        _ => return Err(PyValueError::new_err("needs an HxW or HxWxC image")),
    };
    let data = img
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous image"))?;
    let found = super::detach(py, data.len(), || {
        strip::find_strip_frame(data, h, w, cn, slide)
    });
    Ok(found.map(|f| (f.roi, f.film, f.angle)))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_strip_frame, m)?)
}
