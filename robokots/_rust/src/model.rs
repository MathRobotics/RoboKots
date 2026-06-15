use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::spatial::*;

pub(crate) fn get_usize(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<usize> {
    dict.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing key: {key}")))?
        .extract::<usize>()
}

pub(crate) fn get_string(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<String> {
    dict.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing key: {key}")))?
        .extract::<String>()
}

pub(crate) fn get_vec3_default(
    dict: &Bound<'_, PyDict>,
    key: &str,
    default: [f64; 3],
) -> PyResult<[f64; 3]> {
    match dict.get_item(key)? {
        Some(value) => {
            let vec = value.extract::<Vec<f64>>()?;
            if vec.len() != 3 {
                return Err(PyValueError::new_err(format!("{key} must have length 3")));
            }
            Ok([vec[0], vec[1], vec[2]])
        }
        None => Ok(default),
    }
}

pub(crate) fn get_vec4_default(
    dict: &Bound<'_, PyDict>,
    key: &str,
    default: [f64; 4],
) -> PyResult<[f64; 4]> {
    match dict.get_item(key)? {
        Some(value) => {
            let vec = value.extract::<Vec<f64>>()?;
            if vec.len() != 4 {
                return Err(PyValueError::new_err(format!("{key} must have length 4")));
            }
            Ok([vec[0], vec[1], vec[2], vec[3]])
        }
        None => Ok(default),
    }
}

pub(crate) fn spatial_inertia_from_link(link: &Bound<'_, PyDict>) -> PyResult<[[f64; 6]; 6]> {
    let mass = match link.get_item("mass")? {
        Some(value) => value.extract::<f64>()?,
        None => 0.0,
    };
    let cog = get_vec3_default(link, "cog", [0.0, 0.0, 0.0])?;
    let iv = match link.get_item("inertia")? {
        Some(value) => {
            let vec = value.extract::<Vec<f64>>()?;
            if vec.len() != 6 {
                return Err(PyValueError::new_err("inertia must have length 6"));
            }
            [vec[0], vec[1], vec[2], vec[3], vec[4], vec[5]]
        }
        None => [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    };
    let inertia = [
        [iv[0], iv[3], iv[4]],
        [iv[3], iv[1], iv[5]],
        [iv[4], iv[5], iv[2]],
    ];
    let c_hat = skew(cog);
    let c2 = mat3_mul(c_hat, c_hat);
    let mut out = [[0.0; 6]; 6];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = inertia[r][c] - mass * c2[r][c];
            out[r + 3][c + 3] = if r == c { mass } else { 0.0 };
            out[r + 3][c] = mass * c_hat[r][c];
            out[r][c + 3] = -mass * c_hat[r][c];
        }
    }
    Ok(out)
}
