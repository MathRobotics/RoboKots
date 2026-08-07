use numpy::{
    ndarray::{Array, Dimension, Ix1, Ix2, Ix3},
    Element, IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyArrayMethods,
    PyReadonlyArray as NumpyReadonlyArray, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};

use crate::model::*;
use crate::pinocchio_like::PinocchioLikeWorkspace;
use crate::spatial::*;
use crate::types::{RustBatchOutwardData, RustCompiledRobot, RustFastData, RustOutwardData};
use crate::workspace::{BulkDerivativeWorkspace, CmtmWorkspace, DynamicsCmtmWorkspace, Workspace};

/// Read-only NumPy input with a guaranteed row-major logical layout.
///
/// Standard-layout arrays remain zero-copy. Fortran-order and strided arrays
/// are copied once, in logical index order, before algorithms consume them as
/// flat row-major slices.
struct RowMajorArray<'py, T, D>
where
    T: Element + Clone,
    D: Dimension,
{
    source: NumpyReadonlyArray<'py, T, D>,
    owned: Option<Array<T, D>>,
}

impl<'py, T, D> FromPyObject<'py> for RowMajorArray<'py, T, D>
where
    T: Element + Clone,
    D: Dimension,
{
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let source = NumpyReadonlyArray::extract_bound(obj)?;
        let view = source.as_array();
        let owned = (!view.is_standard_layout()).then(|| view.as_standard_layout().into_owned());
        Ok(Self { source, owned })
    }
}

impl<T, D> RowMajorArray<'_, T, D>
where
    T: Element + Clone,
    D: Dimension,
{
    fn shape(&self) -> &[usize] {
        self.source.shape()
    }

    fn len(&self) -> usize {
        self.source.len()
    }

    fn as_slice(&self) -> PyResult<&[T]> {
        if let Some(owned) = &self.owned {
            return Ok(owned
                .as_slice()
                .expect("owned row-major NumPy input must have a contiguous slice"));
        }
        Ok(self.source.as_slice()?)
    }
}

type PyReadonlyArray1<'py, T> = RowMajorArray<'py, T, Ix1>;
type PyReadonlyArray2<'py, T> = RowMajorArray<'py, T, Ix2>;
type PyReadonlyArray3<'py, T> = RowMajorArray<'py, T, Ix3>;

fn gravity_vec3(gravity: Option<PyReadonlyArray1<'_, f64>>) -> PyResult<[f64; 3]> {
    let Some(gravity) = gravity else {
        return Ok([0.0; 3]);
    };
    let gravity = gravity.as_slice()?;
    if gravity.len() != 3 {
        return Err(PyValueError::new_err("gravity must have shape (3,)"));
    }
    if !gravity.iter().all(|value| value.is_finite()) {
        return Err(PyValueError::new_err(
            "gravity must contain only finite values",
        ));
    }
    Ok([gravity[0], gravity[1], gravity[2]])
}

fn cmtm_world_wrench_value(
    elem_mat: [[f64; 4]; 4],
    elem_vecs: &[f64],
    raw_vecs: &[f64],
    key_order: usize,
) -> Vec<f64> {
    let mut fact = vec![1.0; key_order.max(1)];
    fill_factorial_table(&mut fact);
    let mut cm_rhs = vec![0.0; key_order * 6];
    for k in 0..key_order {
        let scale = 1.0 / fact[k];
        for i in 0..6 {
            cm_rhs[k * 6 + i] = raw_vecs[k * 6 + i] * scale;
        }
    }
    let mut blocks = vec![[[0.0; 6]; 6]; key_order];
    let mut out_cm = vec![0.0; key_order * 6];
    cmtm_apply_mat_adj_wrench_with_blocks_into(
        elem_mat,
        elem_vecs,
        &cm_rhs,
        key_order,
        &fact,
        &mut blocks,
        &mut out_cm,
    );
    let start = (key_order - 1) * 6;
    let scale = fact[key_order - 1];
    out_cm[start..start + 6]
        .iter()
        .map(|value| value * scale)
        .collect()
}

fn order1_link_momentum_value(
    robot: &RustCompiledRobot,
    ws: &DynamicsCmtmWorkspace,
    link_id: usize,
    vec_index: usize,
) -> [f64; 6] {
    let link_vel = cmtm_vecs_slice(&ws.cmtm.link_vecs, link_id, 3);
    mat6_vec6(
        robot.link_inertia[link_id],
        vec6_from_flat(link_vel, vec_index),
    )
}

fn order1_link_force_value(
    robot: &RustCompiledRobot,
    ws: &DynamicsCmtmWorkspace,
    link_id: usize,
) -> [f64; 6] {
    let link_vel = cmtm_vecs_slice(&ws.cmtm.link_vecs, link_id, 3);
    let v0 = vec6_from_flat(link_vel, 0);
    let m0 = mat6_vec6(robot.link_inertia[link_id], v0);
    let m1 = mat6_vec6(robot.link_inertia[link_id], vec6_from_flat(link_vel, 1));
    add6(m1, hat_adj_wrench_vec6(v0, m0))
}

fn order1_joint_force_value(
    robot: &RustCompiledRobot,
    ws: &DynamicsCmtmWorkspace,
    joint_id: usize,
) -> [f64; 6] {
    let child = robot.child_link[joint_id];
    let link_vel = cmtm_vecs_slice(&ws.cmtm.link_vecs, child, 3);
    let v0 = vec6_from_flat(link_vel, 0);
    let start = joint_id * 2 * 6;
    let m0 = vec6_from_flat(&ws.joint_momentum[start..start + 12], 0);
    let m1 = vec6_from_flat(&ws.joint_momentum[start..start + 12], 1);
    add6(m1, hat_adj_wrench_vec6(v0, m0))
}

fn order1_joint_mat_value(
    robot: &RustCompiledRobot,
    ws: &DynamicsCmtmWorkspace,
    joint_id: usize,
) -> [[f64; 4]; 4] {
    let q_index = robot.q_index[joint_id];
    if q_index < 0 {
        return eye4();
    }
    let q = ws.cached_motion[q_index as usize * 3];
    mat4_from_rot_pos(rot_axis(robot.axis[joint_id], q), [0.0; 3])
}

fn order1_joint_vec_value(
    robot: &RustCompiledRobot,
    ws: &DynamicsCmtmWorkspace,
    joint_id: usize,
    vec_index: usize,
) -> [f64; 6] {
    let q_index = robot.q_index[joint_id];
    if q_index < 0 {
        return [0.0; 6];
    }
    let value = ws.cached_motion[q_index as usize * 3 + vec_index + 1];
    [
        robot.axis[joint_id][0] * value,
        robot.axis[joint_id][1] * value,
        robot.axis[joint_id][2] * value,
        0.0,
        0.0,
        0.0,
    ]
}

fn fill_link_local_jacobian(
    robot: &RustCompiledRobot,
    base: &Workspace,
    deriv: &BulkDerivativeWorkspace,
    link_ids: &[i64],
    data_codes: &[i64],
    out: &mut [f64],
) -> PyResult<()> {
    let cols = deriv.cols;
    for (state_index, (&link_id_raw, &data_code)) in
        link_ids.iter().zip(data_codes.iter()).enumerate()
    {
        if link_id_raw < 0 || link_id_raw as usize >= robot.link_num {
            return Err(PyValueError::new_err("invalid link id"));
        }
        if !(0..=4).contains(&data_code) {
            return Err(PyValueError::new_err(
                "data_codes must be 0 vel, 1 acc, 2 momentum, 3 momentum_diff1, or 4 force",
            ));
        }
        let link_id = link_id_raw as usize;
        let r = mat3_from_flat(&base.r, link_id);
        let rt = mat3_transpose(r);
        let w = flat3(&base.w, link_id);
        let lin_v = flat3(&base.lin_v, link_id);
        let alpha = flat3(&base.alpha, link_id);
        let lin_a = flat3(&base.lin_a, link_id);
        let local_w = mat3_vec(rt, w);
        let local_lin_v = mat3_vec(rt, lin_v);
        let local_v0 = [
            local_w[0],
            local_w[1],
            local_w[2],
            local_lin_v[0],
            local_lin_v[1],
            local_lin_v[2],
        ];
        let m0 = mat6_vec6(robot.link_inertia[link_id], local_v0);
        for col in 0..cols {
            let dr = mat3_col(&deriv.r, link_id, col, cols);
            let drt = mat3_transpose(dr);
            let dw = flat3_col(&deriv.w, link_id, col, cols);
            let dlin_v = flat3_col(&deriv.lin_v, link_id, col, cols);
            let dlocal_w = add3(mat3_vec(drt, w), mat3_vec(rt, dw));
            let dlocal_lin_v = add3(mat3_vec(drt, lin_v), mat3_vec(rt, dlin_v));
            let dalpha = flat3_col(&deriv.alpha, link_id, col, cols);
            let dlin_a = flat3_col(&deriv.lin_a, link_id, col, cols);
            let dlocal_alpha = add3(mat3_vec(drt, alpha), mat3_vec(rt, dalpha));
            let dlocal_lin_a = sub3(
                add3(mat3_vec(drt, lin_a), mat3_vec(rt, dlin_a)),
                add3(
                    cross(dlocal_w, local_lin_v),
                    cross(local_w, dlocal_lin_v),
                ),
            );
            let dlocal_v0 = [
                dlocal_w[0],
                dlocal_w[1],
                dlocal_w[2],
                dlocal_lin_v[0],
                dlocal_lin_v[1],
                dlocal_lin_v[2],
            ];
            let dlocal_v1 = [
                dlocal_alpha[0],
                dlocal_alpha[1],
                dlocal_alpha[2],
                dlocal_lin_a[0],
                dlocal_lin_a[1],
                dlocal_lin_a[2],
            ];
            let dm0 = mat6_vec6(robot.link_inertia[link_id], dlocal_v0);
            let dm1 = mat6_vec6(robot.link_inertia[link_id], dlocal_v1);
            let flat_value = match data_code {
                0 => dlocal_v0,
                1 => dlocal_v1,
                2 => dm0,
                3 => dm1,
                4 => add6(
                    dm1,
                    add6(
                        hat_adj_wrench_vec6(dlocal_v0, m0),
                        hat_adj_wrench_vec6(local_v0, dm0),
                    ),
                ),
                _ => unreachable!(),
            };
            let row_base = state_index * 6;
            for i in 0..6 {
                out[(row_base + i) * cols + col] = flat_value[i];
            }
        }
    }
    Ok(())
}

#[pymethods]
impl RustCompiledRobot {
    #[pyo3(signature = (model_data, allow_prismatic = false))]
    #[staticmethod]
    fn from_model_data(model_data: &Bound<'_, PyDict>, allow_prismatic: bool) -> PyResult<Self> {
        let links_any = model_data
            .get_item("links")?
            .ok_or_else(|| PyValueError::new_err("model_data must contain links"))?;
        let joints_any = model_data
            .get_item("joints")?
            .ok_or_else(|| PyValueError::new_err("model_data must contain joints"))?;
        let links = links_any.downcast::<PyList>()?;
        let joints = joints_any.downcast::<PyList>()?;

        let link_num = links.len();
        let joint_num = joints.len();
        let mut parent_link = vec![0usize; joint_num];
        let mut child_link = vec![0usize; joint_num];
        let mut q_index = vec![-1isize; joint_num];
        let mut is_prismatic = vec![false; joint_num];
        let mut axis = vec![[1.0, 0.0, 0.0]; joint_num];
        let mut origin_r = vec![eye3(); joint_num];
        let mut origin_p = vec![[0.0, 0.0, 0.0]; joint_num];
        let mut link_ancestors = vec![Vec::<usize>::new(); link_num];
        let mut link_child_joints = vec![Vec::<usize>::new(); link_num];
        let mut dof = 0usize;

        for (i, joint_any) in joints.iter().enumerate() {
            let joint = joint_any.downcast::<PyDict>()?;
            parent_link[i] = get_usize(joint, "parent_link_id")?;
            child_link[i] = get_usize(joint, "child_link_id")?;
            link_child_joints[parent_link[i]].push(i);
            let joint_type = get_string(joint, "type")?;
            let origin = joint.get_item("origin")?;
            if let Some(origin_any) = origin {
                let origin_dict = origin_any.downcast::<PyDict>()?;
                origin_p[i] = get_vec3_default(origin_dict, "position", [0.0, 0.0, 0.0])?;
                origin_r[i] = quat_to_rot(get_vec4_default(
                    origin_dict,
                    "orientation",
                    [1.0, 0.0, 0.0, 0.0],
                )?);
            }

            if joint_type == "fixed" {
                link_ancestors[child_link[i]] = link_ancestors[parent_link[i]].clone();
                continue;
            }
            if joint_type == "spherical" {
                let q_representation = match joint.get_item("q_representation")? {
                    Some(value) => value.extract::<String>()?,
                    None => String::new(),
                };
                if q_representation != "rotation_vector" {
                    return Err(PyValueError::new_err(format!(
                        "spherical joints require q_representation='rotation_vector'"
                    )));
                }
                return Err(PyValueError::new_err(
                    "Rust backend currently supports fixed/revolute joints only; spherical/floating joints are supported by the Python backend",
                ));
            }
            if joint_type == "floating" {
                let q_representation = match joint.get_item("q_representation")? {
                    Some(value) => value.extract::<String>()?,
                    None => String::new(),
                };
                if q_representation != "expmap" {
                    return Err(PyValueError::new_err(
                        "floating joints require q_representation='expmap'",
                    ));
                }
                return Err(PyValueError::new_err(
                    "Rust backend currently supports fixed/revolute joints only; spherical/floating joints are supported by the Python backend",
                ));
            }
            if joint_type == "prismatic" && !allow_prismatic {
                return Err(PyValueError::new_err(
                    "Rust backend currently supports fixed/revolute joints only; use the Python backend for prismatic or multi-DoF joints",
                ));
            }
            if joint_type != "revolute" && joint_type != "prismatic" {
                return Err(PyValueError::new_err(
                    "Rust RNEA supports fixed/revolute/prismatic joints only; use the Python backend for multi-DoF joints",
                ));
            }
            is_prismatic[i] = joint_type == "prismatic";
            axis[i] = normalize(get_vec3_default(joint, "axis", [0.0, 0.0, 1.0])?);
            q_index[i] = dof as isize;
            let mut ancestors = link_ancestors[parent_link[i]].clone();
            ancestors.push(dof);
            link_ancestors[child_link[i]] = ancestors;
            dof += 1;
        }

        let mut link_inertia = Vec::with_capacity(link_num);
        for link_any in links.iter() {
            let link = link_any.downcast::<PyDict>()?;
            link_inertia.push(spatial_inertia_from_link(link)?);
        }

        let link_motion_columns: Vec<Vec<usize>> = link_ancestors
            .iter()
            .map(|ancestors| {
                let mut cols = Vec::with_capacity(ancestors.len() * 3);
                for &qi in ancestors {
                    cols.push(3 * qi);
                    cols.push(3 * qi + 1);
                    cols.push(3 * qi + 2);
                }
                cols
            })
            .collect();
        let mut link_subtree_motion_columns = link_motion_columns.clone();
        for j in (0..joint_num).rev() {
            let parent = parent_link[j];
            let child = child_link[j];
            let child_cols = link_subtree_motion_columns[child].clone();
            merge_columns(&mut link_subtree_motion_columns[parent], &child_cols);
        }

        Ok(Self {
            link_num,
            joint_num,
            dof,
            parent_link,
            child_link,
            q_index,
            is_prismatic,
            axis,
            origin_r,
            origin_p,
            link_inertia,
            link_ancestors,
            link_motion_columns,
            link_subtree_motion_columns,
            link_child_joints,
        })
    }

    #[getter]
    fn dof(&self) -> usize {
        self.dof
    }

    #[getter]
    fn link_num(&self) -> usize {
        self.link_num
    }

    #[getter]
    fn joint_num(&self) -> usize {
        self.joint_num
    }

    fn create_outward_data(&self, order: usize) -> PyResult<RustOutwardData> {
        if order < 1 {
            return Err(PyValueError::new_err("order must be >= 1"));
        }
        let dynamics_order = order.saturating_sub(2);
        Ok(RustOutwardData {
            robot: self.clone(),
            order,
            dynamics_order,
            kinematics: CmtmWorkspace::new(self, order),
            dynamics: DynamicsCmtmWorkspace::new(self, dynamics_order),
            has_kinematics: false,
            has_dynamics: false,
            has_cached_order1_dynamics: false,
        })
    }

    fn create_fast_data(&self) -> RustFastData {
        self.create_pinocchio_like_data()
    }

    fn create_pinocchio_like_data(&self) -> RustFastData {
        RustFastData {
            robot: self.clone(),
            workspace: PinocchioLikeWorkspace::new(self),
            has_kinematics: false,
            has_dynamics: false,
            has_joint_jacobians: false,
        }
    }

    fn create_batch_outward_data(
        &self,
        order: usize,
        batch: usize,
    ) -> PyResult<RustBatchOutwardData> {
        if order < 1 {
            return Err(PyValueError::new_err("order must be >= 1"));
        }
        let dynamics_order = order.saturating_sub(2);
        let mut kinematics = Vec::with_capacity(batch);
        let mut dynamics = Vec::with_capacity(batch);
        for _ in 0..batch {
            kinematics.push(CmtmWorkspace::new(self, order));
            dynamics.push(DynamicsCmtmWorkspace::new(self, dynamics_order));
        }
        Ok(RustBatchOutwardData {
            robot: self.clone(),
            order,
            dynamics_order,
            batch,
            kinematics,
            dynamics,
            has_kinematics: false,
            has_dynamics: false,
            has_cached_order1_dynamics: false,
        })
    }

    fn forward_kinematics<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray1<'py, f64>,
        v: PyReadonlyArray1<'py, f64>,
        a: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(
        Bound<'py, PyArray3<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
    )> {
        let q = q.as_slice()?;
        let v = v.as_slice()?;
        let a = a.as_slice()?;
        self.check_motion(q, v, a)?;
        let mut ws = Workspace::new(self);
        self.forward_kinematics_into(q, v, a, &mut ws);
        Ok((
            ws.r.into_pyarray(py).reshape([self.link_num, 3, 3])?,
            ws.p.into_pyarray(py).reshape([self.link_num, 3])?,
            ws.w.into_pyarray(py).reshape([self.link_num, 3])?,
            ws.lin_v.into_pyarray(py).reshape([self.link_num, 3])?,
            ws.alpha.into_pyarray(py).reshape([self.link_num, 3])?,
            ws.lin_a.into_pyarray(py).reshape([self.link_num, 3])?,
        ))
    }

    fn forward_kinematics_batch<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray2<'py, f64>,
        v: PyReadonlyArray2<'py, f64>,
        a: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(
        Bound<'py, PyArray4<f64>>,
        Bound<'py, PyArray3<f64>>,
        Bound<'py, PyArray3<f64>>,
        Bound<'py, PyArray3<f64>>,
        Bound<'py, PyArray3<f64>>,
        Bound<'py, PyArray3<f64>>,
    )> {
        let batch = self.check_motion_batch(q.shape(), v.shape(), a.shape())?;
        let q = q.as_slice()?;
        let v = v.as_slice()?;
        let a = a.as_slice()?;
        let motion_len = self.dof;
        let r_len = self.link_num * 9;
        let vec_len = self.link_num * 3;
        let mut r = vec![0.0; batch * r_len];
        let mut p = vec![0.0; batch * vec_len];
        let mut w = vec![0.0; batch * vec_len];
        let mut lin_v = vec![0.0; batch * vec_len];
        let mut alpha = vec![0.0; batch * vec_len];
        let mut lin_a = vec![0.0; batch * vec_len];
        let mut ws = Workspace::new(self);

        for sample in 0..batch {
            let motion_start = sample * motion_len;
            let motion_end = motion_start + motion_len;
            self.forward_kinematics_into(
                &q[motion_start..motion_end],
                &v[motion_start..motion_end],
                &a[motion_start..motion_end],
                &mut ws,
            );
            r[sample * r_len..(sample + 1) * r_len].copy_from_slice(&ws.r);
            p[sample * vec_len..(sample + 1) * vec_len].copy_from_slice(&ws.p);
            w[sample * vec_len..(sample + 1) * vec_len].copy_from_slice(&ws.w);
            lin_v[sample * vec_len..(sample + 1) * vec_len].copy_from_slice(&ws.lin_v);
            alpha[sample * vec_len..(sample + 1) * vec_len].copy_from_slice(&ws.alpha);
            lin_a[sample * vec_len..(sample + 1) * vec_len].copy_from_slice(&ws.lin_a);
        }

        Ok((
            r.into_pyarray(py).reshape([batch, self.link_num, 3, 3])?,
            p.into_pyarray(py).reshape([batch, self.link_num, 3])?,
            w.into_pyarray(py).reshape([batch, self.link_num, 3])?,
            lin_v.into_pyarray(py).reshape([batch, self.link_num, 3])?,
            alpha.into_pyarray(py).reshape([batch, self.link_num, 3])?,
            lin_a.into_pyarray(py).reshape([batch, self.link_num, 3])?,
        ))
    }

    #[pyo3(signature = (q, v, a, gravity = None))]
    fn rnea<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray1<'py, f64>,
        v: PyReadonlyArray1<'py, f64>,
        a: PyReadonlyArray1<'py, f64>,
        gravity: Option<PyReadonlyArray1<'py, f64>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let q = q.as_slice()?;
        let v = v.as_slice()?;
        let a = a.as_slice()?;
        self.check_motion(q, v, a)?;
        let gravity = gravity_vec3(gravity)?;
        let mut ws = Workspace::new(self);
        self.rnea_with_gravity_into(q, v, a, gravity, &mut ws);
        Ok(ws.tau.into_pyarray(py))
    }

    #[pyo3(signature = (q, v, a, gravity = None))]
    fn rnea_batch<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray2<'py, f64>,
        v: PyReadonlyArray2<'py, f64>,
        a: PyReadonlyArray2<'py, f64>,
        gravity: Option<PyReadonlyArray1<'py, f64>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let batch = self.check_motion_batch(q.shape(), v.shape(), a.shape())?;
        let q = q.as_slice()?;
        let v = v.as_slice()?;
        let a = a.as_slice()?;
        let gravity = gravity_vec3(gravity)?;
        let mut out = vec![0.0; batch * self.dof];
        let mut ws = Workspace::new(self);
        for sample in 0..batch {
            let start = sample * self.dof;
            let end = start + self.dof;
            self.rnea_with_gravity_into(
                &q[start..end],
                &v[start..end],
                &a[start..end],
                gravity,
                &mut ws,
            );
            out[start..end].copy_from_slice(&ws.tau);
        }
        Ok(out.into_pyarray(py).reshape([batch, self.dof])?)
    }

    #[pyo3(signature = (q, v, a, eps = 1e-6))]
    fn rnea_jacobian<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray1<'py, f64>,
        v: PyReadonlyArray1<'py, f64>,
        a: PyReadonlyArray1<'py, f64>,
        eps: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let _ = eps;
        let q = q.as_slice()?;
        let v = v.as_slice()?;
        let a = a.as_slice()?;
        self.check_motion(q, v, a)?;
        let jac = self.rnea_jacobian_into(q, v, a, false);
        Ok(jac.into_pyarray(py).reshape([self.dof, 3 * self.dof])?)
    }

    #[pyo3(signature = (q, v, a, eps = 1e-6))]
    fn dynamics_jacobian<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray1<'py, f64>,
        v: PyReadonlyArray1<'py, f64>,
        a: PyReadonlyArray1<'py, f64>,
        eps: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let _ = eps;
        let q = q.as_slice()?;
        let v = v.as_slice()?;
        let a = a.as_slice()?;
        self.check_motion(q, v, a)?;
        let jac = self.rnea_jacobian_into(q, v, a, true);
        Ok(jac.into_pyarray(py).reshape([self.dof, 3 * self.dof])?)
    }

    #[pyo3(signature = (q, v, a, eps = 1e-6))]
    fn dynamics_jacobian_batch<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray2<'py, f64>,
        v: PyReadonlyArray2<'py, f64>,
        a: PyReadonlyArray2<'py, f64>,
        eps: f64,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let _ = eps;
        let batch = self.check_motion_batch(q.shape(), v.shape(), a.shape())?;
        let q = q.as_slice()?;
        let v = v.as_slice()?;
        let a = a.as_slice()?;
        let cols = 3 * self.dof;
        let mut out = vec![0.0; batch * self.dof * cols];
        let motion_cols = &self.link_motion_columns;
        let mut base = Workspace::new(self);
        let mut deriv = BulkDerivativeWorkspace::new(self, cols);
        for sample in 0..batch {
            let motion_start = sample * self.dof;
            let motion_end = motion_start + self.dof;
            let out_start = sample * self.dof * cols;
            self.rnea_jacobian_bulk_interleaved_fill(
                &q[motion_start..motion_end],
                &v[motion_start..motion_end],
                &a[motion_start..motion_end],
                motion_cols,
                &mut base,
                &mut deriv,
                &mut out[out_start..out_start + self.dof * cols],
            );
        }
        Ok(out.into_pyarray(py).reshape([batch, self.dof, cols])?)
    }

    fn dynamics_jacobian_matmul_rhs<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray1<'py, f64>,
        v: PyReadonlyArray1<'py, f64>,
        a: PyReadonlyArray1<'py, f64>,
        rhs: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let q = q.as_slice()?;
        let v = v.as_slice()?;
        let a = a.as_slice()?;
        self.check_motion(q, v, a)?;
        let rhs_shape = rhs.shape();
        if rhs_shape.len() != 2 || rhs_shape[0] != 3 * self.dof {
            return Err(PyValueError::new_err(
                "rhs must have shape (3 * robot dof, rhs_cols)",
            ));
        }
        let rhs_cols = rhs_shape[1];
        let rhs = rhs.as_slice()?;
        let mut base = Workspace::new(self);
        let mut deriv = BulkDerivativeWorkspace::new(self, rhs_cols);
        let mut out = vec![0.0; self.dof * rhs_cols];
        self.rnea_jacobian_matmul_interleaved_fill(
            q,
            v,
            a,
            rhs,
            rhs_cols,
            &mut base,
            &mut deriv,
            &mut out,
        );
        Ok(out.into_pyarray(py).reshape([self.dof, rhs_cols])?)
    }

    fn dynamics_jacobian_matmul_rhs_batch<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray2<'py, f64>,
        v: PyReadonlyArray2<'py, f64>,
        a: PyReadonlyArray2<'py, f64>,
        rhs: PyReadonlyArray3<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let batch = self.check_motion_batch(q.shape(), v.shape(), a.shape())?;
        let rhs_shape = rhs.shape();
        if rhs_shape.len() != 3 || rhs_shape[0] != batch || rhs_shape[1] != 3 * self.dof {
            return Err(PyValueError::new_err(
                "rhs batch must have shape (batch, 3 * robot dof, rhs_cols)",
            ));
        }
        let rhs_cols = rhs_shape[2];
        let q = q.as_slice()?;
        let v = v.as_slice()?;
        let a = a.as_slice()?;
        let rhs = rhs.as_slice()?;
        let mut base = Workspace::new(self);
        let mut deriv = BulkDerivativeWorkspace::new(self, rhs_cols);
        let mut out = vec![0.0; batch * self.dof * rhs_cols];
        for sample in 0..batch {
            let motion_start = sample * self.dof;
            let motion_end = motion_start + self.dof;
            let rhs_start = sample * 3 * self.dof * rhs_cols;
            let rhs_end = rhs_start + 3 * self.dof * rhs_cols;
            let out_start = sample * self.dof * rhs_cols;
            self.rnea_jacobian_matmul_interleaved_fill(
                &q[motion_start..motion_end],
                &v[motion_start..motion_end],
                &a[motion_start..motion_end],
                &rhs[rhs_start..rhs_end],
                rhs_cols,
                &mut base,
                &mut deriv,
                &mut out[out_start..out_start + self.dof * rhs_cols],
            );
        }
        Ok(out
            .into_pyarray(py)
            .reshape([batch, self.dof, rhs_cols])?)
    }

    fn dynamics_jacobian_transpose_matmul_rhs<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray1<'py, f64>,
        v: PyReadonlyArray1<'py, f64>,
        a: PyReadonlyArray1<'py, f64>,
        rhs: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let q = q.as_slice()?;
        let v = v.as_slice()?;
        let a = a.as_slice()?;
        self.check_motion(q, v, a)?;
        let rhs_shape = rhs.shape();
        if rhs_shape.len() != 2 || rhs_shape[0] != self.dof {
            return Err(PyValueError::new_err(
                "rhs must have shape (robot dof, rhs_cols)",
            ));
        }
        let rhs_cols = rhs_shape[1];
        let rhs = rhs.as_slice()?;
        let input_cols = 3 * self.dof;
        let jac = self.rnea_jacobian_bulk_interleaved_into(q, v, a);
        let mut out = vec![0.0; input_cols * rhs_cols];
        for row in 0..self.dof {
            for input_col in 0..input_cols {
                let value = jac[row * input_cols + input_col];
                for rhs_col in 0..rhs_cols {
                    out[input_col * rhs_cols + rhs_col] += value * rhs[row * rhs_cols + rhs_col];
                }
            }
        }
        Ok(out.into_pyarray(py).reshape([input_cols, rhs_cols])?)
    }

    fn dynamics_jacobian_transpose_matmul_rhs_batch<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray2<'py, f64>,
        v: PyReadonlyArray2<'py, f64>,
        a: PyReadonlyArray2<'py, f64>,
        rhs: PyReadonlyArray3<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let batch = self.check_motion_batch(q.shape(), v.shape(), a.shape())?;
        let rhs_shape = rhs.shape();
        if rhs_shape.len() != 3 || rhs_shape[0] != batch || rhs_shape[1] != self.dof {
            return Err(PyValueError::new_err(
                "rhs batch must have shape (batch, robot dof, rhs_cols)",
            ));
        }
        let rhs_cols = rhs_shape[2];
        let q = q.as_slice()?;
        let v = v.as_slice()?;
        let a = a.as_slice()?;
        let rhs = rhs.as_slice()?;
        let input_cols = 3 * self.dof;
        let mut out = vec![0.0; batch * input_cols * rhs_cols];
        let motion_cols = &self.link_motion_columns;
        let mut base = Workspace::new(self);
        let mut deriv = BulkDerivativeWorkspace::new(self, input_cols);
        let mut jac = vec![0.0; self.dof * input_cols];
        for sample in 0..batch {
            let motion_start = sample * self.dof;
            let motion_end = motion_start + self.dof;
            self.rnea_jacobian_bulk_interleaved_fill(
                &q[motion_start..motion_end],
                &v[motion_start..motion_end],
                &a[motion_start..motion_end],
                motion_cols,
                &mut base,
                &mut deriv,
                &mut jac,
            );
            let rhs_start = sample * self.dof * rhs_cols;
            let out_start = sample * input_cols * rhs_cols;
            for row in 0..self.dof {
                for input_col in 0..input_cols {
                    let value = jac[row * input_cols + input_col];
                    for rhs_col in 0..rhs_cols {
                        out[out_start + input_col * rhs_cols + rhs_col] +=
                            value * rhs[rhs_start + row * rhs_cols + rhs_col];
                    }
                }
            }
        }
        Ok(out
            .into_pyarray(py)
            .reshape([batch, input_cols, rhs_cols])?)
    }

    fn joint_jacobians<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let q = q.as_slice()?;
        if q.len() != self.dof {
            return Err(PyValueError::new_err("q length must match robot dof"));
        }
        let jac = self.joint_jacobians_vec(q);
        Ok(jac.into_pyarray(py).reshape([self.link_num, 6, self.dof])?)
    }

    fn joint_jacobians_batch<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray4<f64>>> {
        let shape = q.shape();
        if shape.len() != 2 || shape[1] != self.dof {
            return Err(PyValueError::new_err(
                "q batch shape must be (batch, robot dof)",
            ));
        }
        let batch = shape[0];
        let q = q.as_slice()?;
        let jac_len = self.link_num * 6 * self.dof;
        let mut out = vec![0.0; batch * jac_len];
        let mut ws = Workspace::new(self);
        for sample in 0..batch {
            let motion_start = sample * self.dof;
            let motion_end = motion_start + self.dof;
            self.joint_jacobians_into(&q[motion_start..motion_end], &mut ws);
            out[sample * jac_len..(sample + 1) * jac_len].copy_from_slice(&ws.jac);
        }
        Ok(out
            .into_pyarray(py)
            .reshape([batch, self.link_num, 6, self.dof])?)
    }

    fn link_local_jacobian<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray1<'py, f64>,
        v: PyReadonlyArray1<'py, f64>,
        a: PyReadonlyArray1<'py, f64>,
        link_ids: PyReadonlyArray1<'py, i64>,
        data_codes: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let q = q.as_slice()?;
        let v = v.as_slice()?;
        let a = a.as_slice()?;
        self.check_motion(q, v, a)?;
        let link_ids = link_ids.as_slice()?;
        let data_codes = data_codes.as_slice()?;
        if link_ids.len() != data_codes.len() {
            return Err(PyValueError::new_err(
                "link_ids and data_codes must have the same length",
            ));
        }
        let cols = 3 * self.dof;
        let mut base = Workspace::new(self);
        let mut deriv = BulkDerivativeWorkspace::new(self, cols);
        self.forward_kinematics_into(q, v, a, &mut base);
        deriv.clear();
        self.forward_kinematics_bulk_derivative_into(
            q,
            v,
            a,
            &base,
            &mut deriv,
            &self.link_motion_columns,
        );
        let mut out = vec![0.0; link_ids.len() * 6 * cols];
        fill_link_local_jacobian(self, &base, &deriv, link_ids, data_codes, &mut out)?;
        Ok(out.into_pyarray(py).reshape([link_ids.len() * 6, cols])?)
    }

    fn link_local_jacobian_matmul_rhs<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray1<'py, f64>,
        v: PyReadonlyArray1<'py, f64>,
        a: PyReadonlyArray1<'py, f64>,
        rhs: PyReadonlyArray2<'py, f64>,
        link_ids: PyReadonlyArray1<'py, i64>,
        data_codes: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let q = q.as_slice()?;
        let v = v.as_slice()?;
        let a = a.as_slice()?;
        self.check_motion(q, v, a)?;
        let rhs_shape = rhs.shape();
        if rhs_shape.len() != 2 || rhs_shape[0] != 3 * self.dof {
            return Err(PyValueError::new_err(
                "rhs must have shape (3 * robot dof, rhs_cols)",
            ));
        }
        let rhs_cols = rhs_shape[1];
        let rhs = rhs.as_slice()?;
        let link_ids = link_ids.as_slice()?;
        let data_codes = data_codes.as_slice()?;
        if link_ids.len() != data_codes.len() {
            return Err(PyValueError::new_err(
                "link_ids and data_codes must have the same length",
            ));
        }
        let mut base = Workspace::new(self);
        let mut deriv = BulkDerivativeWorkspace::new(self, rhs_cols);
        self.rnea_into(q, v, a, &mut base);
        deriv.clear();
        self.forward_kinematics_directional_derivative_into(
            q,
            v,
            a,
            rhs,
            rhs_cols,
            &base,
            &mut deriv,
        );
        let mut out = vec![0.0; link_ids.len() * 6 * rhs_cols];
        fill_link_local_jacobian(self, &base, &deriv, link_ids, data_codes, &mut out)?;
        Ok(out
            .into_pyarray(py)
            .reshape([link_ids.len() * 6, rhs_cols])?)
    }

    fn link_vel_acc_jacobian<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray1<'py, f64>,
        v: PyReadonlyArray1<'py, f64>,
        a: PyReadonlyArray1<'py, f64>,
        link_ids: PyReadonlyArray1<'py, i64>,
        data_codes: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.link_local_jacobian(py, q, v, a, link_ids, data_codes)
    }

    fn link_vel_acc_jacobian_matmul_rhs<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray1<'py, f64>,
        v: PyReadonlyArray1<'py, f64>,
        a: PyReadonlyArray1<'py, f64>,
        rhs: PyReadonlyArray2<'py, f64>,
        link_ids: PyReadonlyArray1<'py, i64>,
        data_codes: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.link_local_jacobian_matmul_rhs(py, q, v, a, rhs, link_ids, data_codes)
    }

    fn kinematics_cmtm<'py>(
        &self,
        py: Python<'py>,
        motion: PyReadonlyArray1<'py, f64>,
        order: usize,
    ) -> PyResult<(
        Bound<'py, PyArray3<f64>>,
        Bound<'py, PyArray3<f64>>,
        Bound<'py, PyArray3<f64>>,
        Bound<'py, PyArray3<f64>>,
    )> {
        if order < 1 {
            return Err(PyValueError::new_err("order must be >= 1"));
        }
        let motion = motion.as_slice()?;
        self.check_cmtm_motion(motion, order)?;
        let mut ws = CmtmWorkspace::new(self, order);
        self.kinematics_cmtm_into(motion, order, &mut ws);
        Ok((
            ws.link_mat
                .into_pyarray(py)
                .reshape([self.link_num, 4, 4])?,
            ws.link_vecs
                .into_pyarray(py)
                .reshape([self.link_num, order - 1, 6])?,
            ws.joint_mat
                .into_pyarray(py)
                .reshape([self.joint_num, 4, 4])?,
            ws.joint_vecs
                .into_pyarray(py)
                .reshape([self.joint_num, order - 1, 6])?,
        ))
    }

    fn kinematics_cmtm_batch<'py>(
        &self,
        py: Python<'py>,
        motions: PyReadonlyArray2<'py, f64>,
        order: usize,
    ) -> PyResult<(
        Bound<'py, PyArray4<f64>>,
        Bound<'py, PyArray4<f64>>,
        Bound<'py, PyArray4<f64>>,
        Bound<'py, PyArray4<f64>>,
    )> {
        if order < 1 {
            return Err(PyValueError::new_err("order must be >= 1"));
        }
        let shape = motions.shape();
        if shape.len() != 2 || shape[1] != self.dof * order {
            return Err(PyValueError::new_err(
                "motions batch shape must be (batch, robot dof * order)",
            ));
        }
        let batch = shape[0];
        let motions = motions.as_slice()?;
        let motion_len = self.dof * order;
        let link_mat_len = self.link_num * 16;
        let link_vec_len = self.link_num * (order - 1) * 6;
        let joint_mat_len = self.joint_num * 16;
        let joint_vec_len = self.joint_num * (order - 1) * 6;
        let mut link_mat = vec![0.0; batch * link_mat_len];
        let mut link_vecs = vec![0.0; batch * link_vec_len];
        let mut joint_mat = vec![0.0; batch * joint_mat_len];
        let mut joint_vecs = vec![0.0; batch * joint_vec_len];
        let mut ws = CmtmWorkspace::new(self, order);

        for sample in 0..batch {
            let motion_start = sample * motion_len;
            self.kinematics_cmtm_into(
                &motions[motion_start..motion_start + motion_len],
                order,
                &mut ws,
            );
            link_mat[sample * link_mat_len..(sample + 1) * link_mat_len]
                .copy_from_slice(&ws.link_mat);
            link_vecs[sample * link_vec_len..(sample + 1) * link_vec_len]
                .copy_from_slice(&ws.link_vecs);
            joint_mat[sample * joint_mat_len..(sample + 1) * joint_mat_len]
                .copy_from_slice(&ws.joint_mat);
            joint_vecs[sample * joint_vec_len..(sample + 1) * joint_vec_len]
                .copy_from_slice(&ws.joint_vecs);
        }

        Ok((
            link_mat
                .into_pyarray(py)
                .reshape([batch, self.link_num, 4, 4])?,
            link_vecs
                .into_pyarray(py)
                .reshape([batch, self.link_num, order - 1, 6])?,
            joint_mat
                .into_pyarray(py)
                .reshape([batch, self.joint_num, 4, 4])?,
            joint_vecs
                .into_pyarray(py)
                .reshape([batch, self.joint_num, order - 1, 6])?,
        ))
    }

    fn dynamics_cmtm<'py>(
        &self,
        py: Python<'py>,
        motion: PyReadonlyArray1<'py, f64>,
        dynamics_order: usize,
    ) -> PyResult<(
        Bound<'py, PyArray3<f64>>,
        Bound<'py, PyArray3<f64>>,
        Bound<'py, PyArray3<f64>>,
        Bound<'py, PyArray3<f64>>,
        Bound<'py, PyArray3<f64>>,
    )> {
        let kin_order = dynamics_order + 2;
        let motion = motion.as_slice()?;
        self.check_cmtm_motion(motion, kin_order)?;
        let mut ws = DynamicsCmtmWorkspace::new(self, dynamics_order);
        self.dynamics_cmtm_into(motion, dynamics_order, &mut ws);
        Ok((
            ws.link_momentum
                .into_pyarray(py)
                .reshape([self.link_num, dynamics_order + 1, 6])?,
            ws.link_force
                .into_pyarray(py)
                .reshape([self.link_num, dynamics_order, 6])?,
            ws.joint_momentum
                .into_pyarray(py)
                .reshape([self.joint_num, dynamics_order + 1, 6])?,
            ws.joint_force
                .into_pyarray(py)
                .reshape([self.joint_num, dynamics_order, 6])?,
            ws.joint_torque
                .into_pyarray(py)
                .reshape([self.joint_num, dynamics_order, 1])?,
        ))
    }

    fn dynamics_outward_cmtm<'py>(
        &self,
        py: Python<'py>,
        motion: PyReadonlyArray1<'py, f64>,
        dynamics_order: usize,
    ) -> PyResult<(
        Bound<'py, PyArray3<f64>>,
        Bound<'py, PyArray3<f64>>,
        Bound<'py, PyArray3<f64>>,
        Bound<'py, PyArray3<f64>>,
        Bound<'py, PyArray3<f64>>,
        Bound<'py, PyArray3<f64>>,
        Bound<'py, PyArray3<f64>>,
        Bound<'py, PyArray3<f64>>,
        Bound<'py, PyArray3<f64>>,
    )> {
        let kin_order = dynamics_order + 2;
        let motion = motion.as_slice()?;
        self.check_cmtm_motion(motion, kin_order)?;
        let mut ws = DynamicsCmtmWorkspace::new(self, dynamics_order);
        self.dynamics_cmtm_into(motion, dynamics_order, &mut ws);
        Ok((
            ws.cmtm
                .link_mat
                .into_pyarray(py)
                .reshape([self.link_num, 4, 4])?,
            ws.cmtm
                .link_vecs
                .into_pyarray(py)
                .reshape([self.link_num, kin_order - 1, 6])?,
            ws.cmtm
                .joint_mat
                .into_pyarray(py)
                .reshape([self.joint_num, 4, 4])?,
            ws.cmtm
                .joint_vecs
                .into_pyarray(py)
                .reshape([self.joint_num, kin_order - 1, 6])?,
            ws.link_momentum
                .into_pyarray(py)
                .reshape([self.link_num, dynamics_order + 1, 6])?,
            ws.link_force
                .into_pyarray(py)
                .reshape([self.link_num, dynamics_order, 6])?,
            ws.joint_momentum
                .into_pyarray(py)
                .reshape([self.joint_num, dynamics_order + 1, 6])?,
            ws.joint_force
                .into_pyarray(py)
                .reshape([self.joint_num, dynamics_order, 6])?,
            ws.joint_torque
                .into_pyarray(py)
                .reshape([self.joint_num, dynamics_order, 1])?,
        ))
    }

    fn dynamics_cmtm_batch<'py>(
        &self,
        py: Python<'py>,
        motions: PyReadonlyArray2<'py, f64>,
        dynamics_order: usize,
    ) -> PyResult<(
        Bound<'py, PyArray4<f64>>,
        Bound<'py, PyArray4<f64>>,
        Bound<'py, PyArray4<f64>>,
        Bound<'py, PyArray4<f64>>,
        Bound<'py, PyArray4<f64>>,
    )> {
        let kin_order = dynamics_order + 2;
        let shape = motions.shape();
        if shape.len() != 2 || shape[1] != self.dof * kin_order {
            return Err(PyValueError::new_err(
                "motions batch shape must be (batch, robot dof * (dynamics_order + 2))",
            ));
        }
        let batch = shape[0];
        let motions = motions.as_slice()?;
        let motion_len = self.dof * kin_order;
        let link_momentum_len = self.link_num * (dynamics_order + 1) * 6;
        let link_force_len = self.link_num * dynamics_order * 6;
        let joint_momentum_len = self.joint_num * (dynamics_order + 1) * 6;
        let joint_force_len = self.joint_num * dynamics_order * 6;
        let joint_torque_len = self.joint_num * dynamics_order;
        let mut link_momentum = vec![0.0; batch * link_momentum_len];
        let mut link_force = vec![0.0; batch * link_force_len];
        let mut joint_momentum = vec![0.0; batch * joint_momentum_len];
        let mut joint_force = vec![0.0; batch * joint_force_len];
        let mut joint_torque = vec![0.0; batch * joint_torque_len];
        let mut ws = DynamicsCmtmWorkspace::new(self, dynamics_order);

        for sample in 0..batch {
            let motion_start = sample * motion_len;
            self.dynamics_cmtm_into(
                &motions[motion_start..motion_start + motion_len],
                dynamics_order,
                &mut ws,
            );
            link_momentum[sample * link_momentum_len..(sample + 1) * link_momentum_len]
                .copy_from_slice(&ws.link_momentum);
            link_force[sample * link_force_len..(sample + 1) * link_force_len]
                .copy_from_slice(&ws.link_force);
            joint_momentum[sample * joint_momentum_len..(sample + 1) * joint_momentum_len]
                .copy_from_slice(&ws.joint_momentum);
            joint_force[sample * joint_force_len..(sample + 1) * joint_force_len]
                .copy_from_slice(&ws.joint_force);
            joint_torque[sample * joint_torque_len..(sample + 1) * joint_torque_len]
                .copy_from_slice(&ws.joint_torque);
        }

        Ok((
            link_momentum.into_pyarray(py).reshape([
                batch,
                self.link_num,
                dynamics_order + 1,
                6,
            ])?,
            link_force
                .into_pyarray(py)
                .reshape([batch, self.link_num, dynamics_order, 6])?,
            joint_momentum.into_pyarray(py).reshape([
                batch,
                self.joint_num,
                dynamics_order + 1,
                6,
            ])?,
            joint_force
                .into_pyarray(py)
                .reshape([batch, self.joint_num, dynamics_order, 6])?,
            joint_torque
                .into_pyarray(py)
                .reshape([batch, self.joint_num, dynamics_order, 1])?,
        ))
    }

    fn dynamics_outward_cmtm_batch<'py>(
        &self,
        py: Python<'py>,
        motions: PyReadonlyArray2<'py, f64>,
        dynamics_order: usize,
    ) -> PyResult<(
        Bound<'py, PyArray4<f64>>,
        Bound<'py, PyArray4<f64>>,
        Bound<'py, PyArray4<f64>>,
        Bound<'py, PyArray4<f64>>,
        Bound<'py, PyArray4<f64>>,
        Bound<'py, PyArray4<f64>>,
        Bound<'py, PyArray4<f64>>,
        Bound<'py, PyArray4<f64>>,
        Bound<'py, PyArray4<f64>>,
    )> {
        let kin_order = dynamics_order + 2;
        let shape = motions.shape();
        if shape.len() != 2 || shape[1] != self.dof * kin_order {
            return Err(PyValueError::new_err(
                "motions batch shape must be (batch, robot dof * (dynamics_order + 2))",
            ));
        }
        let batch = shape[0];
        let motions = motions.as_slice()?;
        let motion_len = self.dof * kin_order;
        let link_mat_len = self.link_num * 16;
        let link_vec_len = self.link_num * (kin_order - 1) * 6;
        let joint_mat_len = self.joint_num * 16;
        let joint_vec_len = self.joint_num * (kin_order - 1) * 6;
        let link_momentum_len = self.link_num * (dynamics_order + 1) * 6;
        let link_force_len = self.link_num * dynamics_order * 6;
        let joint_momentum_len = self.joint_num * (dynamics_order + 1) * 6;
        let joint_force_len = self.joint_num * dynamics_order * 6;
        let joint_torque_len = self.joint_num * dynamics_order;
        let mut link_mat = vec![0.0; batch * link_mat_len];
        let mut link_vecs = vec![0.0; batch * link_vec_len];
        let mut joint_mat = vec![0.0; batch * joint_mat_len];
        let mut joint_vecs = vec![0.0; batch * joint_vec_len];
        let mut link_momentum = vec![0.0; batch * link_momentum_len];
        let mut link_force = vec![0.0; batch * link_force_len];
        let mut joint_momentum = vec![0.0; batch * joint_momentum_len];
        let mut joint_force = vec![0.0; batch * joint_force_len];
        let mut joint_torque = vec![0.0; batch * joint_torque_len];
        let mut ws = DynamicsCmtmWorkspace::new(self, dynamics_order);

        for sample in 0..batch {
            let motion_start = sample * motion_len;
            self.dynamics_cmtm_into(
                &motions[motion_start..motion_start + motion_len],
                dynamics_order,
                &mut ws,
            );
            link_mat[sample * link_mat_len..(sample + 1) * link_mat_len]
                .copy_from_slice(&ws.cmtm.link_mat);
            link_vecs[sample * link_vec_len..(sample + 1) * link_vec_len]
                .copy_from_slice(&ws.cmtm.link_vecs);
            joint_mat[sample * joint_mat_len..(sample + 1) * joint_mat_len]
                .copy_from_slice(&ws.cmtm.joint_mat);
            joint_vecs[sample * joint_vec_len..(sample + 1) * joint_vec_len]
                .copy_from_slice(&ws.cmtm.joint_vecs);
            link_momentum[sample * link_momentum_len..(sample + 1) * link_momentum_len]
                .copy_from_slice(&ws.link_momentum);
            link_force[sample * link_force_len..(sample + 1) * link_force_len]
                .copy_from_slice(&ws.link_force);
            joint_momentum[sample * joint_momentum_len..(sample + 1) * joint_momentum_len]
                .copy_from_slice(&ws.joint_momentum);
            joint_force[sample * joint_force_len..(sample + 1) * joint_force_len]
                .copy_from_slice(&ws.joint_force);
            joint_torque[sample * joint_torque_len..(sample + 1) * joint_torque_len]
                .copy_from_slice(&ws.joint_torque);
        }

        Ok((
            link_mat
                .into_pyarray(py)
                .reshape([batch, self.link_num, 4, 4])?,
            link_vecs
                .into_pyarray(py)
                .reshape([batch, self.link_num, kin_order - 1, 6])?,
            joint_mat
                .into_pyarray(py)
                .reshape([batch, self.joint_num, 4, 4])?,
            joint_vecs
                .into_pyarray(py)
                .reshape([batch, self.joint_num, kin_order - 1, 6])?,
            link_momentum.into_pyarray(py).reshape([
                batch,
                self.link_num,
                dynamics_order + 1,
                6,
            ])?,
            link_force
                .into_pyarray(py)
                .reshape([batch, self.link_num, dynamics_order, 6])?,
            joint_momentum.into_pyarray(py).reshape([
                batch,
                self.joint_num,
                dynamics_order + 1,
                6,
            ])?,
            joint_force
                .into_pyarray(py)
                .reshape([batch, self.joint_num, dynamics_order, 6])?,
            joint_torque
                .into_pyarray(py)
                .reshape([batch, self.joint_num, dynamics_order, 1])?,
        ))
    }
}

#[pymethods]
impl RustFastData {
    fn compute_kinematics(
        &mut self,
        q: PyReadonlyArray1<'_, f64>,
        v: PyReadonlyArray1<'_, f64>,
        a: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<()> {
        let q = q.as_slice()?;
        let v = v.as_slice()?;
        let a = a.as_slice()?;
        self.robot.check_motion(q, v, a)?;
        self.robot
            .pinocchio_forward_kinematics_into(q, v, a, &mut self.workspace);
        self.has_kinematics = true;
        self.has_dynamics = false;
        self.has_joint_jacobians = false;
        Ok(())
    }

    fn compute_dynamics(
        &mut self,
        q: PyReadonlyArray1<'_, f64>,
        v: PyReadonlyArray1<'_, f64>,
        a: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<()> {
        let q = q.as_slice()?;
        let v = v.as_slice()?;
        let a = a.as_slice()?;
        self.robot.check_motion(q, v, a)?;
        self.robot.pinocchio_rnea_into(q, v, a, &mut self.workspace);
        self.has_kinematics = true;
        self.has_dynamics = true;
        self.has_joint_jacobians = false;
        Ok(())
    }

    fn compute_joint_jacobians(&mut self, q: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let q = q.as_slice()?;
        if q.len() != self.robot.dof {
            return Err(PyValueError::new_err("q length must match robot dof"));
        }
        self.robot
            .pinocchio_joint_jacobians_into(q, &mut self.workspace);
        self.has_kinematics = true;
        self.has_dynamics = false;
        self.has_joint_jacobians = true;
        Ok(())
    }

    fn rotations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f64>>> {
        if !self.has_kinematics {
            return Err(PyValueError::new_err(
                "compute_kinematics, compute_dynamics, or compute_joint_jacobians must be called first",
            ));
        }
        Ok(self
            .workspace
            .r
            .clone()
            .into_pyarray(py)
            .reshape([self.robot.link_num, 3, 3])?)
    }

    fn positions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if !self.has_kinematics {
            return Err(PyValueError::new_err(
                "compute_kinematics, compute_dynamics, or compute_joint_jacobians must be called first",
            ));
        }
        Ok(self
            .workspace
            .p
            .clone()
            .into_pyarray(py)
            .reshape([self.robot.link_num, 3])?)
    }

    fn tau<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if !self.has_dynamics {
            return Err(PyValueError::new_err(
                "compute_dynamics must be called before reading tau",
            ));
        }
        Ok(self.workspace.tau.clone().into_pyarray(py))
    }

    fn joint_jacobians<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f64>>> {
        if !self.has_joint_jacobians {
            return Err(PyValueError::new_err(
                "compute_joint_jacobians must be called before reading joint_jacobians",
            ));
        }
        Ok(self.workspace.jac.clone().into_pyarray(py).reshape([
            self.robot.link_num,
            6,
            self.robot.dof,
        ])?)
    }
}

#[pymethods]
impl RustOutwardData {
    #[getter]
    fn order(&self) -> usize {
        self.order
    }

    #[getter]
    fn dynamics_order(&self) -> usize {
        self.dynamics_order
    }

    fn compute_kinematics(&mut self, motion: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let motion = motion.as_slice()?;
        self.robot.check_cmtm_motion(motion, self.order)?;
        self.robot
            .kinematics_cmtm_into(motion, self.order, &mut self.kinematics);
        self.has_kinematics = true;
        self.has_dynamics = false;
        self.has_cached_order1_dynamics = false;
        Ok(())
    }

    fn compute_dynamics(&mut self, motion: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        if self.order < 2 {
            return Err(PyValueError::new_err("dynamics data requires order >= 2"));
        }
        let motion = motion.as_slice()?;
        self.robot.check_cmtm_motion(motion, self.order)?;
        if self.order == 3 && self.dynamics_order == 1 {
            self.robot
                .dynamics_cmtm_order1_cached_into(motion, &mut self.dynamics);
            self.has_cached_order1_dynamics = true;
        } else {
            self.robot
                .dynamics_cmtm_into(motion, self.dynamics_order, &mut self.dynamics);
            self.has_cached_order1_dynamics = false;
        }
        self.has_kinematics = true;
        self.has_dynamics = true;
        Ok(())
    }

    fn compute_dynamics_minimal(&mut self, motion: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        if self.order < 2 {
            return Err(PyValueError::new_err("dynamics data requires order >= 2"));
        }
        let motion = motion.as_slice()?;
        self.robot.check_cmtm_motion(motion, self.order)?;
        self.robot
            .dynamics_cmtm_minimal_into(motion, self.dynamics_order, &mut self.dynamics);
        self.has_kinematics = true;
        self.has_dynamics = true;
        self.has_cached_order1_dynamics = false;
        Ok(())
    }

    fn link_mat<'py>(
        &self,
        py: Python<'py>,
        link_id: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.check_link_id(link_id)?;
        let ws = self.cmtm_source()?;
        let start = link_id * 16;
        Ok(ws.link_mat[start..start + 16]
            .to_vec()
            .into_pyarray(py)
            .reshape([4, 4])?)
    }

    fn joint_mat<'py>(
        &self,
        py: Python<'py>,
        joint_id: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.check_joint_id(joint_id)?;
        if self.has_cached_order1_dynamics {
            let mat = order1_joint_mat_value(&self.robot, &self.dynamics, joint_id);
            let mut out = Vec::with_capacity(16);
            for row in mat {
                out.extend_from_slice(&row);
            }
            return Ok(out.into_pyarray(py).reshape([4, 4])?);
        }
        let ws = self.cmtm_source()?;
        let start = joint_id * 16;
        Ok(ws.joint_mat[start..start + 16]
            .to_vec()
            .into_pyarray(py)
            .reshape([4, 4])?)
    }

    fn link_vec<'py>(
        &self,
        py: Python<'py>,
        link_id: usize,
        key_order: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.check_link_id(link_id)?;
        let vec_index = self.cmtm_vec_index(key_order)?;
        let ws = self.cmtm_source()?;
        let start = (link_id * (self.order - 1) + vec_index) * 6;
        Ok(ws.link_vecs[start..start + 6].to_vec().into_pyarray(py))
    }

    fn joint_vec<'py>(
        &self,
        py: Python<'py>,
        joint_id: usize,
        key_order: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.check_joint_id(joint_id)?;
        let vec_index = self.cmtm_vec_index(key_order)?;
        if self.has_cached_order1_dynamics {
            return Ok(
                order1_joint_vec_value(&self.robot, &self.dynamics, joint_id, vec_index)
                    .to_vec()
                    .into_pyarray(py),
            );
        }
        let ws = self.cmtm_source()?;
        let start = (joint_id * (self.order - 1) + vec_index) * 6;
        Ok(ws.joint_vecs[start..start + 6].to_vec().into_pyarray(py))
    }

    fn link_momentum<'py>(
        &self,
        py: Python<'py>,
        link_id: usize,
        key_order: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.check_link_id(link_id)?;
        self.check_dynamics_computed()?;
        let vec_index = self.momentum_vec_index(key_order)?;
        if self.has_cached_order1_dynamics {
            return Ok(
                order1_link_momentum_value(&self.robot, &self.dynamics, link_id, vec_index)
                    .to_vec()
                    .into_pyarray(py),
            );
        }
        let start = (link_id * (self.dynamics_order + 1) + vec_index) * 6;
        Ok(self.dynamics.link_momentum[start..start + 6]
            .to_vec()
            .into_pyarray(py))
    }

    fn joint_momentum<'py>(
        &self,
        py: Python<'py>,
        joint_id: usize,
        key_order: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.check_joint_id(joint_id)?;
        self.check_dynamics_computed()?;
        let vec_index = self.momentum_vec_index(key_order)?;
        let start = (joint_id * (self.dynamics_order + 1) + vec_index) * 6;
        Ok(self.dynamics.joint_momentum[start..start + 6]
            .to_vec()
            .into_pyarray(py))
    }

    fn world_link_momentum<'py>(
        &self,
        py: Python<'py>,
        link_id: usize,
        key_order: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.check_link_id(link_id)?;
        self.check_dynamics_computed()?;
        self.momentum_vec_index(key_order)?;
        let ws = self.cmtm_source()?;
        let mat = mat4_from_flat(&ws.link_mat, link_id);
        let vecs = cmtm_vecs_slice(&ws.link_vecs, link_id, self.order);
        if self.has_cached_order1_dynamics {
            let mut raw = [0.0; 12];
            for index in 0..key_order {
                let value = order1_link_momentum_value(&self.robot, &self.dynamics, link_id, index);
                raw[index * 6..index * 6 + 6].copy_from_slice(&value);
            }
            let out = cmtm_world_wrench_value(mat, vecs, &raw[..key_order * 6], key_order);
            return Ok(out.into_pyarray(py));
        }
        let start = link_id * (self.dynamics_order + 1) * 6;
        let out = cmtm_world_wrench_value(
            mat,
            vecs,
            &self.dynamics.link_momentum[start..start + key_order * 6],
            key_order,
        );
        Ok(out.into_pyarray(py))
    }

    fn world_joint_momentum<'py>(
        &self,
        py: Python<'py>,
        joint_id: usize,
        key_order: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.check_joint_id(joint_id)?;
        self.check_dynamics_computed()?;
        self.momentum_vec_index(key_order)?;
        let link_id = self.robot.child_link[joint_id];
        let ws = self.cmtm_source()?;
        let mat = mat4_from_flat(&ws.link_mat, link_id);
        let vecs = cmtm_vecs_slice(&ws.link_vecs, link_id, self.order);
        let start = joint_id * (self.dynamics_order + 1) * 6;
        let out = cmtm_world_wrench_value(
            mat,
            vecs,
            &self.dynamics.joint_momentum[start..start + key_order * 6],
            key_order,
        );
        Ok(out.into_pyarray(py))
    }

    fn link_force<'py>(
        &self,
        py: Python<'py>,
        link_id: usize,
        key_order: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.check_link_id(link_id)?;
        self.check_dynamics_computed()?;
        let vec_index = self.force_vec_index(key_order)?;
        if self.has_cached_order1_dynamics {
            return Ok(
                order1_link_force_value(&self.robot, &self.dynamics, link_id)
                    .to_vec()
                    .into_pyarray(py),
            );
        }
        let start = (link_id * self.dynamics_order + vec_index) * 6;
        Ok(self.dynamics.link_force[start..start + 6]
            .to_vec()
            .into_pyarray(py))
    }

    fn world_link_force<'py>(
        &self,
        py: Python<'py>,
        link_id: usize,
        key_order: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.check_link_id(link_id)?;
        self.check_dynamics_computed()?;
        self.force_vec_index(key_order)?;
        let ws = self.cmtm_source()?;
        let mat = mat4_from_flat(&ws.link_mat, link_id);
        let vecs = cmtm_vecs_slice(&ws.link_vecs, link_id, self.order);
        if self.has_cached_order1_dynamics {
            let force = order1_link_force_value(&self.robot, &self.dynamics, link_id);
            let out = cmtm_world_wrench_value(mat, vecs, &force, key_order);
            return Ok(out.into_pyarray(py));
        }
        let start = link_id * self.dynamics_order * 6;
        let out = cmtm_world_wrench_value(
            mat,
            vecs,
            &self.dynamics.link_force[start..start + key_order * 6],
            key_order,
        );
        Ok(out.into_pyarray(py))
    }

    fn world_joint_force<'py>(
        &self,
        py: Python<'py>,
        joint_id: usize,
        key_order: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.check_joint_id(joint_id)?;
        self.check_dynamics_computed()?;
        self.force_vec_index(key_order)?;
        let link_id = self.robot.child_link[joint_id];
        let ws = self.cmtm_source()?;
        let mat = mat4_from_flat(&ws.link_mat, link_id);
        let vecs = cmtm_vecs_slice(&ws.link_vecs, link_id, self.order);
        if self.has_cached_order1_dynamics {
            let force = order1_joint_force_value(&self.robot, &self.dynamics, joint_id);
            let out = cmtm_world_wrench_value(mat, vecs, &force, key_order);
            return Ok(out.into_pyarray(py));
        }
        let start = joint_id * self.dynamics_order * 6;
        let out = cmtm_world_wrench_value(
            mat,
            vecs,
            &self.dynamics.joint_force[start..start + key_order * 6],
            key_order,
        );
        Ok(out.into_pyarray(py))
    }

    fn joint_force<'py>(
        &self,
        py: Python<'py>,
        joint_id: usize,
        key_order: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.check_joint_id(joint_id)?;
        self.check_dynamics_computed()?;
        let vec_index = self.force_vec_index(key_order)?;
        if self.has_cached_order1_dynamics {
            return Ok(
                order1_joint_force_value(&self.robot, &self.dynamics, joint_id)
                    .to_vec()
                    .into_pyarray(py),
            );
        }
        let start = (joint_id * self.dynamics_order + vec_index) * 6;
        Ok(self.dynamics.joint_force[start..start + 6]
            .to_vec()
            .into_pyarray(py))
    }

    fn joint_torque<'py>(
        &self,
        py: Python<'py>,
        joint_id: usize,
        key_order: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.check_joint_id(joint_id)?;
        self.check_dynamics_computed()?;
        let vec_index = self.force_vec_index(key_order)?;
        let start = joint_id * self.dynamics_order + vec_index;
        Ok(vec![self.dynamics.joint_torque[start]].into_pyarray(py))
    }

    fn cmtm_wrench_var_jacob_matvec<'py>(
        &self,
        py: Python<'py>,
        elem_mat: PyReadonlyArray2<'py, f64>,
        elem_vecs: PyReadonlyArray2<'py, f64>,
        arb_cm_vecs: PyReadonlyArray2<'py, f64>,
        rhs: PyReadonlyArray1<'py, f64>,
        inverse: bool,
        transpose: bool,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let elem_shape = elem_mat.shape();
        let vec_shape = elem_vecs.shape();
        let arb_shape = arb_cm_vecs.shape();
        if elem_shape != [4, 4] {
            return Err(PyValueError::new_err("elem_mat must have shape (4, 4)"));
        }
        if vec_shape.len() != 2 || vec_shape[1] != 6 {
            return Err(PyValueError::new_err(
                "elem_vecs must have shape (order - 1, 6)",
            ));
        }
        if arb_shape.len() != 2 || arb_shape[1] != 6 {
            return Err(PyValueError::new_err(
                "arb_cm_vecs must have shape (order, 6)",
            ));
        }
        let order = arb_shape[0];
        if vec_shape[0] + 1 < order {
            return Err(PyValueError::new_err(
                "elem_vecs order must be at least arb_cm_vecs order - 1",
            ));
        }
        if rhs.len() != order * 6 {
            return Err(PyValueError::new_err("rhs length must be order * 6"));
        }

        let elem_mat = mat4_from_slice(elem_mat.as_slice()?);
        let elem_vecs = elem_vecs.as_slice()?;
        let arb_cm_vecs = arb_cm_vecs.as_slice()?;
        let rhs = rhs.as_slice()?;
        let mut fact = vec![1.0; order.max(1)];
        fill_factorial_table(&mut fact);
        let mut blocks = vec![[[0.0; 6]; 6]; order];
        let mut tmp = vec![0.0; order * 6];
        let mut inv_arb = vec![0.0; order * 6];
        let mut out = vec![0.0; order * 6];
        cmtm_wrench_var_jacob_matvec_into(
            elem_mat,
            elem_vecs,
            arb_cm_vecs,
            rhs,
            order,
            inverse,
            transpose,
            &fact,
            &mut blocks,
            &mut tmp,
            &mut inv_arb,
            &mut out,
        );
        Ok(out.into_pyarray(py))
    }

    fn cmtm_wrench_var_jacob_matmul_rhs<'py>(
        &self,
        py: Python<'py>,
        elem_mat: PyReadonlyArray2<'py, f64>,
        elem_vecs: PyReadonlyArray2<'py, f64>,
        arb_cm_vecs: PyReadonlyArray2<'py, f64>,
        rhs: PyReadonlyArray2<'py, f64>,
        inverse: bool,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let elem_shape = elem_mat.shape();
        let vec_shape = elem_vecs.shape();
        let arb_shape = arb_cm_vecs.shape();
        let rhs_shape = rhs.shape();
        if elem_shape != [4, 4] {
            return Err(PyValueError::new_err("elem_mat must have shape (4, 4)"));
        }
        if vec_shape.len() != 2 || vec_shape[1] != 6 {
            return Err(PyValueError::new_err(
                "elem_vecs must have shape (order - 1, 6)",
            ));
        }
        if arb_shape.len() != 2 || arb_shape[1] != 6 {
            return Err(PyValueError::new_err(
                "arb_cm_vecs must have shape (order, 6)",
            ));
        }
        let order = arb_shape[0];
        if vec_shape[0] + 1 < order {
            return Err(PyValueError::new_err(
                "elem_vecs order must be at least arb_cm_vecs order - 1",
            ));
        }
        if rhs_shape.len() != 2 || rhs_shape[0] != order * 6 {
            return Err(PyValueError::new_err(format!(
                "rhs must have shape ({}, rhs_dim)",
                order * 6
            )));
        }

        let rhs_dim = rhs_shape[1];
        let elem_mat = mat4_from_slice(elem_mat.as_slice()?);
        let elem_vecs = elem_vecs.as_slice()?;
        let arb_cm_vecs = arb_cm_vecs.as_slice()?;
        let rhs = rhs.as_slice()?;
        let mut fact = vec![1.0; order.max(1)];
        fill_factorial_table(&mut fact);
        let mut blocks = vec![[[0.0; 6]; 6]; order];
        let mut tmp = vec![0.0; order * 6];
        let mut inv_arb = vec![0.0; order * 6];
        let mut rhs_col = vec![0.0; order * 6];
        let mut out_col = vec![0.0; order * 6];
        let mut out = vec![0.0; order * 6 * rhs_dim];

        cmtm_wrench_var_jacob_matmul_rhs_into(
            elem_mat,
            elem_vecs,
            arb_cm_vecs,
            rhs,
            order,
            rhs_dim,
            inverse,
            &fact,
            &mut blocks,
            &mut tmp,
            &mut inv_arb,
            &mut rhs_col,
            &mut out_col,
            &mut out,
        );
        Ok(out.into_pyarray(py).reshape([order * 6, rhs_dim])?)
    }
}

#[pymethods]
impl RustBatchOutwardData {
    #[getter]
    fn order(&self) -> usize {
        self.order
    }

    #[getter]
    fn dynamics_order(&self) -> usize {
        self.dynamics_order
    }

    #[getter]
    fn batch(&self) -> usize {
        self.batch
    }

    fn compute_kinematics(&mut self, motions: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        self.check_motion_shape(motions.shape())?;
        let motions = motions.as_slice()?;
        let motion_len = self.robot.dof * self.order;
        for sample in 0..self.batch {
            let start = sample * motion_len;
            let end = start + motion_len;
            self.robot.kinematics_cmtm_into(
                &motions[start..end],
                self.order,
                &mut self.kinematics[sample],
            );
        }
        self.has_kinematics = true;
        self.has_dynamics = false;
        self.has_cached_order1_dynamics = false;
        Ok(())
    }

    fn compute_dynamics(&mut self, motions: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        if self.order < 2 {
            return Err(PyValueError::new_err("dynamics data requires order >= 2"));
        }
        self.check_motion_shape(motions.shape())?;
        let motions = motions.as_slice()?;
        let motion_len = self.robot.dof * self.order;
        for sample in 0..self.batch {
            let start = sample * motion_len;
            let end = start + motion_len;
            if self.order == 3 && self.dynamics_order == 1 {
                self.robot.dynamics_cmtm_order1_cached_into(
                    &motions[start..end],
                    &mut self.dynamics[sample],
                );
            } else {
                self.robot.dynamics_cmtm_into(
                    &motions[start..end],
                    self.dynamics_order,
                    &mut self.dynamics[sample],
                );
            }
        }
        self.has_kinematics = true;
        self.has_dynamics = true;
        self.has_cached_order1_dynamics = self.order == 3 && self.dynamics_order == 1;
        Ok(())
    }

    fn compute_dynamics_minimal(&mut self, motions: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        if self.order < 2 {
            return Err(PyValueError::new_err("dynamics data requires order >= 2"));
        }
        self.check_motion_shape(motions.shape())?;
        let motions = motions.as_slice()?;
        let motion_len = self.robot.dof * self.order;
        for sample in 0..self.batch {
            let start = sample * motion_len;
            let end = start + motion_len;
            self.robot.dynamics_cmtm_minimal_into(
                &motions[start..end],
                self.dynamics_order,
                &mut self.dynamics[sample],
            );
        }
        self.has_kinematics = true;
        self.has_dynamics = true;
        self.has_cached_order1_dynamics = false;
        Ok(())
    }

    fn link_mat<'py>(
        &self,
        py: Python<'py>,
        link_id: usize,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        self.check_link_id(link_id)?;
        let mut out = vec![0.0; self.batch * 16];
        for sample in 0..self.batch {
            let ws = self.cmtm_source(sample)?;
            let src = link_id * 16;
            let dst = sample * 16;
            out[dst..dst + 16].copy_from_slice(&ws.link_mat[src..src + 16]);
        }
        Ok(out.into_pyarray(py).reshape([self.batch, 4, 4])?)
    }

    fn joint_mat<'py>(
        &self,
        py: Python<'py>,
        joint_id: usize,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        self.check_joint_id(joint_id)?;
        let mut out = vec![0.0; self.batch * 16];
        for sample in 0..self.batch {
            let dst = sample * 16;
            if self.has_cached_order1_dynamics {
                let mat = order1_joint_mat_value(&self.robot, &self.dynamics[sample], joint_id);
                for row in 0..4 {
                    out[dst + row * 4..dst + row * 4 + 4].copy_from_slice(&mat[row]);
                }
            } else {
                let ws = self.cmtm_source(sample)?;
                let src = joint_id * 16;
                out[dst..dst + 16].copy_from_slice(&ws.joint_mat[src..src + 16]);
            }
        }
        Ok(out.into_pyarray(py).reshape([self.batch, 4, 4])?)
    }

    fn link_vec<'py>(
        &self,
        py: Python<'py>,
        link_id: usize,
        key_order: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.check_link_id(link_id)?;
        let vec_index = self.cmtm_vec_index(key_order)?;
        let mut out = vec![0.0; self.batch * 6];
        for sample in 0..self.batch {
            let ws = self.cmtm_source(sample)?;
            let src = (link_id * (self.order - 1) + vec_index) * 6;
            let dst = sample * 6;
            out[dst..dst + 6].copy_from_slice(&ws.link_vecs[src..src + 6]);
        }
        Ok(out.into_pyarray(py).reshape([self.batch, 6])?)
    }

    fn joint_vec<'py>(
        &self,
        py: Python<'py>,
        joint_id: usize,
        key_order: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.check_joint_id(joint_id)?;
        let vec_index = self.cmtm_vec_index(key_order)?;
        let mut out = vec![0.0; self.batch * 6];
        for sample in 0..self.batch {
            let dst = sample * 6;
            if self.has_cached_order1_dynamics {
                out[dst..dst + 6].copy_from_slice(&order1_joint_vec_value(
                    &self.robot,
                    &self.dynamics[sample],
                    joint_id,
                    vec_index,
                ));
            } else {
                let ws = self.cmtm_source(sample)?;
                let src = (joint_id * (self.order - 1) + vec_index) * 6;
                out[dst..dst + 6].copy_from_slice(&ws.joint_vecs[src..src + 6]);
            }
        }
        Ok(out.into_pyarray(py).reshape([self.batch, 6])?)
    }

    fn link_momentum<'py>(
        &self,
        py: Python<'py>,
        link_id: usize,
        key_order: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.check_link_id(link_id)?;
        self.check_dynamics_computed()?;
        let vec_index = self.momentum_vec_index(key_order)?;
        let mut out = vec![0.0; self.batch * 6];
        for sample in 0..self.batch {
            let dst = sample * 6;
            if self.has_cached_order1_dynamics {
                out[dst..dst + 6].copy_from_slice(&order1_link_momentum_value(
                    &self.robot,
                    &self.dynamics[sample],
                    link_id,
                    vec_index,
                ));
            } else {
                let src = (link_id * (self.dynamics_order + 1) + vec_index) * 6;
                out[dst..dst + 6]
                    .copy_from_slice(&self.dynamics[sample].link_momentum[src..src + 6]);
            }
        }
        Ok(out.into_pyarray(py).reshape([self.batch, 6])?)
    }

    fn joint_momentum<'py>(
        &self,
        py: Python<'py>,
        joint_id: usize,
        key_order: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.check_joint_id(joint_id)?;
        self.check_dynamics_computed()?;
        let vec_index = self.momentum_vec_index(key_order)?;
        let mut out = vec![0.0; self.batch * 6];
        for sample in 0..self.batch {
            let src = (joint_id * (self.dynamics_order + 1) + vec_index) * 6;
            let dst = sample * 6;
            out[dst..dst + 6].copy_from_slice(&self.dynamics[sample].joint_momentum[src..src + 6]);
        }
        Ok(out.into_pyarray(py).reshape([self.batch, 6])?)
    }

    fn world_link_momentum<'py>(
        &self,
        py: Python<'py>,
        link_id: usize,
        key_order: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.check_link_id(link_id)?;
        self.check_dynamics_computed()?;
        self.momentum_vec_index(key_order)?;
        let mut out = vec![0.0; self.batch * 6];
        for sample in 0..self.batch {
            let ws = self.cmtm_source(sample)?;
            let mat = mat4_from_flat(&ws.link_mat, link_id);
            let vecs = cmtm_vecs_slice(&ws.link_vecs, link_id, self.order);
            if self.has_cached_order1_dynamics {
                let mut raw = [0.0; 12];
                for index in 0..key_order {
                    let value = order1_link_momentum_value(
                        &self.robot,
                        &self.dynamics[sample],
                        link_id,
                        index,
                    );
                    raw[index * 6..index * 6 + 6].copy_from_slice(&value);
                }
                let value = cmtm_world_wrench_value(mat, vecs, &raw[..key_order * 6], key_order);
                out[sample * 6..sample * 6 + 6].copy_from_slice(&value);
                continue;
            }
            let src = link_id * (self.dynamics_order + 1) * 6;
            let value = cmtm_world_wrench_value(
                mat,
                vecs,
                &self.dynamics[sample].link_momentum[src..src + key_order * 6],
                key_order,
            );
            out[sample * 6..sample * 6 + 6].copy_from_slice(&value);
        }
        Ok(out.into_pyarray(py).reshape([self.batch, 6])?)
    }

    fn world_joint_momentum<'py>(
        &self,
        py: Python<'py>,
        joint_id: usize,
        key_order: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.check_joint_id(joint_id)?;
        self.check_dynamics_computed()?;
        self.momentum_vec_index(key_order)?;
        let link_id = self.robot.child_link[joint_id];
        let mut out = vec![0.0; self.batch * 6];
        for sample in 0..self.batch {
            let ws = self.cmtm_source(sample)?;
            let mat = mat4_from_flat(&ws.link_mat, link_id);
            let vecs = cmtm_vecs_slice(&ws.link_vecs, link_id, self.order);
            let src = joint_id * (self.dynamics_order + 1) * 6;
            let value = cmtm_world_wrench_value(
                mat,
                vecs,
                &self.dynamics[sample].joint_momentum[src..src + key_order * 6],
                key_order,
            );
            out[sample * 6..sample * 6 + 6].copy_from_slice(&value);
        }
        Ok(out.into_pyarray(py).reshape([self.batch, 6])?)
    }

    fn link_force<'py>(
        &self,
        py: Python<'py>,
        link_id: usize,
        key_order: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.check_link_id(link_id)?;
        self.check_dynamics_computed()?;
        let vec_index = self.force_vec_index(key_order)?;
        let mut out = vec![0.0; self.batch * 6];
        for sample in 0..self.batch {
            let dst = sample * 6;
            if self.has_cached_order1_dynamics {
                out[dst..dst + 6].copy_from_slice(&order1_link_force_value(
                    &self.robot,
                    &self.dynamics[sample],
                    link_id,
                ));
            } else {
                let src = (link_id * self.dynamics_order + vec_index) * 6;
                out[dst..dst + 6].copy_from_slice(&self.dynamics[sample].link_force[src..src + 6]);
            }
        }
        Ok(out.into_pyarray(py).reshape([self.batch, 6])?)
    }

    fn world_link_force<'py>(
        &self,
        py: Python<'py>,
        link_id: usize,
        key_order: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.check_link_id(link_id)?;
        self.check_dynamics_computed()?;
        self.force_vec_index(key_order)?;
        let mut out = vec![0.0; self.batch * 6];
        for sample in 0..self.batch {
            let ws = self.cmtm_source(sample)?;
            let mat = mat4_from_flat(&ws.link_mat, link_id);
            let vecs = cmtm_vecs_slice(&ws.link_vecs, link_id, self.order);
            if self.has_cached_order1_dynamics {
                let force = order1_link_force_value(&self.robot, &self.dynamics[sample], link_id);
                let value = cmtm_world_wrench_value(mat, vecs, &force, key_order);
                out[sample * 6..sample * 6 + 6].copy_from_slice(&value);
                continue;
            }
            let src = link_id * self.dynamics_order * 6;
            let value = cmtm_world_wrench_value(
                mat,
                vecs,
                &self.dynamics[sample].link_force[src..src + key_order * 6],
                key_order,
            );
            out[sample * 6..sample * 6 + 6].copy_from_slice(&value);
        }
        Ok(out.into_pyarray(py).reshape([self.batch, 6])?)
    }

    fn world_joint_force<'py>(
        &self,
        py: Python<'py>,
        joint_id: usize,
        key_order: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.check_joint_id(joint_id)?;
        self.check_dynamics_computed()?;
        self.force_vec_index(key_order)?;
        let link_id = self.robot.child_link[joint_id];
        let mut out = vec![0.0; self.batch * 6];
        for sample in 0..self.batch {
            let ws = self.cmtm_source(sample)?;
            let mat = mat4_from_flat(&ws.link_mat, link_id);
            let vecs = cmtm_vecs_slice(&ws.link_vecs, link_id, self.order);
            if self.has_cached_order1_dynamics {
                let force = order1_joint_force_value(&self.robot, &self.dynamics[sample], joint_id);
                let value = cmtm_world_wrench_value(mat, vecs, &force, key_order);
                out[sample * 6..sample * 6 + 6].copy_from_slice(&value);
                continue;
            }
            let src = joint_id * self.dynamics_order * 6;
            let value = cmtm_world_wrench_value(
                mat,
                vecs,
                &self.dynamics[sample].joint_force[src..src + key_order * 6],
                key_order,
            );
            out[sample * 6..sample * 6 + 6].copy_from_slice(&value);
        }
        Ok(out.into_pyarray(py).reshape([self.batch, 6])?)
    }

    fn joint_force<'py>(
        &self,
        py: Python<'py>,
        joint_id: usize,
        key_order: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.check_joint_id(joint_id)?;
        self.check_dynamics_computed()?;
        let vec_index = self.force_vec_index(key_order)?;
        let mut out = vec![0.0; self.batch * 6];
        for sample in 0..self.batch {
            let dst = sample * 6;
            if self.has_cached_order1_dynamics {
                out[dst..dst + 6].copy_from_slice(&order1_joint_force_value(
                    &self.robot,
                    &self.dynamics[sample],
                    joint_id,
                ));
            } else {
                let src = (joint_id * self.dynamics_order + vec_index) * 6;
                out[dst..dst + 6].copy_from_slice(&self.dynamics[sample].joint_force[src..src + 6]);
            }
        }
        Ok(out.into_pyarray(py).reshape([self.batch, 6])?)
    }

    fn joint_torque<'py>(
        &self,
        py: Python<'py>,
        joint_id: usize,
        key_order: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.check_joint_id(joint_id)?;
        self.check_dynamics_computed()?;
        let vec_index = self.force_vec_index(key_order)?;
        let mut out = vec![0.0; self.batch];
        for sample in 0..self.batch {
            let src = joint_id * self.dynamics_order + vec_index;
            out[sample] = self.dynamics[sample].joint_torque[src];
        }
        Ok(out.into_pyarray(py).reshape([self.batch, 1])?)
    }

    fn cmtm_wrench_var_jacob_matvec<'py>(
        &self,
        py: Python<'py>,
        elem_mat: PyReadonlyArray3<'py, f64>,
        elem_vecs: PyReadonlyArray3<'py, f64>,
        arb_cm_vecs: PyReadonlyArray3<'py, f64>,
        rhs: PyReadonlyArray2<'py, f64>,
        inverse: bool,
        transpose: bool,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let elem_shape = elem_mat.shape();
        let vec_shape = elem_vecs.shape();
        let arb_shape = arb_cm_vecs.shape();
        let rhs_shape = rhs.shape();
        if elem_shape != [self.batch, 4, 4] {
            return Err(PyValueError::new_err(format!(
                "elem_mat must have shape ({}, 4, 4)",
                self.batch
            )));
        }
        if vec_shape.len() != 3 || vec_shape[0] != self.batch || vec_shape[2] != 6 {
            return Err(PyValueError::new_err(
                "elem_vecs must have shape (batch, order - 1, 6)",
            ));
        }
        if arb_shape.len() != 3 || arb_shape[0] != self.batch || arb_shape[2] != 6 {
            return Err(PyValueError::new_err(
                "arb_cm_vecs must have shape (batch, order, 6)",
            ));
        }
        let order = arb_shape[1];
        if vec_shape[1] + 1 < order {
            return Err(PyValueError::new_err(
                "elem_vecs order must be at least arb_cm_vecs order - 1",
            ));
        }
        if rhs_shape != [self.batch, order * 6] {
            return Err(PyValueError::new_err(format!(
                "rhs must have shape ({}, {})",
                self.batch,
                order * 6
            )));
        }

        let elem_mat = elem_mat.as_slice()?;
        let elem_vecs = elem_vecs.as_slice()?;
        let arb_cm_vecs = arb_cm_vecs.as_slice()?;
        let rhs = rhs.as_slice()?;
        let elem_mat_len = 16;
        let elem_vec_len = vec_shape[1] * 6;
        let arb_len = order * 6;
        let mut fact = vec![1.0; order.max(1)];
        fill_factorial_table(&mut fact);
        let mut blocks = vec![[[0.0; 6]; 6]; order];
        let mut tmp = vec![0.0; order * 6];
        let mut inv_arb = vec![0.0; order * 6];
        let mut out = vec![0.0; self.batch * order * 6];

        for sample in 0..self.batch {
            let elem_start = sample * elem_mat_len;
            let elem_vec_start = sample * elem_vec_len;
            let arb_start = sample * arb_len;
            let rhs_start = sample * arb_len;
            cmtm_wrench_var_jacob_matvec_into(
                mat4_from_slice(&elem_mat[elem_start..elem_start + elem_mat_len]),
                &elem_vecs[elem_vec_start..elem_vec_start + elem_vec_len],
                &arb_cm_vecs[arb_start..arb_start + arb_len],
                &rhs[rhs_start..rhs_start + arb_len],
                order,
                inverse,
                transpose,
                &fact,
                &mut blocks,
                &mut tmp,
                &mut inv_arb,
                &mut out[rhs_start..rhs_start + arb_len],
            );
        }
        Ok(out.into_pyarray(py).reshape([self.batch, order * 6])?)
    }

    fn cmtm_wrench_var_jacob_matmul_rhs<'py>(
        &self,
        py: Python<'py>,
        elem_mat: PyReadonlyArray3<'py, f64>,
        elem_vecs: PyReadonlyArray3<'py, f64>,
        arb_cm_vecs: PyReadonlyArray3<'py, f64>,
        rhs: PyReadonlyArray3<'py, f64>,
        inverse: bool,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let elem_shape = elem_mat.shape();
        let vec_shape = elem_vecs.shape();
        let arb_shape = arb_cm_vecs.shape();
        let rhs_shape = rhs.shape();
        if elem_shape != [self.batch, 4, 4] {
            return Err(PyValueError::new_err(format!(
                "elem_mat must have shape ({}, 4, 4)",
                self.batch
            )));
        }
        if vec_shape.len() != 3 || vec_shape[0] != self.batch || vec_shape[2] != 6 {
            return Err(PyValueError::new_err(
                "elem_vecs must have shape (batch, order - 1, 6)",
            ));
        }
        if arb_shape.len() != 3 || arb_shape[0] != self.batch || arb_shape[2] != 6 {
            return Err(PyValueError::new_err(
                "arb_cm_vecs must have shape (batch, order, 6)",
            ));
        }
        let order = arb_shape[1];
        if vec_shape[1] + 1 < order {
            return Err(PyValueError::new_err(
                "elem_vecs order must be at least arb_cm_vecs order - 1",
            ));
        }
        if rhs_shape.len() != 3 || rhs_shape[0] != self.batch || rhs_shape[1] != order * 6 {
            return Err(PyValueError::new_err(format!(
                "rhs must have shape ({}, {}, rhs_dim)",
                self.batch,
                order * 6
            )));
        }

        let rhs_dim = rhs_shape[2];
        let elem_mat = elem_mat.as_slice()?;
        let elem_vecs = elem_vecs.as_slice()?;
        let arb_cm_vecs = arb_cm_vecs.as_slice()?;
        let rhs = rhs.as_slice()?;
        let elem_mat_len = 16;
        let elem_vec_len = vec_shape[1] * 6;
        let arb_len = order * 6;
        let rhs_len = order * 6 * rhs_dim;
        let mut fact = vec![1.0; order.max(1)];
        fill_factorial_table(&mut fact);
        let mut blocks = vec![[[0.0; 6]; 6]; order];
        let mut tmp = vec![0.0; order * 6];
        let mut inv_arb = vec![0.0; order * 6];
        let mut rhs_col = vec![0.0; order * 6];
        let mut out_col = vec![0.0; order * 6];
        let mut out = vec![0.0; self.batch * rhs_len];

        for sample in 0..self.batch {
            let elem_start = sample * elem_mat_len;
            let elem_vec_start = sample * elem_vec_len;
            let arb_start = sample * arb_len;
            let rhs_start = sample * rhs_len;
            cmtm_wrench_var_jacob_matmul_rhs_into(
                mat4_from_slice(&elem_mat[elem_start..elem_start + elem_mat_len]),
                &elem_vecs[elem_vec_start..elem_vec_start + elem_vec_len],
                &arb_cm_vecs[arb_start..arb_start + arb_len],
                &rhs[rhs_start..rhs_start + rhs_len],
                order,
                rhs_dim,
                inverse,
                &fact,
                &mut blocks,
                &mut tmp,
                &mut inv_arb,
                &mut rhs_col,
                &mut out_col,
                &mut out[rhs_start..rhs_start + rhs_len],
            );
        }
        Ok(out
            .into_pyarray(py)
            .reshape([self.batch, order * 6, rhs_dim])?)
    }
}

fn mat4_from_slice(slice: &[f64]) -> [[f64; 4]; 4] {
    [
        [slice[0], slice[1], slice[2], slice[3]],
        [slice[4], slice[5], slice[6], slice[7]],
        [slice[8], slice[9], slice[10], slice[11]],
        [slice[12], slice[13], slice[14], slice[15]],
    ]
}
