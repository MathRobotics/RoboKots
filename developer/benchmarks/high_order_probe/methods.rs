    /// Benchmark-only high-order state evaluation. No production dispatch changes.
    #[pyo3(signature = (motions, dynamics_order, gravity, variant, loops))]
    fn benchmark_high_order(
        &self, motions: PyReadonlyArray2<'_, f64>, dynamics_order: usize,
        gravity: PyReadonlyArray1<'_, f64>, variant: &str, loops: usize,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        use std::time::Instant;
        let order = dynamics_order + 2;
        if order < 4 || loops == 0 || !["current","spatial","transport","both"].contains(&variant) {
            return Err(PyValueError::new_err("invalid high-order probe configuration"));
        }
        let batch = motions.shape()[0];
        let input = self.dof * order;
        if batch == 0 || motions.shape()[1] != input { return Err(PyValueError::new_err("invalid motion shape")); }
        let motion = motions.as_slice()?;
        let gravity = gravity_vec3(Some(gravity))?;
        let mut ws = DynamicsCmtmWorkspace::new(self, dynamics_order);
        let t = Instant::now();
        for _ in 0..loops {
            for i in 0..batch {
                let x = std::hint::black_box(&motion[i*input..(i+1)*input]);
                match variant {
                    "spatial" => self.dynamics_probe_spatial_into(x, dynamics_order, gravity, &mut ws),
                    "transport" => self.dynamics_probe_transport_into(x, dynamics_order, gravity, &mut ws),
                    "both" => self.dynamics_probe_both_into(x, dynamics_order, gravity, &mut ws),
                    _ => self.dynamics_cmtm_into(x, dynamics_order, gravity, &mut ws),
                }
                std::hint::black_box(&ws);
            }
        }
        let total = t.elapsed().as_secs_f64() / loops as f64;
        // Every semantic output is validated, not just projected joint torque.
        let mut values = Vec::new();
        for v in [&ws.cmtm.link_mat, &ws.cmtm.link_vecs, &ws.cmtm.joint_mat, &ws.cmtm.joint_vecs,
                  &ws.link_momentum, &ws.link_force, &ws.joint_momentum, &ws.joint_force,
                  &ws.joint_torque, &ws.joint_gravity_force, &ws.link_local_gravity] {
            values.extend_from_slice(v);
        }
        let t = Instant::now();
        for _ in 0..loops {
            for i in 0..batch {
                let x = std::hint::black_box(&motion[i*input..(i+1)*input]);
                if variant == "spatial" || variant == "both" {
                    self.probe_spatial_kinematics_into(x, order, &mut ws.cmtm);
                } else { self.kinematics_cmtm_into(x, order, &mut ws.cmtm); }
                std::hint::black_box(&ws.cmtm);
            }
        }
        Ok((vec![total, t.elapsed().as_secs_f64()/loops as f64], values))
    }
