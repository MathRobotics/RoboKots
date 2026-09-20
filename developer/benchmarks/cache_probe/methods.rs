    /// Benchmark-only: not installed in the production extension.
    #[pyo3(signature = (motions, directions, weights, outputs, dynamics_order, gravity, policy, reuse))]
    fn benchmark_state_cache(
        &self, motions: PyReadonlyArray2<'_, f64>, directions: PyReadonlyArray3<'_, f64>,
        weights: PyReadonlyArray3<'_, f64>, outputs: Vec<DynamicsOutput>, dynamics_order: usize,
        gravity: PyReadonlyArray1<'_, f64>, policy: &str, reuse: usize,
    ) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>)> {
        use std::time::Instant;
        let rows = self.check_dynamics_outputs(&outputs, dynamics_order)?;
        let batch = motions.shape()[0];
        let input = self.dof * (dynamics_order + 2);
        let cols = directions.shape()[2];
        if motions.shape() != [batch, input] || directions.shape() != [batch, input, cols]
            || weights.shape() != [batch, rows, cols] || reuse == 0 {
            return Err(PyValueError::new_err("invalid probe shapes"));
        }
        if !["current", "eager", "lazy_fill", "lazy_recompute"].contains(&policy) {
            return Err(PyValueError::new_err("invalid policy"));
        }
        let motion = motions.as_slice()?; let directions = directions.as_slice()?;
        let weights = weights.as_slice()?; let gravity = gravity_vec3(Some(gravity))?;
        let lean = dynamics_order == 1 && gravity == [0.0; 3];
        let mut states: Vec<_> = (0..batch).map(|_| DynamicsCmtmWorkspace::new(self, dynamics_order)).collect();
        let mut times = vec![0.0; 6]; // dynamics, promote, first JVP/VJP, further JVP/VJP
        let mut jvp = vec![0.0; batch * rows * cols];
        let mut vjp = vec![0.0; batch * input * cols];
        let t = Instant::now();
        for (i, state) in states.iter_mut().enumerate() {
            let x = &motion[i * input..(i + 1) * input];
            if lean && policy != "eager" { self.dynamics_cmtm_order1_cached_into(x, state); }
            else { self.dynamics_cmtm_into(x, dynamics_order, gravity, state); }
        }
        times[0] = t.elapsed().as_secs_f64();
        let t = Instant::now();
        if lean && (policy == "lazy_fill" || policy == "lazy_recompute") {
            for (i, state) in states.iter_mut().enumerate() {
                let x = &motion[i * input..(i + 1) * input];
                if policy == "lazy_fill" { self.probe_complete_order1(x, state); }
                else { self.dynamics_cmtm_into(x, dynamics_order, gravity, state); }
            }
        }
        times[1] = t.elapsed().as_secs_f64();
        for evaluation in 0..reuse {
            let t = Instant::now();
            let mut tangent = DynamicsCmtmTangentWorkspace::new(self, dynamics_order, cols);
            // Baseline really allocates a separate primal on every JVP call.
            let mut scratch = if policy == "current" { Some(DynamicsCmtmWorkspace::new(self, dynamics_order)) } else { None };
            for (i, state) in states.iter_mut().enumerate() {
                let x = &motion[i * input..(i + 1) * input];
                let rhs = &directions[i * input * cols..(i + 1) * input * cols];
                let out = &mut jvp[i * rows * cols..(i + 1) * rows * cols];
                if let Some(primal) = scratch.as_mut() {
                    self.dynamics_selected_tangent_into(x, rhs, &outputs, dynamics_order, gravity, primal, &mut tangent, out);
                } else {
                    self.dynamics_selected_tangent_prepared_into(x, rhs, &outputs, dynamics_order, gravity, state, &mut tangent, out);
                }
            }
            times[if evaluation == 0 { 2 } else { 4 }] += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let mut scratch = if policy == "current" { Some(DynamicsCmtmWorkspace::new(self, dynamics_order)) } else { None };
            for (i, state) in states.iter_mut().enumerate() {
                let x = &motion[i * input..(i + 1) * input];
                let rhs = &weights[i * rows * cols..(i + 1) * rows * cols];
                let out = &mut vjp[i * input * cols..(i + 1) * input * cols];
                if let Some(primal) = scratch.as_mut() {
                    self.dynamics_selected_reverse_into(x, rhs, &outputs, dynamics_order, gravity, cols, primal, out);
                } else {
                    self.dynamics_selected_reverse_prepared_into(x, rhs, &outputs, dynamics_order, gravity, cols, state, out);
                }
            }
            times[if evaluation == 0 { 3 } else { 5 }] += t.elapsed().as_secs_f64();
        }
        let state = &states[0];
        let memory = vec![
            state.probe_bytes() * batch,
            state.cmtm.probe_bytes() * batch, // separately allocated kinematics in RustOutwardData
            state.probe_semantic_bytes() * batch,
            DynamicsCmtmTangentWorkspace::new(self, dynamics_order, cols).probe_bytes(),
        ];
        Ok((times, jvp, vjp, memory))
    }
