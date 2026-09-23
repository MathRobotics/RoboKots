use robokots_rust::model::{JointKind, JointModel, LinkModel, RobotModel};
use robokots_rust::types::RustCompiledRobot;
use robokots_rust::{ReferenceFrame, StateOutput, StateOwner, StateQuantity};

fn model() -> RustCompiledRobot {
    let links = ["world", "arm", "tip"]
        .into_iter()
        .enumerate()
        .map(|(i, name)| LinkModel {
            name: name.into(),
            mass: if i == 1 { 2.0 } else { 0.0 },
            cog: if i == 1 { [0.3, 0.0, 0.0] } else { [0.0; 3] },
            inertia: if i == 1 {
                [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]
            } else {
                [0.0; 6]
            },
        })
        .collect();
    let joints = vec![
        JointModel {
            name: "axis".into(),
            parent_link: 0,
            child_link: 1,
            kind: JointKind::Revolute,
            axis: [0.0, 0.0, 1.0],
            position: [0.0; 3],
            orientation: [1.0, 0.0, 0.0, 0.0],
        },
        JointModel {
            name: "fixed".into(),
            parent_link: 1,
            child_link: 2,
            kind: JointKind::Fixed,
            axis: [0.0; 3],
            position: [1.0, 0.0, 0.0],
            orientation: [1.0, 0.0, 0.0, 0.0],
        },
    ];
    RustCompiledRobot::from_model(&RobotModel { links, joints }, false).unwrap()
}

fn values(robot: &RustCompiledRobot, motion: &[f64], gravity: [f64; 3]) -> Vec<f64> {
    let mut state = robot.create_outward_data(4).unwrap();
    state.compute_dynamics(motion, gravity).unwrap();
    let mat = state.link_mat(2).unwrap();
    let mut out = vec![mat[3], mat[7], mat[11]];
    out.extend(state.world_link_momentum(1, 1).unwrap());
    out.extend(state.joint_torque(0, 1).unwrap());
    out
}

#[test]
fn external_crate_model_state_and_matrix_free_products() {
    let robot = model();
    let info = robot.model_info();
    assert_eq!(info.dof, 1);
    assert_eq!(info.link_names, ["world", "arm", "tip"]);
    assert_eq!(info.joints[1].dof, 0);
    assert_eq!(info.joints[1].dof_index, 1);
    assert_eq!(info.joints[1].child_link, 2);
    let x = [0.4, -0.2, 0.3, 0.15];
    let g = [0.2, -0.3, -9.81];
    let outputs = [
        StateOutput::new(
            StateOwner::Link(2),
            StateQuantity::Position,
            0,
            ReferenceFrame::World,
        ),
        StateOutput::new(
            StateOwner::Link(1),
            StateQuantity::Momentum,
            0,
            ReferenceFrame::World,
        ),
        StateOutput::new(
            StateOwner::Joint(0),
            StateQuantity::Torque,
            0,
            ReferenceFrame::Local,
        ),
    ];
    let mut selected = robot.create_selected_workspace(4).unwrap();
    let mut identity = vec![0.0; 16];
    for i in 0..4 {
        identity[i * 4 + i] = 1.0;
    }
    let jac = selected
        .apply(&x, &identity, &outputs, 1, 4, g, false)
        .unwrap();
    let eps = 1e-6;
    for col in 0..4 {
        let mut xp = x;
        let mut xm = x;
        xp[col] += eps;
        xm[col] -= eps;
        let vp = values(&robot, &xp, g);
        let vm = values(&robot, &xm, g);
        for row in 0..10 {
            assert!((jac[row * 4 + col] - (vp[row] - vm[row]) / (2.0 * eps)).abs() < 2e-8);
        }
    }
    let direction = [0.1, -0.3, 0.2, 0.7];
    let weight: Vec<f64> = (0..10).map(|i| 0.03 * (i as f64) - 0.1).collect();
    let jvp = selected
        .apply(&x, &direction, &outputs, 1, 1, g, false)
        .unwrap();
    let vjp = selected
        .apply(&x, &weight, &outputs, 1, 1, g, true)
        .unwrap();
    for row in 0..10 {
        let expected: f64 = (0..4).map(|col| jac[row * 4 + col] * direction[col]).sum();
        assert!((expected - jvp[row]).abs() < 1e-12);
    }
    for col in 0..4 {
        let expected: f64 = (0..10).map(|row| jac[row * 4 + col] * weight[row]).sum();
        assert!((expected - vjp[col]).abs() < 1e-12);
    }
    assert_eq!(selected.cache_info(), (0, 1, 1));
    assert!(selected.apply(&x, &[], &outputs, 1, 1, g, false).is_err());
    assert!(selected
        .apply(&x, &direction, &outputs, 1, 1, [f64::NAN; 3], false)
        .is_err());
    assert_eq!(
        selected
            .apply(&x, &direction, &outputs, 1, 1, g, false)
            .unwrap(),
        jvp
    );

    let mut minimal = robot.create_outward_data(4).unwrap();
    minimal.compute_dynamics_minimal(&x, g).unwrap();
    assert!(minimal.joint_torque(0, 1).is_ok());
    assert!(minimal.link_momentum(1, 1).is_err());
    assert!(minimal.world_joint_force(0, 1).is_err());
    let mut aba = robot.create_aba_data();
    aba.prepare_into(&x[..1], &x[1..2], g).unwrap();
    aba.solve_into(&values(&robot, &x, g)[9..]).unwrap();
    assert!((aba.acceleration()[0] - x[2]).abs() < 1e-12);
}

#[test]
fn native_batch_getters_and_cache_misses() {
    let robot = model();
    let mut state = robot.create_batch_outward_data(4, 2).unwrap();
    let x = [0.4, -0.2, 0.3, 0.15, -0.5, 0.1, 0.2, -0.1];
    assert!(state.link_mat(0, 2).is_err());
    state.compute_dynamics(&x, [0.2, -0.3, -9.81]).unwrap();
    for sample in 0..2 {
        let mut single = robot.create_outward_data(4).unwrap();
        single
            .compute_dynamics(&x[sample * 4..sample * 4 + 4], [0.2, -0.3, -9.81])
            .unwrap();
        assert_eq!(
            single.joint_mat(0).unwrap(),
            state.joint_mat(sample, 0).unwrap()
        );
        assert_eq!(
            single.world_link_force(1, 1).unwrap(),
            state.world_link_force(sample, 1, 1).unwrap()
        );
    }
    assert!(state.link_mat(2, 0).is_err());
    assert!(state.compute_kinematics(&x[..3]).is_err());
    state.compute_kinematics(&x).unwrap();
    assert!(state.joint_torque(0, 0, 1).is_err());
    let mut selected = robot.create_selected_workspace(4).unwrap();
    let outputs = [StateOutput::new(
        StateOwner::Link(2),
        StateQuantity::Position,
        0,
        ReferenceFrame::World,
    )];
    let direction = [1.0; 8];
    let result = selected
        .apply(&x, &direction, &outputs, 2, 1, [0.0; 3], false)
        .unwrap();
    assert_eq!(selected.cache_info(), (2, 0, 2));
    let mut changed = x;
    changed[0] += 0.1;
    let new_result = selected
        .apply(&changed, &direction, &outputs, 2, 1, [0.0; 3], false)
        .unwrap();
    assert_eq!(selected.cache_info(), (3, 0, 2));
    assert_ne!(result[..3], new_result[..3]);
    assert_eq!(result[3..], new_result[3..]);
}

#[test]
fn prismatic_capability_is_explicit() {
    let input = RobotModel {
        links: ["root", "child"]
            .into_iter()
            .map(|name| LinkModel {
                name: name.into(),
                mass: 1.0,
                cog: [0.0; 3],
                inertia: [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            })
            .collect(),
        joints: vec![JointModel {
            name: "slide".into(),
            kind: JointKind::Prismatic,
            parent_link: 0,
            child_link: 1,
            axis: [0.0, 0.0, 1.0],
            position: [0.0; 3],
            orientation: [1.0, 0.0, 0.0, 0.0],
        }],
    };
    let robot = RustCompiledRobot::from_model(&input, true).unwrap();
    assert!(!robot.model_info().supports_cmtm);
    assert!(robot.create_outward_data(3).is_err());
    assert!(robot.create_selected_workspace(3).is_err());
    let mut aba = robot.create_aba_data();
    aba.prepare_into(&[0.0], &[0.0], [0.0, 0.0, -9.81]).unwrap();
    aba.solve_into(&[9.81]).unwrap();
    assert!(aba.acceleration()[0].abs() < 1e-12);
}

#[test]
fn native_json_input_preserves_ids_and_runs_without_python() {
    let input = serde_json::json!({
        "schema_version": "0.0.2",
        "links": [
            {"id": 1, "name": "arm", "mass": 2.0},
            {"id": 0, "name": "world"}
        ],
        "joints": [{"id": 0, "name": "axis", "type": "revolute",
                    "parent_link_id": 0, "child_link_id": 1, "axis": [0, 0, 1]}]
    });
    let robot = RustCompiledRobot::from_json(&input.to_string(), true).unwrap();
    assert_eq!(robot.model_info().link_names, ["world", "arm"]);
    let mut state = robot.create_outward_data(3).unwrap();
    state.compute_dynamics(&[0.2, 0.0, 0.7], [0.0; 3]).unwrap();
    assert!((state.joint_torque(0, 1).unwrap()[0] - 0.7).abs() < 1e-12);
    for (field, value) in [
        ("id", serde_json::json!(true)),
        ("name", serde_json::json!("world")),
        ("mass", serde_json::json!(-1.0)),
    ] {
        let mut bad = input.clone();
        bad["links"][0][field] = value;
        assert!(RustCompiledRobot::from_json(&bad.to_string(), true).is_err());
    }
    assert!(RustCompiledRobot::from_json("{", true).is_err());
    assert!(RustCompiledRobot::from_json("{}", true).is_err());
}

#[test]
fn native_typed_model_rejects_invalid_numeric_values() {
    let input = r#"{"schema_version":"0.0.2","links":[{"id":0,"name":"root"}],"joints":[]}"#;
    let mut model = RobotModel::from_json(input).unwrap();
    model.links[0].cog[1] = f64::NAN;
    assert!(RustCompiledRobot::from_model(&model, true).is_err());
}

#[test]
fn public_world_motion_and_typed_output_api() {
    let robot = model();
    let x = [0.4, -0.2, 0.3, 0.15, -0.11, 0.2];
    let mut state = robot.create_outward_data(6).unwrap();
    assert!(state.world_link_vec(2, 2).is_err());
    state.compute_kinematics(&x).unwrap();
    for order in 2..=6 {
        let expected = vec![0., 0., x[order - 1], 0., 0., 0.];
        for value in [
            state.world_link_vec(2, order).unwrap(),
            state.world_joint_vec(0, order).unwrap(),
        ] {
            for (a, b) in value.iter().zip(&expected) {
                assert!((a - b).abs() < 1e-12);
            }
        }
        assert_eq!(state.world_joint_vec(1, order).unwrap(), vec![0.; 6]);
    }
    assert!(state.world_link_vec(99, 2).is_err());
    assert!(state.world_joint_vec(0, 7).is_err());
    let mut batch = robot.create_batch_outward_data(6, 2).unwrap();
    batch
        .compute_dynamics(&x.repeat(2), [0.2, -0.3, -9.81])
        .unwrap();
    assert!(batch.world_link_vec(2, 2, 2).is_err());
    for sample in 0..2 {
        for order in 2..=6 {
            assert_eq!(
                batch.world_joint_vec(sample, 0, order).unwrap(),
                state.world_joint_vec(0, order).unwrap()
            );
        }
    }
    let output = StateOutput::new(
        StateOwner::Joint(0),
        StateQuantity::SpatialMotion,
        2,
        ReferenceFrame::World,
    );
    assert_eq!(output.width(), 6);
    assert_eq!(
        StateOutput::new(
            StateOwner::Link(0),
            StateQuantity::Rotation,
            0,
            ReferenceFrame::Local
        )
        .width(),
        3
    );
    let mut selected = robot.create_selected_workspace(6).unwrap();
    let mut direction = [0.; 6];
    direction[3] = 1.;
    assert_eq!(
        selected
            .apply(&x, &direction, &[output], 1, 1, [0.; 3], false)
            .unwrap(),
        vec![0., 0., 1., 0., 0., 0.]
    );
    let invalid = StateOutput::new(
        StateOwner::Link(0),
        StateQuantity::Torque,
        0,
        ReferenceFrame::World,
    );
    assert!(selected
        .apply(&x, &direction, &[invalid], 1, 1, [0.; 3], false)
        .is_err());
}

#[test]
fn public_rnea_aba_scalar_batch_and_validation() {
    let robot = model();
    let (q, v, a, gravity) = ([0.4], [-0.2], [0.3], [0.2, -0.3, -9.81]);
    let tau = robot.inverse_dynamics(&q, &v, &a, gravity).unwrap();
    let expected = (3.0 + 2.0 * 0.3 * 0.3) * a[0]
        + 2.0 * 0.3 * (q[0].sin() * gravity[0] - q[0].cos() * gravity[1]);
    assert!((tau[0] - expected).abs() < 1e-12);
    assert!((robot.forward_dynamics(&q, &v, &tau, gravity).unwrap()[0] - a[0]).abs() < 1e-12);
    let torques = robot
        .inverse_dynamics_batch(&q.repeat(2), &v.repeat(2), &a.repeat(2), 2, gravity)
        .unwrap();
    assert_eq!(torques, tau.repeat(2));
    for value in robot
        .forward_dynamics_batch(&q.repeat(2), &v.repeat(2), &torques, 2, gravity)
        .unwrap()
    {
        assert!((value - a[0]).abs() < 1e-12);
    }
    assert!(robot.inverse_dynamics(&[], &v, &a, gravity).is_err());
    assert!(robot
        .forward_dynamics(&[f64::NAN], &v, &tau, gravity)
        .is_err());
    assert!(robot
        .inverse_dynamics_batch(&q, &v, &a, usize::MAX, gravity)
        .is_err());
    assert!(robot
        .inverse_dynamics(&q, &v, &a, [f64::INFINITY; 3])
        .is_err());
    assert!(robot
        .forward_dynamics_batch(&q, &v, &[f64::NAN], 1, gravity)
        .is_err());
}
