"""Selected spatial outputs as small operators on local CMTM variations.

The packed variation is [body pose tangent, ordinary velocity derivatives].
Operators are independent of robot input dimension; products never form a
whole-robot Jacobian. Mixed assembly shares these operators in both directions.
"""
from math import factorial

import numpy as np
from mathrobo import CMVector

from robokots.core.state.access import state_cmtm, state_cmtm_wrench, state_cmvec, state_rel_cmtm
from robokots.core.state.spec import keys_kinematics, keys_joint_motion, keys_force, data_type_dof, StateType
from robokots.core.kernels.joint import joint_select_diag_mat
from robokots.outward.data import state_sample


def needs_spatial_selection(states):
    mixed = any(s.is_dynamics for s in states)
    return any((s.data_type in keys_kinematics or s.data_type in keys_joint_motion)
               and (s.owner_type == 'joint' or s.frame_name == 'world'
                    or (mixed and s.data_type in ('pos', 'rot', 'frame'))) for s in states)


def is_spatial(s):
    return not s.is_dynamics


def _operators(robot, state, s, order):
    owner = (s.owner_type, s.owner_name)
    c = state_cmtm(state, s.owner_name, s.owner_type, order)
    result = {}
    width = data_type_dof(s.data_type)
    if s.key_order == 1:
        op = np.zeros((width, 6*order))
        transform = np.eye(6)
        if s.frame_name == 'world':
            # Pose values retain their owner transform. World selects a spatial
            # tangent, while None/local selects the established body tangent.
            transform = np.asarray(c.mat_adj())[:6, :6]
            if s.data_type == 'pos':
                transform = np.zeros((6, 6))
                transform[3:, 3:] = np.asarray(c.elem_mat())[:3, :3]
        rows = slice(3, 6) if s.data_type == 'pos' else slice(0, 3) if s.data_type == 'rot' else slice(0, 6)
        op[:, :6] = transform[rows]
        return {owner: op}
    n = s.key_order - 1
    op = np.zeros((6, 6*order))
    if s.frame_name != 'world':
        op[:, 6*n:6*(n+1)] = np.eye(6)
        return {owner: op}
    frame_name = s.owner_name if s.owner_type == 'link' else robot.links[robot.joint(s.owner_name).child_link_id].name
    frame_owner = ('link', frame_name)
    frame = state_cmtm(state, frame_name, 'link', n)
    vectors = np.asarray(c.vecs())[:n]
    arb = CMVector(vectors)
    factors = np.repeat([factorial(i) for i in range(n)], 6)
    adj = np.asarray(frame.mat_adj()) * factors[:, None] / factors[None, :]
    op[:, 6:6*(n+1)] = adj[-6:]
    result[owner] = op
    transform_op = np.zeros_like(op)
    transform_op[:, :6*n] = (factors[:, None] * np.asarray(frame.mat_var_x_arb_vec_jacob(arb, frame='bframe')) @ np.asarray(frame.tangent_mat()))[-6:]
    result[frame_owner] = result.get(frame_owner, 0) + transform_op
    return result


def _local_routes(robot, state, owner, order):
    """Yield (joint, local variation / joint coordinates) along an owner route."""
    kind, name = owner
    if kind == 'joint':
        joint = robot.joint(name)
        if joint.dof > 1:
            raise NotImplementedError('selected joint spatial derivatives require fixed or one-DoF joints')
        if joint.dof:
            yield joint, joint_select_diag_mat(joint.select_mat, order)
        return
    link = robot.link(name)
    links, joints = [], []
    robot.route_target_link(link, links, joints)
    inv_tangent = np.asarray(state_cmtm(state, name, 'link', order).tangent_mat_inv())
    for index in joints:
        joint = robot.joints[index]
        if not joint.dof:
            continue
        child = robot.links[joint.child_link_id].name
        rel = state_rel_cmtm(state, name, child, 'link', order)
        jc = state_cmtm(state, joint.name, 'joint', order)
        yield joint, inv_tangent @ np.asarray(rel.mat_adj()) @ np.asarray(jc.tangent_mat()) @ joint_select_diag_mat(joint.select_mat, order)


def spatial_apply(robot, state, states, order, rhs, *, transpose=False):
    """Matrix RHS, scalar or native batched state; outputs preserve selection order."""
    rhs = np.asarray(rhs)
    batch = np.asarray(state_cmtm(state, robot.links[0].name, 'link', order).elem_mat()).shape[:-2]
    if batch:
        rhs = np.broadcast_to(rhs, batch + rhs.shape[-2:])
        return np.stack([spatial_apply(robot, state_sample(robot, state, i), states, order, rhs[i], transpose=transpose)
                         for i in np.ndindex(batch)]).reshape(batch + (-1, rhs.shape[-1]))
    widths = [robot.joint(s.owner_name).dof if s.owner_type == 'joint' and s.data_type in keys_joint_motion
              else data_type_dof(s.data_type) for s in states]
    out = np.zeros((robot.dof*order if transpose else sum(widths), rhs.shape[-1]))
    row = 0
    routes = {}
    for s, width in zip(states, widths):
        if s.owner_type == 'joint' and s.data_type in keys_joint_motion:
            joint = robot.joint(s.owner_name)
            cols = (joint.dof_index + np.arange(joint.dof))*order + s.key_order-1
            if transpose:
                out[cols] += rhs[row:row+width]
            else:
                out[row:row+width] = rhs[cols]
        else:
            for owner, op in _operators(robot, state, s, order).items():
                if owner not in routes:
                    routes[owner] = list(_local_routes(robot, state, owner, order))
                for joint, local in routes[owner]:
                    cols = slice(joint.dof_index*order, (joint.dof_index+joint.dof)*order)
                    if transpose:
                        out[cols] += local.T @ (op.T @ rhs[row:row+width])
                    else:
                        out[row:row+width] += op @ (local @ rhs[cols])
        row += width
    return out


def selected_apply(robot, state, states, order, rhs, *, transpose=False, dynamics_apply=None):
    batch = np.asarray(state_cmtm(state, robot.links[0].name, 'link', order).elem_mat()).shape[:-2]
    rhs = np.broadcast_to(rhs, batch + rhs.shape[-2:])
    if any(s.data_type in keys_force and s.frame_name == 'world' for s in states):
        return _world_force_apply(robot, state, states, order, rhs, transpose, dynamics_apply)
    widths = [robot.joint(s.owner_name).dof if s.owner_type == 'joint' and (s.data_type in keys_joint_motion or s.data_type.startswith('torque'))
              else data_type_dof(s.data_type) for s in states]
    offsets = np.cumsum([0] + widths)
    kin = [i for i,s in enumerate(states) if is_spatial(s)]
    dyn = [i for i,s in enumerate(states) if not is_spatial(s)]
    rows = lambda indices: np.concatenate([np.arange(offsets[i], offsets[i+1]) for i in indices])
    if transpose:
        out = spatial_apply(robot, state, [states[i] for i in kin], order, np.take(rhs, rows(kin), axis=-2), transpose=True)
        if dyn:
            out += dynamics_apply([states[i] for i in dyn], np.take(rhs, rows(dyn), axis=-2))
        return out
    parts = spatial_apply(robot, state, [states[i] for i in kin], order, rhs)
    out = np.empty(parts.shape[:-2] + (sum(widths), rhs.shape[-1]))
    out[..., rows(kin), :] = parts
    if dyn:
        out[..., rows(dyn), :] = dynamics_apply([states[i] for i in dyn], rhs)
    return out


def split_outputs(robot, states, result, *, vector=False, list_output=False):
    if vector:
        result = result[..., 0]
    if not list_output:
        return result
    widths = [robot.joint(s.owner_name).dof if s.owner_type == 'joint' and (s.data_type in keys_joint_motion or s.data_type.startswith('torque'))
              else data_type_dof(s.data_type) for s in states]
    offsets = np.cumsum([0] + widths)
    return [result[..., offsets[i]:offsets[i+1]] if vector else result[..., offsets[i]:offsets[i+1], :]
            for i in range(len(states))]


def _world_force_apply(robot, state, states, order, rhs, transpose, dynamics_apply):
    """Expand world forces into local force and moving-frame variations."""
    expanded, blocks = [], []
    spatial = ('frame','vel','acc','jerk','snap','crackle','pop','lock','drop','shot','put')
    for s in states:
        if s.data_type not in keys_force or s.frame_name != 'world':
            width = robot.joint(s.owner_name).dof if s.owner_type == 'joint' and (s.data_type in keys_joint_motion or s.data_type.startswith('torque')) else data_type_dof(s.data_type)
            expanded.append(s)
            blocks.append(np.eye(width))
            continue
        n = s.key_order
        link = s.owner_name if s.owner_type == 'link' else robot.links[robot.joint(s.owner_name).child_link_id].name
        frame = state_cmtm(state, link, 'link', n)
        wrench = state_cmtm_wrench(state, link, 'link', n)
        force = state_cmvec(state, s.owner_name, s.owner_type, 'force', n)
        factors = np.repeat([factorial(i) for i in range(n)], 6)
        adj = np.asarray(wrench.mat_adj()) * factors[:,None] / factors[None,:]
        var = factors[:,None] * np.asarray(wrench.mat_var_x_arb_vec_jacob(CMVector(np.asarray(force.vecs())), frame='bframe')) @ np.asarray(frame.tangent_mat())
        blocks.append(np.concatenate([adj[..., -6:, :], var[..., -6:, :]], axis=-1))
        expanded.extend(StateType(s.owner_type,s.owner_name,'force' if i == 0 else f'force_diff{i}') for i in range(n))
        expanded.extend(StateType('link',link,spatial[i]) for i in range(n))
    if transpose:
        row = 0
        parts = []
        for block in blocks:
            width = block.shape[-2]
            parts.append(np.swapaxes(block,-1,-2) @ rhs[..., row:row+width, :])
            row += width
        return selected_apply(robot,state,expanded,order,np.concatenate(parts,axis=-2),transpose=True,dynamics_apply=dynamics_apply)
    values = selected_apply(robot,state,expanded,order,rhs,dynamics_apply=dynamics_apply)
    row = 0
    parts = []
    for block in blocks:
        width = block.shape[-1]
        parts.append(block @ values[..., row:row+width, :])
        row += width
    return np.concatenate(parts,axis=-2)
