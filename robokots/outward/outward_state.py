from mathrobo import CMTM, SE3, SE3wrench

from ..basic.robot import RobotStruct
from ..basic.state import StateType, data_type_dof

from ..basic.state_dict import *

def outward_state_dof(robot : RobotStruct, state_type : StateType, dim : int = 3) -> int:
    if "torque" in state_type.data_type:
        joint = robot.joint_list([state_type.owner_name])[0]
        return joint.dof
    else:
        return data_type_dof(state_type.data_type, dim = dim)

def outward_state(robot : RobotStruct, state_dict : dict, state_type : StateType):
    if state_type.owner_type == "link":
        link_name = state_type.owner_name
    elif state_type.owner_type == "joint":
        joint = robot.joint_list([state_type.owner_name])[0]
        link_name = robot.links[joint.child_link_id].name

    if state_type.frame_name == "world":
        if state_type.is_dynamics:
            cmtm_wrench = state_dict_to_cmtm_wrench(state_dict, link_name, "link", state_type.key_order)
        else:
            cmtm = state_dict_to_cmtm(state_dict, link_name, "link", state_type.key_order)

    if state_type.data_type == "frame":
        return state_dict_to_frame(state_dict, state_type.owner_name)
    elif state_type.data_type == "cmtm":
        return state_dict_to_cmtm(state_dict, state_type.owner_name, state_type.owner_type)
    elif "momentum" in state_type.data_type:
        if state_type.frame_name == 'world':
            local_momentum = state_dict_to_cmvec(state_dict, state_type.owner_name, \
                                                 state_type.owner_type,
                                                 "momentum", \
                                                 state_type.key_order).cm_vec()
            world_momentum = CMVector.set_cmvecs((cmtm_wrench.mat_adj() @ local_momentum).reshape(-1,6)).vecs()
            return world_momentum[-1]
        else:
            return np.array(state_dict[state_type.alliance])
    elif "force" in state_type.data_type:
        if state_type.frame_name == 'world':
            local_force = state_dict_to_cmvec(state_dict, state_type.owner_name, \
                                                state_type.owner_type,
                                                "force", \
                                                state_type.key_order).cm_vec()
            world_force = CMVector.set_cmvecs((cmtm_wrench.mat_adj() @ local_force).reshape(-1,6)).vecs()
            return world_force[-1]
        else:
            return np.array(state_dict[state_type.alliance])
    elif "torque" in state_type.data_type:
        return np.array(state_dict[state_type.alliance])
    else:
        return np.array(state_dict[state_type.alliance])

def outward_state_cmvec(robot : RobotStruct, state_dict : dict, state_type : StateType, order : int) -> CMVector:
    vec = state_dict_to_cmvec(state_dict, state_type.owner_name, state_type.owner_type, state_type.data_type, state_type.key_order)
    if state_type.frame_name == "world":
        if state_type.owner_type == "link":
            link_name = state_type.owner_name
        elif state_type.owner_type == "joint":
            joint = robot.joint_list([state_type.owner_name])
            link_name = robot.links[joint[0].child_link_id].name
        cmtm_wrench = state_dict_to_cmtm_wrench(state_dict, link_name, "link", order)
        vec = CMTM.change_elemclass(cmtm_wrench, SE3wrench).mat_adj() @ vec.cm_vec()
    return vec

def outward_total_state_cmvec(robot : RobotStruct, state_dict : dict, owner_type : str, data_type : str, frame_name : None, order : int) -> CMVector:
    if owner_type == "link":
        name_list = robot.link_names
    elif owner_type == "joint":
        name_list = robot.joint_names

    for i, name in enumerate(name_list):
        vec = outward_state_cmvec(robot, state_dict, StateType(owner_type, name, data_type, frame_name), order)
        if i == 0:
            total_vec = np.zeros((len(name_list), vec._len))
        total_vec[i] = vec.cm_vec()
    return total_vec.flatten()