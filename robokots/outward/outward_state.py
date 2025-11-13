from mathrobo import CMTM, SE3, SE3wrench

from ..basic.robot import RobotStruct
from ..basic.state import StateType

from ..basic.state_dict import *

def outward_state(robot : RobotStruct, state_dict : dict, state_type : StateType):
    if state_type.owner_type == "link":
        link_name = state_type.owner_name
    elif state_type.owner_type == "joint":
        joint = robot.joint_list([state_type.owner_name])
        link_name = robot.links[joint[0].child_link_id].name

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
            world_momentum = CMVector((Factorial.mat(state_type.key_order, dim=6) @ cmtm_wrench.mat_adj() @ local_momentum).reshape(-1,6)).vecs()
            return world_momentum[-1]
        else:
            return np.array(state_dict[state_type.alliance])
    elif "force" in state_type.data_type:
        if state_type.frame_name == 'world':
            local_force = state_dict_to_cmvec(state_dict, state_type.owner_name, \
                                                state_type.owner_type,
                                                "force", \
                                                state_type.key_order).cm_vec()
            world_force = CMVector((Factorial.mat(state_type.key_order, dim=6) @ cmtm_wrench.mat_adj() @ local_force).reshape(-1,6)).vecs()
            return world_force[-1]
        else:
            return np.array(state_dict[state_type.alliance])
    elif "torque" in state_type.data_type:
        return np.array(state_dict[state_type.alliance])
    else:
        return np.array(state_dict[state_type.alliance])
