from mathrobo import CMTM, SE3, SE3wrench

from ..basic.robot import RobotStruct
from ..basic.state import StateType

from ..basic.state_dict import *

def outward_state(robot : RobotStruct, state_dict : dict, state_type : StateType):
    if state_type.owner_type == "link":
        link_name = state_type.owner_name
    elif state_type.owner_type == "joint":
        joint = robot.joint_list(state_type.owner_name)
        link_name = robot.links[joint.parent_link_id].name

    data_alias = state_type.owner_name+"_"+state_type.owner_type+"_"+state_type.data_type

    if state_type.frame_name == "world":
        if state_type.is_dynamics:
            cmtm_wrench = state_dict_to_cmtm_wrench(state_dict, link_name, state_type.order)
        else:
            cmtm = state_dict_to_cmtm(state_dict, link_name, state_type.order)

    if state_type.data_type == "frame":
        return state_dict_to_frame(state_dict, state_type.owner_name)
    elif state_type.data_type == "cmtm":
        return state_dict_to_cmtm(state_dict, state_type.owner_name)
    elif "momentum" in state_type.data_type:
        if state_type.frame_name == 'world':
            local_momentum = state_dict_to_cmvec(state_dict, state_type.owner_name, \
                                                 state_type.owner_type+"_momentum", \
                                                 state_type.order).cm_vec()
            world_momentum = CMVector((cmtm_wrench.mat_adj() @ local_momentum).reshape(-1,6)).vecs()
            return world_momentum[-1]
        else:
            return np.array(state_dict[data_alias])
    elif "force" in state_type.data_type:
        if state_type.frame_name == 'world':
            local_force = state_dict_to_cmvec(state_dict, state_type.owner_name, \
                                                state_type.owner_type+"_force", \
                                                state_type.order).cm_vec()
            world_force = CMVector((cmtm_wrench.mat_adj() @ local_force).reshape(-1,6)).vecs()
            return world_force[-1]
        else:
            return np.array(state_dict[data_alias])
    elif "torque" in state_type.data_type:
        return np.array(state_dict[data_alias])
    else:
        return np.array(state_dict[data_alias])
