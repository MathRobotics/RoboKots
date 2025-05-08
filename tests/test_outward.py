import numpy as np

from mathrobo import SE3, CMTM
from robokots.outward import *

class MockJoint: 
    def __init__(self, name, parent_link_id=0):
        self.name = name
        self.parent_link_id = parent_link_id
        self.origin = SE3.eye()
        self.select_mat = np.eye(6)

class MockLink:
    def __init__(self, name):
        self.name = name
        self.id = 0
        self.origin = SE3.eye()
        self.select_mat = np.eye(6)
        self.parent_link_id = 0

# Mock robot object with link and joint names
class MockRobot:
    link_names = ["link1", "link2"]
    joint_names = ["joint1", "joint2"]
    joints = [MockJoint("joint1"), MockJoint("joint2")]
    links = [MockLink("link1"), MockLink("link2")]

    def __init__(self):
        self.links[0].parent_link_id = 0
        self.links[1].parent_link_id = 1
        self.joints[0].parent_link_id = 0
        self.joints[1].parent_link_id = 1
        self.joints[0].origin = SE3.eye()
        self.joints[1].origin = SE3.eye()
        self.joints[0].select_mat = np.eye(6)
        self.joints[1].select_mat = np.eye(6)
        self.links[0].origin = SE3.eye()
        self.links[1].origin = SE3.eye()
        self.links[0].select_mat = np.eye(6)
        self.links[1].select_mat = np.eye(6)
        self.links[0].parent_link_id = 0
        self.links[1].parent_link_id = 1
        self.links[0].id = 0
        self.links[1].id = 1
        self.dof = 2