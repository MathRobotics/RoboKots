#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.07.21 Created by T.Ishigaki

import numpy as np

from .joint_struct import *
from .link_struct import *

class RobotStruct:
  root_joint_id : int = 0
  root_link_id : int = 0

  def __init__(self, links_, joints_):
    self.joints = joints_
    self.links = links_

    self.robot_init()

  def robot_init(self):
    self.joint_num = len(self.joints)  
    self.link_num = len(self.links)  

    self.cnct_mat = np.zeros((self.link_num, self.link_num))

    self.dof = 0
    self.joint_dof = 0
    self.link_dof = 0
    
    for j in self.joints:
      self.joint_dof += j.dof
      if j.is_root:
        self.root_joint_id = j.id
      
    for l in self.links:
      self.link_dof += l.dof
      if l.is_root:
        self.root_link_id = l.id
      
    self.dof = self.joint_dof + self.link_dof
    
    self.set_link_names()
    self.set_joint_names()
      
  def connectivity(self):
    return self.cnct_mat
  
  def set_link_names(self):
    self.link_names = []
    for l in self.links:
      self.link_names.append(l.name)
      
  def set_joint_names(self):
    self.joint_names = []
    for j in self.joints:
      self.joint_names.append(j.name)
  
  @staticmethod
  def read_model_file(robot_et):    
    joints = []
    links = []
    
    i = 0
    dof_index = 0
    joint_list_et = robot_et.findall('./joint_list/joint')
    for joint_et in joint_list_et:
      j = JointStruct()
      j.name = joint_et.attrib.get('name')
      j.joint_type = joint_et.attrib.get('joint_type')
      j.id = i
      j.dof_index = dof_index
      j.connect_link = []

      if j.id == 0:
        j.is_root = True
      else:
        j.is_root = False

      j.init()
      joints.append(j)

      i += 1
      dof_index += j.dof
      
    i = 0
    link_list_et = robot_et.findall('./link_list/link')
    for link_et in link_list_et:
      l = LinkStruct()
      l.name = link_et.attrib.get('name')
      l.link_type = link_et.attrib.get('link_type')

      l.cog = np.array(eval(link_et.attrib.get('cog')))
      l.mass = eval(link_et.attrib.get('mass'))
      l.inertia_param = np.array(eval(link_et.attrib.get('inertia_param')))
      
      l.id = i
      l.dof_index = dof_index
      
      l.connect_joint = []
      l.connect_pos = {}
      l.connect_rot = {}
      for link_joint_et in link_et:
        for j in joints:
          if j.name == link_joint_et.attrib.get('name'):
            l.connect_joint.append(j.id)
            j.connect_link.append(l.id)
            
            if link_joint_et.attrib.get('connect_pos') != None:
              l.connect_pos.update({j.id : np.array(eval(link_joint_et.attrib.get('connect_pos')))})
            else:
              l.connect_pos.update({j.id : np.zeros(3)})

            if link_joint_et.attrib.get('connect_rot') != None: 
              l.connect_rot.update({j.id : np.array(eval(link_joint_et.attrib.get('connect_rot')))})
            else:
              l.connect_rot.update({j.id : np.identity(3)})

      if l.id == 0:
        l.is_root = True
      else:
        l.is_root = False

      l.init()
      links.append(l)
      
      i += 1
      dof_index += l.dof
    return links, joints
  
  @staticmethod
  def read_model_file2(data):    
    joints = []
    links = []
    
    i = 0
    dof_index = 0

    for link in data["links"]:
      l = LinkStruct()
      l.id = link["id"]
      l.name = link["name"]
      l.cog = np.array(link["cog"])
      l.mass = link["mass"]
      l.inertia_param = link["inertia"]

  
    for joint in data["joints"]:
      j = JointStruct()
      j.id = joint["id"]
      j.name = joint["name"]
      j.joint_type = joint["type"]
      j.axis = joint["axis"]
      j.parent_link = joint["parent_link_id"]
      j.child_link = joint["child_link_id"]

    return links, joints
  
  @staticmethod
  def search_neiborhood(robot, link_id):
    link_list = []
    link = robot.links[link_id]
    for j_id in link.connect_joint:
      joint = robot.joints[j_id]
      link_list.append(joint.connect_link)

    return link_list
  
  # 閉リンク構造未対応
  def search_root_route(robot, link_id, tree_route):
    link = robot.links[link_id]

    if(link.is_edge() and (not link.is_root())):
      return False
    else:
      link_id_list = RobotStruct.search_neiborhood(robot, link_id)
      for l_id in link_id_list:
        if (robot.links[l_id].is_root()):
          route.append(l_id)
        else:
          route = RobotStruct.search_root_route(robot, l_id, tree_route)
          if (route):
            route.append(link_id)
      return route