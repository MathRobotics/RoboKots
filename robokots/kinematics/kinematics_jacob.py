#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.08.12 Created by T.Ishigaki

import numpy as np

class BasicJacobian:

  @staticmethod
  def calc(self, robot, link_id):
    return None
  
class VelJacobian(BasicJacobian):
  
  @staticmethod
  def calc(robot, link_id):
    jacob = np.zeros(6, robot.dof)

    link = robot.links[link_id]

    while(not link.is_root()):

      jacob[:,robot.dof]

    return jacob
  
  @staticmethod
  def calc_all_link(self, robot):

    return None