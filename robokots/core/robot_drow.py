#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025.04.06 Created by T.Ishigaki

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class RobotColor:
  def __init__(self, link_color = 'blue', joint_color = 'red'):
    self.link_color = link_color
    self.joint_color = joint_color

def set_equall_aspect(ax, data, margin):
  margin = 0.1
  ax_min = np.zeros(2)
  ax_max = np.zeros(2)
  box_length = np.zeros(2)
  for i in range(2):
    ax_min[i] = min(data[:,i])-margin
    ax_max[i] = max(data[:,i])+margin
    box_length[i] = ax_max[i] - ax_min[i]
    
  box_length_max = max((box_length[0], box_length[1]))
  box_ratio = box_length_max / box_length

  ax_ave = (ax_max + ax_min) / 2

  ax.set_aspect("equal")
  ax.set_xlim(ax_ave[0] - box_length[0]*box_ratio[0]*0.5, ax_ave[0] + box_length[0]*box_ratio[0]*0.5)
  ax.set_ylim(ax_ave[1] - box_length[1]*box_ratio[1]*0.5, ax_ave[1] + box_length[1]*box_ratio[1]*0.5)

def set_equall_aspect_3d(ax, data, margin):
  margin = 0.1
  ax_min = np.zeros(3)
  ax_max = np.zeros(3)
  box_length = np.zeros(3)
  for i in range(3):
    ax_min[i] = min(data[:,i])-margin
    ax_max[i] = max(data[:,i])+margin
    box_length[i] = ax_max[i] - ax_min[i]
    
  box_length_max = max((box_length[0], box_length[1], box_length[2]))
  box_ratio = box_length_max / box_length

  ax_ave = (ax_max + ax_min) / 2

  ax.set_box_aspect((box_length_max,box_length_max,box_length_max))
  ax.set_xlim3d(ax_ave[0] - box_length[0]*box_ratio[0]*0.5, ax_ave[0] + box_length[0]*box_ratio[0]*0.5)
  ax.set_ylim3d(ax_ave[1] - box_length[1]*box_ratio[1]*0.5, ax_ave[1] + box_length[1]*box_ratio[1]*0.5)
  ax.set_zlim3d(ax_ave[2] - box_length[2]*box_ratio[2]*0.5, ax_ave[2] + box_length[2]*box_ratio[2]*0.5)
  
def show_robot(joint_conectivity, marker_pos, save = False, ax = None, color : RobotColor = None):
  if ax is None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

  if color is None:
    color = RobotColor()

  ax.scatter(marker_pos[:,0], marker_pos[:,1], marker_pos[:,2], c=color.joint_color, marker='o')

  for c in joint_conectivity:
    c_id = c[0]
    p_id = c[1]
    if 0 <= c_id < marker_pos.shape[0] and 0 <= p_id < marker_pos.shape[0]:
      ax.plot(
        [marker_pos[c_id,0], marker_pos[p_id,0]], 
        [marker_pos[c_id,1], marker_pos[p_id,1]], 
        [marker_pos[c_id,2], marker_pos[p_id,2]], c=color.link_color)
    else:
      print(f"Invalid indices: c_id={c_id}, p_id={p_id}")

  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  
  set_equall_aspect_3d(ax, marker_pos, 0.1)

  if ax is None:
    plt.show()
    
  if save:  
    plt.savefig('simple_draw.png')

def show_robot_traj(joint_conectivity, marker_pos_list, save = False, ax = None, color : RobotColor = None):
  if ax is None:
    fig = plt.figure()
    ax_ = fig.add_subplot(111, projection='3d')
  else:
    ax_ = ax

  if color is None:
    color = RobotColor()

  arr_swapped = marker_pos_list.transpose(1, 0, 2)

  if isinstance(ax, Axes3D):
    for marker_pos in arr_swapped:
      ax_.scatter(marker_pos[:,0], marker_pos[:,1], marker_pos[:,2], c=color.joint_color, marker='o', alpha=0.1)
      for c in joint_conectivity:
        c_id = c[0]
        p_id = c[1]
        if 0 <= c_id < marker_pos.shape[0] and 0 <= p_id < marker_pos.shape[0]:
          ax_.plot(
            [marker_pos[c_id,0], marker_pos[p_id,0]], 
            [marker_pos[c_id,1], marker_pos[p_id,1]], 
            [marker_pos[c_id,2], marker_pos[p_id,2]], color.link_color, alpha=0.1)
        else:
          print(f"Invalid indices: c_id={c_id}, p_id={p_id}")
      
    ax_.set_xlabel('X')
    ax_.set_ylabel('Y')
    ax_.set_zlabel('Z')
    
    set_equall_aspect_3d(ax_, marker_pos, 0.1)
  else:
    for marker_pos in arr_swapped:
      ax_.scatter(marker_pos[:,0], marker_pos[:,1], c=color.joint_color, marker='o', alpha=0.1)
      for c in joint_conectivity:
        c_id = c[0]
        p_id = c[1]
        if 0 <= c_id < marker_pos.shape[0] and 0 <= p_id < marker_pos.shape[0]:
          ax_.plot(
            [marker_pos[c_id,0], marker_pos[p_id,0]], 
            [marker_pos[c_id,1], marker_pos[p_id,1]], color.link_color, alpha=0.1)
        else:
          print(f"Invalid indices: c_id={c_id}, p_id={p_id}")
      
    ax_.set_xlabel('X')
    ax_.set_ylabel('Y')

    set_equall_aspect(ax_, marker_pos, 0.1)
    

  if ax is None:
    plt.show()

  if save:  
    plt.savefig('simple_draw.png')

def show_link_points(link_pos, ax = None, dimension=3):
  if ax is None:
    plot = plt.figure()
    if dimension == 2:
      ax = plot.add_subplot(111)
      ax.axis('equal')
      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.grid()
    elif dimension == 3:
      ax = plot.add_subplot(111, projection='3d')
      set_equall_aspect_3d(ax, link_pos, 0.1)
      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.set_zlabel('Z')
      
  ax.scatter(link_pos[:,0], link_pos[:,1], c='r', marker='o')  
  plt.show()