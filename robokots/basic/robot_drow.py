#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025.04.06 Created by T.Ishigaki

import numpy as np
import matplotlib.pyplot as plt

def d_set_equall_aspect_3d(ax, data, margin):
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
  
def d_show_robot(joint_conectivity, marker_pos, save = False):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  ax.scatter(marker_pos[:,0], marker_pos[:,1], marker_pos[:,2], c='r', marker='o')

  for c in joint_conectivity:
    c_id = c[0]
    p_id = c[1]
    if 0 <= c_id < marker_pos.shape[0] and 0 <= p_id < marker_pos.shape[0]:
      ax.plot(
        [marker_pos[c_id,0], marker_pos[p_id,0]], 
        [marker_pos[c_id,1], marker_pos[p_id,1]], 
        [marker_pos[c_id,2], marker_pos[p_id,2]], 'b')
    else:
      print(f"Invalid indices: c_id={c_id}, p_id={p_id}")

  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  
  d_set_equall_aspect_3d(ax, marker_pos, 0.1)

  plt.show()
  if save:  
    plt.savefig('simple_draw.png')

def d_show_link_points(link_pos):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  ax.scatter(link_pos[:,0], link_pos[:,1], link_pos[:,2], c='r', marker='o')

  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')

  d_set_equall_aspect_3d(ax, link_pos, 0.1)

  plt.show()