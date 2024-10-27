<?xml version="1.0"?>
<robot>
  <joint_list>
    <joint
      name = "root_joint"
      joint_type = "fix"
    >
    </joint>
    <joint
      name = "joint1"
      joint_type = "revolution"
    >
    </joint>
    <joint
      name = "joint2"
      joint_type = "revolution"
    >
    </joint>
    <joint
      name = "joint3"
      joint_type = "revolution"
    >
    </joint>
  </joint_list>
  <link_list>
    <link
      name="root_link"
      link_type="rigid"
      cog = "[0., 1., 0.]"
      mass = "100."
      inertia_param = "[1., 1., 1., 0., 0., 0.]"
    >
      <joint
        name = "root_joint">
      </joint>
      <joint
        name = "joint1"
        connect_pos = "[1., 0., 0.]"
        connect_rot =
        "[
          [1., 0., 0.],
          [0., 1., 0.],
          [0., 0., 1.]
        ]">
      </joint>
    </link>
    <link
      name="link1"
      joint_type="revolution"
      link_type="rigid"
      cog = "[0., 1., 0.]"
      mass = "100."
      inertia_param = "[1., 1., 1., 0., 0., 0.]"
    >
      <joint
        name = "joint1">
      </joint>
      <joint
        name = "joint2"
        connect_pos = "[1., 0., 0.]"
        connect_rot =
        "[
          [1., 0., 0.],
          [0., 1., 0.],
          [0., 0., 1.]
        ]">
      </joint>
    </link>
    <link
      name="link2"
      joint_type="revolution"
      link_type="rigid"
      cog = "[0., 1., 0.]"
      mass = "100."
      inertia_param = "[1., 1., 1., 0., 0., 0.]"
    >
      <joint
        name = "joint2">
      </joint>
      <joint
        name = "joint3"
        connect_pos = "[1., 0., 0.]"
        connect_rot =
        "[
          [1., 0., 0.],
          [0., 1., 0.],
          [0., 0., 1.]
        ]">
      </joint>
    </link>
    <link
      name="link3"
      joint_type="revolution"
      link_type="rigid"
      cog = "[0., 1., 0.]"
      mass = "100."
      inertia_param = "[1., 1., 1., 0., 0., 0.]"
    >
      <joint
        name = "joint3">
      </joint>
    </link>  
  </link_list>
</robot>