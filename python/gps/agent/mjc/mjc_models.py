import numpy as np

from gps.agent.mjc.model_builder import default_model, pointmass_model, MJCModel

COLOR_MAP = {
    'red': [1, 0, 0, 1],
    'green': [0, 1, 0, 1],
    'blue': [0, 0, 1, 1],
    'white': [1, 1, 1, 1],
    'yellow': [1, 1, 0, 1],
    'purple': [1, 0, 1, 1],
    'cyan': [0, 1, 1, 1],
}

def reacher():
    """
    An example usage of MJCModel building the reacher task

    Returns:
        An MJCModel
    """
    mjcmodel = default_model('reacher')
    worldbody = mjcmodel.root.worldbody()

    # Arena
    worldbody.geom(conaffinity="0",fromto="-.3 -.3 .01 .3 -.3 .01",name="sideS",rgba="0.9 0.4 0.6 1",size=".02",type="capsule")
    worldbody.geom(conaffinity="0",fromto=" .3 -.3 .01 .3  .3 .01",name="sideE",rgba="0.9 0.4 0.6 1",size=".02",type="capsule")
    worldbody.geom(conaffinity="0",fromto="-.3  .3 .01 .3  .3 .01",name="sideN",rgba="0.9 0.4 0.6 1",size=".02",type="capsule")
    worldbody.geom(conaffinity="0",fromto="-.3 -.3 .01 -.3 .3 .01",name="sideW",rgba="0.9 0.4 0.6 1",size=".02",type="capsule")

    # Arm
    worldbody.geom(conaffinity="0",contype="0",fromto="0 0 0 0 0 0.02",name="root",rgba="0.9 0.4 0.6 1",size=".011",type="cylinder")
    body = worldbody.body(name="body0", pos="0 0 .01")
    body.geom(fromto="0 0 0 0.1 0 0",name="link0",rgba="0.0 0.4 0.6 1",size=".01",type="capsule")
    # body.joint(axis="0 0 1",limited="false",name="joint0",pos="0 0 0",type="hinge")
    body.joint(axis="0 0 1",limited="true",name="joint0",pos="0 0 0",range="-3.14 3.14",type="hinge")
    body = body.body(name="body1",pos="0.1 0 0")
    body.joint(axis="0 0 1",limited="true",name="joint1",pos="0 0 0",range="-3.0 3.0",type="hinge")
    body.geom(fromto="0 0 0 0.1 0 0",name="link1",rgba="0.0 0.4 0.6 1",size=".01",type="capsule")
    body = body.body(name="fingertip",pos="0.11 0 0")
    body.site(name="fingertip",pos="0 0 0",size="0.01")
    body.geom(contype="0",name="fingertip",pos="0 0 0",rgba="0.0 0.8 0.6 1",size=".01",type="sphere")

    # Target
    body = worldbody.body(name="target",pos=".1 -.1 .01")
    body.geom(rgba="1. 0. 0. 1",type="box",size="0.01 0.01 0.01",density='0.00001',contype="0",conaffinity="0")
    body.site(name="target",pos="0 0 0",size="0.01")

    # Actuators
    actuator = mjcmodel.root.actuator()
    actuator.motor(ctrllimited="true",ctrlrange="-1.0 1.0",gear="200.0",joint="joint0")
    actuator.motor(ctrllimited="true",ctrlrange="-1.0 1.0",gear="200.0",joint="joint1")

    return mjcmodel

def weighted_reacher(density=1.0):
    """
    An example usage of MJCModel building the weighted reacher task
    Args:
        density: the density of the fingertip
    Returns:
        An MJCModel
    """
    mjcmodel = default_model('reacher')
    worldbody = mjcmodel.root.worldbody()

    # Arena
    worldbody.geom(conaffinity="0",fromto="-.3 -.3 .01 .3 -.3 .01",name="sideS",rgba="0.9 0.4 0.6 1",size=".02",type="capsule")
    worldbody.geom(conaffinity="0",fromto=" .3 -.3 .01 .3  .3 .01",name="sideE",rgba="0.9 0.4 0.6 1",size=".02",type="capsule")
    worldbody.geom(conaffinity="0",fromto="-.3  .3 .01 .3  .3 .01",name="sideN",rgba="0.9 0.4 0.6 1",size=".02",type="capsule")
    worldbody.geom(conaffinity="0",fromto="-.3 -.3 .01 -.3 .3 .01",name="sideW",rgba="0.9 0.4 0.6 1",size=".02",type="capsule")

    # Arm
    worldbody.geom(conaffinity="0",contype="0",fromto="0 0 0 0 0 0.02",name="root",rgba="0.9 0.4 0.6 1",size=".011",type="cylinder")
    body = worldbody.body(name="body0", pos="0 0 .01")
    body.geom(fromto="0 0 0 0.1 0 0",name="link0",rgba="0.0 0.4 0.6 1",size=".01",type="capsule")
    # body.joint(axis="0 0 1",limited="false",name="joint0",pos="0 0 0",type="hinge")
    body.joint(axis="0 0 1",limited="true",name="joint0",pos="0 0 0",range="-3.14 3.14",type="hinge")
    body = body.body(name="body1",pos="0.1 0 0")
    body.joint(axis="0 0 1",limited="true",name="joint1",pos="0 0 0",range="-3.0 3.0",type="hinge")
    body.geom(fromto="0 0 0 0.1 0 0",name="link1",rgba="0.0 0.4 0.6 1",size=".01",type="capsule")
    body = body.body(name="fingertip",pos="0.11 0 0")
    body.site(name="fingertip",pos="0 0 0",size="0.01")
    body.geom(contype="0",name="fingertip",pos="0 0 0",rgba="0.0 0.8 0.6 1",size=".01",type="sphere", density=density)

    # Target
    body = worldbody.body(name="target",pos=".1 -.1 .01")
    body.geom(rgba="1. 0. 0. 1",type="box",size="0.01 0.01 0.01",density='0.00001',contype="0",conaffinity="0")
    body.site(name="target",pos="0 0 0",size="0.01")

    # Actuators
    actuator = mjcmodel.root.actuator()
    actuator.motor(ctrllimited="true",ctrlrange="-1.0 1.0",gear="200.0",joint="joint0")
    actuator.motor(ctrllimited="true",ctrlrange="-1.0 1.0",gear="200.0",joint="joint1")

    return mjcmodel


def colored_reacher(ncubes=6, target_color="red", cube_size=0.012, target_pos=(.1,-.1)):
    mjcmodel = default_model('reacher', regen_fn=lambda: colored_reacher(ncubes, target_color, cube_size, target_pos))
    worldbody = mjcmodel.root.worldbody()

    # Arena
    #worldbody.geom(conaffinity="0",fromto="-.3 -.3 .01 .3 -.3 .01",name="sideS",rgba="0.9 0.4 0.6 1",size=".02",type="capsule")
    #worldbody.geom(conaffinity="0",fromto=" .3 -.3 .01 .3  .3 .01",name="sideE",rgba="0.9 0.4 0.6 1",size=".02",type="capsule")
    #worldbody.geom(conaffinity="0",fromto="-.3  .3 .01 .3  .3 .01",name="sideN",rgba="0.9 0.4 0.6 1",size=".02",type="capsule")
    #worldbody.geom(conaffinity="0",fromto="-.3 -.3 .01 -.3 .3 .01",name="sideW",rgba="0.9 0.4 0.6 1",size=".02",type="capsule")


    # Arm
    worldbody.geom(conaffinity="0",contype="0",fromto="0 0 0 0 0 0.02",name="root",rgba="0.9 0.4 0.6 1",size=".011",type="cylinder")
    body = worldbody.body(name="body0", pos="0 0 .01")
    body.geom(fromto="0 0 0 0.1 0 0",name="link0",rgba="0.0 0.4 0.6 1",size=".01",type="capsule")
    # body.joint(axis="0 0 1",limited="false",name="joint0",pos="0 0 0",type="hinge")
    body.joint(axis="0 0 1",limited="true",name="joint0",pos="0 0 0",range="-3.14 3.14",type="hinge")
    body = body.body(name="body1",pos="0.1 0 0")
    body.joint(axis="0 0 1",limited="true",name="joint1",pos="0 0 0",range="-3.0 3.0",type="hinge")
    body.geom(fromto="0 0 0 0.1 0 0",name="link1",rgba="0.0 0.4 0.6 1",size=".01",type="capsule")
    body = body.body(name="fingertip",pos="0.11 0 0")
    body.site(name="fingertip",pos="0 0 0",size="0.01")
    body.geom(contype="0",name="fingertip",pos="0 0 0",rgba=COLOR_MAP[target_color],size=".01",type="sphere")

    # Target
    _target_pos = [target_pos[0], target_pos[1], 0.01]
    body = worldbody.body(name="target",pos=_target_pos)
    body.geom(rgba=COLOR_MAP[target_color],type="box",size=cube_size*np.ones(3),density='0.00001',contype="0",conaffinity="0")
    body.site(name="target",pos="0 0 0",size="0.01")

    # Distractor cubes
    available_colors = COLOR_MAP.keys()
    available_colors.remove(target_color)
    for i in range(ncubes-1):
        pos = np.random.rand(3)
        pos = pos*0.5-0.25
        pos[2] = 0.01
        body = worldbody.body(name="cube_%d"%i,pos=pos)

        color = np.random.choice(available_colors)
        body.geom(rgba=COLOR_MAP[color],type="box",size=cube_size*np.ones(3),density='0.00001',contype="0",conaffinity="0")

    # Actuators
    actuator = mjcmodel.root.actuator()
    actuator.motor(ctrllimited="true",ctrlrange="-1.0 1.0",gear="200.0",joint="joint0")
    actuator.motor(ctrllimited="true",ctrlrange="-1.0 1.0",gear="200.0",joint="joint1")


    #Background
    background = worldbody.body(name='background_body', pos=[0,0,-10], axisangle=[0,1,0,0.05])
    background_color = [0,0,0,1] #[0.2,0.2,0.2,1]
    background.geom(name='background_box', type='box', rgba=background_color, size=[100,100,.1], contype=3, conaffinity=3)
    return mjcmodel


def obstacle_pointmass(target_position=np.array([1.3, 0.5, 0]), wall_center=0.0, hole_height=1.0, control_limit=100,
                       add_hole_indicator=False):
    """
    An example usage of MJCModel building the pointmass task
    Args:
        target_position: the position of the target.
        wall_center: center of wall hole, y-coordinate
        hole_height:
        wall_1_center: the center of the first wall.
        wall_2_center: the center of the second wall.
        wall_height: the height of each wall.
    Returns:
        An MJCModel
    """
    mjcmodel = pointmass_model('pointmass')
    worldbody = mjcmodel.root.worldbody()

    background = worldbody.body(name='background_body', pos=[0,0,-1], axisangle=[0,1,0,0.05])
    background_color = [0,0,0,1] #[0.2,0.2,0.2,1]
    background.geom(name='background_box', type='box', rgba=background_color, size=[100,100,.1], contype=3, conaffinity=3)

    # Particle
    body = worldbody.body(name='particle', pos="0 0 0")
    # body.geom(name="particle_geom", type="capsule", fromto="-0.01 0 0 0.01 0 0", size="0.05")
    body.geom(name="particle_geom", type="sphere", rgba=[.4,.4,1,1], size="0.07")
    body.site(name="particle_site", pos="0 0 0", size="0.01")
    body.joint(name="ball_x", type="slide", pos="0 0 0", axis="1 0 0")
    body.joint(name="ball_y", type="slide", pos="0 0 0", axis="0 1 0")

    # Target
    body = worldbody.body(name="target", pos=target_position)
    # body.geom(name="target_geom", type="capsule", fromto="-0.01 0 0 0.01 0 0", size="0.05", rgba="0 0.9 0.1 1")
    body.geom(name="target_geom", type="sphere", size="0.07", rgba="0 0.9 0.1 1")

    # Walls
    wall_x = 0.5
    wall_z = 0.0
    h = hole_height
    wall_1_center = [wall_x, wall_center-h/2, wall_z]
    wall_2_center = [wall_x, wall_center+h/2, wall_z]

    body = worldbody.body(name="wall1", pos=wall_1_center)
    # body = worldbody.body(name="wall1", pos=np.array([0.5, -0.3, 0.]))
    y1, y2 = wall_1_center[1], wall_2_center[1]
    body.geom(name="wall1_geom", type="capsule", fromto=np.array([0., y1-10, 0., 0., y1, 0.]), size="0.1", contype="1", rgba="0.9 0 0.1 1")
    # body.geom(name="wall1_geom", type="capsule", fromto=np.array([0., 0., 0., 1., 0., 0.]), size="0.1", contype="1", rgba="0.9 0 0.1 1")
    body = worldbody.body(name="wall2", pos=wall_2_center)
    # body = worldbody.body(name="wall2", pos=np.array([0.15, -0.3, 0.]))
    body.geom(name="wall2_geom", type="capsule", fromto=np.array([0., y2, 0., 0., y2+10, 0.]), size="0.1", contype="1", rgba="0.9 0 0.1 1")
    # body.geom(name="wall2_geom", type="capsule", fromto=np.array([0., 0., 0., -1., 0., 0.]), size="0.1", contype="1", rgba="0.9 0 0.1 1")
    if add_hole_indicator:
        y3 = wall_center
        h = 1.0
        body = worldbody.body(name="hole_indicator", pos=[wall_x, wall_center, wall_z])
        body.geom(name="hole_indicator", type="capsule", fromto=np.array([0., -y3-h, 0., 0., y3+h, 0.]), size="0.1", contype="2", rgba="0.0 0.9 0.9 1")


    # Actuators
    actuator = mjcmodel.root.actuator()
    actuator.motor(joint="ball_x", ctrlrange=[-control_limit, control_limit], ctrllimited="true")
    actuator.motor(joint="ball_y", ctrlrange=[-control_limit, control_limit], ctrllimited="true")
    return mjcmodel

def weighted_pointmass(target_position=np.array([1.3, 0.5, 0]), density=0.01, control_limit=1.0):
    """
    An example usage of MJCModel building the pointmass task
    Args:
        target_position: the position of the target.
        density: the density of the pointmass
        control_limit: the control range of the pointmass 
    Returns:
        An MJCModel
    """
    mjcmodel = pointmass_model('pointmass')
    worldbody = mjcmodel.root.worldbody()

    # Particle
    body = worldbody.body(name='particle', pos="0 0 0")
    # body.geom(name="particle_geom", type="capsule", fromto="-0.01 0 0 0.01 0 0", size="0.05")
    body.geom(name="particle_geom", type="sphere", density=density, size="0.05")
    body.site(name="particle_site", pos="0 0 0", size="0.01")
    body.joint(name="ball_x", type="slide", pos="0 0 0", axis="1 0 0")
    body.joint(name="ball_y", type="slide", pos="0 0 0", axis="0 1 0")

    # Target
    body = worldbody.body(name="target", pos=target_position)
    # body.geom(name="target_geom", type="capsule", fromto="-0.01 0 0 0.01 0 0", size="0.05", rgba="0 0.9 0.1 1")
    body.geom(name="target_geom", type="sphere", size="0.05", rgba="0 0.9 0.1 1")

    # Actuators
    actuator = mjcmodel.root.actuator()
    actuator.motor(joint="ball_x", ctrlrange=[-control_limit, control_limit], ctrllimited="true")
    actuator.motor(joint="ball_y", ctrlrange=[-control_limit, control_limit], ctrllimited="true")
    return mjcmodel


def block_push():
    mjcmodel = MJCModel('block_push')
    mjcmodel.root.compiler(inertiafromgeom="true",angle="radian",coordinate="local")
    mjcmodel.root.option(timestep="0.01",gravity="0 0 0",iterations="20",integrator="Euler")
    default = mjcmodel.root.default()
    default.joint(armature='0.04', damping=1, limited='true')
    default.geom(friction=".8 .1 .1",density="300",margin="0.002",condim="1",contype="1",conaffinity="1")

    worldbody = mjcmodel.root.worldbody()

    palm = worldbody.body(name='palm', pos=[0,0,0])
    palm.geom(type='capsule', fromto=[0,-0.1,0,0,0.1,0], size=.12)
    proximal1 = palm.body(name='proximal_1', pos=[0,0,0])
    proximal1.joint(name='proximal_j_1', type='hinge', pos=[0,0,0], axis=[0,1,0], range=[-2.5,2.3])
    proximal1.geom(type='capsule', fromto=[0,0,0,0.4,0,0], size=0.06, contype=1, conaffinity=1)
    distal1 = proximal1.body(name='distal_1', pos=[0.4,0,0])
    distal1.joint(name = "distal_j_1", type = "hinge", pos = "0 0 0", axis = "0 1 0", range = "-2.3213 2.3", damping = "1.0")
    distal1.geom(type="capsule", fromto="0 0 0 0.4 0 0", size="0.06", contype="1", conaffinity="1")
    distal2 = distal1.body(name='distal_2', pos=[0.4,0,0])
    distal2.joint(name="distal_j_2",type="hinge",pos="0 0 0",axis="0 1 0",range="-2.3213 2.3",damping="1.0")
    distal2.geom(type="capsule",fromto="0 0 0 0.4 0 0",size="0.06",contype="1",conaffinity="1")
    distal4 = distal2.body(name='distal_4', pos=[0.4,0,0])
    distal4.site(name="tip arml",pos="0.1 0 -0.2",size="0.01")
    distal4.site(name="tip armr",pos="0.1 0 0.2",size="0.01")
    distal4.joint(name="distal_j_3",type="hinge",pos="0 0 0",axis="1 0 0",range="-3.3213 3.3",damping="0.5")
    distal4.geom(type="capsule",fromto="0 0 -0.2 0 0 0.2",size="0.04",contype="1",conaffinity="1")
    distal4.geom(type="capsule",fromto="0 0 -0.2 0.2 0 -0.2",size="0.04",contype="1",conaffinity="1")
    distal4.geom(type="capsule",fromto="0 0 0.2 0.2 0 0.2",size="0.04",contype="1",conaffinity="1")

    object = worldbody.body(name='object', pos=[0,0,0])
    object.geom(rgba="1. 1. 1. 1",type="box",size="0.05 0.05 0.05",density='0.00001',contype="1",conaffinity="1")
    object.joint(name="obj_slidez",type="slide",pos="0.025 0.025 0.025",axis="0 0 1",range="-10.3213 10.3",damping="0.5")
    object.joint(name="obj_slidex",type="slide",pos="0.025 0.025 0.025",axis="1 0 0",range="-10.3213 10.3",damping="0.5")
    distal10 = object.body(name='distal_10', pos=[0,0,0])
    distal10.site(name='obj_pos', pos=[0.025,0.025,0.025], size=0.01)

    goal = worldbody.body(name='goal', pos=[0,0,0])
    goal.geom(rgba="1. 0. 0. 1",type="box",size="0.1 0.1 0.1",density='0.00001',contype="0",conaffinity="0")
    distal11 = goal.body(name='distal_11', pos=[0,0,0])
    distal11.site(name='goal_pos', pos=[0.05,0.05,0.05], size=0.01)


    actuator = mjcmodel.root.actuator()
    actuator.motor(joint="proximal_j_1",ctrlrange="-2 2",ctrllimited="true")
    actuator.motor(joint="distal_j_1",ctrlrange="-2 2",ctrllimited="true")
    actuator.motor(joint="distal_j_2",ctrlrange="-2 2",ctrllimited="true")
    actuator.motor(joint="distal_j_3",ctrlrange="-2 2",ctrllimited="true")

    import sys; mjcmodel.root.write(sys.stdout)

    return mjcmodel

