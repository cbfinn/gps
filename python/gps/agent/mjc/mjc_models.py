import numpy as np

from gps.agent.mjc.model_builder import default_model, pointmass_model


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
    body.joint(axis="0 0 1",limited="false",name="joint0",pos="0 0 0",type="hinge")
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


def colored_reacher(ncubes=6, target_color="red", cube_size=0.015):
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
    body.joint(axis="0 0 1",limited="false",name="joint0",pos="0 0 0",type="hinge")
    body = body.body(name="body1",pos="0.1 0 0")
    body.joint(axis="0 0 1",limited="true",name="joint1",pos="0 0 0",range="-3.0 3.0",type="hinge")
    body.geom(fromto="0 0 0 0.1 0 0",name="link1",rgba="0.0 0.4 0.6 1",size=".01",type="capsule")
    body = body.body(name="fingertip",pos="0.11 0 0")
    body.site(name="fingertip",pos="0 0 0",size="0.01")
    body.geom(contype="0",name="fingertip",pos="0 0 0",rgba=COLOR_MAP[target_color],size=".01",type="sphere")

    # Target
    body = worldbody.body(name="target",pos=".1 -.1 .01")
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
    return mjcmodel

def pointmass():
    """
    An example usage of MJCModel building the pointmass task

    Returns:
        An MJCModel
    """
    mjcmodel = pointmass_model('pointmass')
    worldbody = mjcmodel.root.worldbody()

    # Particle
    body = worldbody.body(name='particle', pos="0 0 0")
    body.geom(name="particle_geom", type="capsule", fromto="-0.01 0 0 0.01 0 0", size="0.05")
    body.site(name="particle_site", pos="0 0 0", size="0.01")
    body.joint(name="ball_x", type="slide", pos="0 0 0", axis="1 0 0")
    body.joint(name="ball_y", type="slide", pos="0 0 0", axis="0 1 0")

    # Target
    body = worldbody.body(name="target", pos="3 0 0")
    body.geom(name="target_geom", type="capsule", fromto="-0.01 0 0 0.01 0 0", size="0.1", rgba="0 0.9 0.1 1")

    # Walls

