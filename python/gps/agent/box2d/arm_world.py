""" This file defines an environment for the Box2D 2 Link Arm simulator. """
import Box2D as b2
import numpy as np
from framework import Framework
from gps.agent.box2d.settings import fwSettings
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS

class ArmWorld(Framework):
    """ This class defines the 2 Link Arm and its environment."""
    name = "2 Link Arm"
    def __init__(self, x0, target, render):
        self.render = render
        if self.render:
            super(ArmWorld, self).__init__()
        else:
            self.world = b2.b2World(gravity=(0, -10), doSleep=True)

        self.world.gravity = (0.0, 0.0)

        fixture_length = 5
        self.x0 = x0

        rectangle_fixture = b2.b2FixtureDef(
            shape=b2.b2PolygonShape(box=(.5, fixture_length)),
            density=.5,
            friction=1,
        )
        square_fixture = b2.b2FixtureDef(
            shape=b2.b2PolygonShape(box=(1, 1)),
            density=100.0,
            friction=1,
        )
        self.base = self.world.CreateBody(
            position=(0, 15),
            fixtures=square_fixture,
        )

        self.body1 = self.world.CreateDynamicBody(
            position=(0, 2),
            fixtures=rectangle_fixture,
            angle=b2.b2_pi,
        )

        self.body2 = self.world.CreateDynamicBody(
            fixtures=rectangle_fixture,
            position=(0, 2),
            angle=b2.b2_pi,
        )
        self.target1 = self.world.CreateDynamicBody(
            fixtures=rectangle_fixture,
            position=(0, 0),
            angle=b2.b2_pi,
        )
        self.target2 = self.world.CreateDynamicBody(
            fixtures=rectangle_fixture,
            position=(0, 0),
            angle=b2.b2_pi,
        )

        self.joint1 = self.world.CreateRevoluteJoint(
            bodyA=self.base,
            bodyB=self.body1,
            localAnchorA=(0, 0),
            localAnchorB=(0, fixture_length),
            enableMotor=True,
            maxMotorTorque=400,
            enableLimit=False,
        )

        self.joint2 = self.world.CreateRevoluteJoint(
            bodyA=self.body1,
            bodyB=self.body2,
            localAnchorA=(0, -(fixture_length - 0.5)),
            localAnchorB=(0, fixture_length - 0.5),
            enableMotor=True,
            maxMotorTorque=400,
            enableLimit=False,
        )

        self.set_joint_angles(self.body1, self.body2, x0[0], x0[1])
        self.set_joint_angles(self.target1, self.target2, target[0], target[1])
        self.target1.active = False
        self.target2.active = False

        self.joint1.motorSpeed = x0[2]
        self.joint2.motorSpeed = x0[3]

    def set_joint_angles(self, body1, body2, angle1, angle2):
        """ Converts the given absolute angle of the arms to joint angles"""
        pos = self.base.GetWorldPoint((0, 0))
        body1.angle = angle1 + np.pi
        new_pos = body1.GetWorldPoint((0, 5))
        body1.position += pos - new_pos
        body2.angle = angle2 + body1.angle
        pos = body1.GetWorldPoint((0, -4.5))
        new_pos = body2.GetWorldPoint((0, 4.5))
        body2.position += pos - new_pos


    def run(self):
        """Initiates the first time step
        """
        if self.render:
            super(ArmWorld, self).run()
        else:
            self.run_next(None)

    def run_next(self, action):
        """Moves forward in time one step. Calls the renderer if applicable."""
        if self.render:
            super(ArmWorld, self).run_next(action)
        else:
            if action is not None:
                self.joint1.motorSpeed = action[0]
                self.joint2.motorSpeed = action[1]
            self.world.Step(1.0 / fwSettings.hz, fwSettings.velocityIterations,
                            fwSettings.positionIterations)

    def Step(self, settings, action):
        """Moves forward in time one step. Called by the renderer"""
        self.joint1.motorSpeed = action[0]
        self.joint2.motorSpeed = action[1]

        super(ArmWorld, self).Step(settings)

    def reset_world(self):
        """Returns the world to its intial state"""
        self.world.ClearForces()
        self.joint1.motorSpeed = 0
        self.joint2.motorSpeed = 0
        self.body1.linearVelocity = (0, 0)
        self.body1.angularVelocity = 0
        self.body2.linearVelocity = (0, 0)
        self.body2.angularVelocity = 0
        self.set_joint_angles(self.body1, self.body2, self.x0[0], self.x0[1])


    def get_state(self):
        """Retrieves the state of the point mass"""
        state = {JOINT_ANGLES: np.array([self.joint1.angle,
                                         self.joint2.angle]),
                 JOINT_VELOCITIES: np.array([self.joint1.speed,
                                             self.joint2.speed]),
                 END_EFFECTOR_POINTS: np.append(np.array(self.body2.position),[0])}

        return state

