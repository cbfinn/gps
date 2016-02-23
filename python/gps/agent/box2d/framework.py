#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/python
#
# C++ version Copyright (c) 2006-2007 Erin Catto http://www.box2d.org
# Python version by Ken Lauer / sirkne at gmail dot com
#
# This software is provided 'as-is', without any express or implied
# warranty.  In no event will the authors be held liable for any damages
# arising from the use of this software.
# Permission is granted to anyone to use this software for any purpose,
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions:
# 1. The origin of this software must not be misrepresented; you must not
# claim that you wrote the original software. If you use this software
# in a product, an acknowledgment in the product documentation would be
# appreciated but is not required.
# 2. Altered source versions must be plainly marked as such, and must not be
# misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.

"""
The file sets up the basics of a Box2d world.
Future worlds should use this as the base class.
"""
import Box2D as b2
from time import time
from gps.agent.box2d.settings import fwSettings


class fwQueryCallback(b2.b2QueryCallback):
    """
    This callback for each fixture in the world.
    """
    def __init__(self, p):
        super(fwQueryCallback, self).__init__()
        self.point = p
        self.fixture = None

    def ReportFixture(self, fixture):
        """
        This method is called to query for a fixture.
        """
        body = fixture.body
        if body.type == b2.b2_dynamicBody:
            inside = fixture.TestPoint(self.point)
            if inside:
                self.fixture = fixture
                # We found the object, so stop the query
                return False
        # Continue the query
        return True

class FrameworkBase(b2.b2ContactListener):
    """
    The base of the main Box2D GUI framework.

    """
    name = "None"
    description = None
    TEXTLINE_START = 30
    colors = {
        'joint_line' : b2.b2Color(0.8, 0.8, 0.8),
        'contact_add' : b2.b2Color(0.3, 0.95, 0.3),
        'contact_persist' : b2.b2Color(0.3, 0.3, 0.95),
        'contact_normal' : b2.b2Color(0.4, 0.9, 0.4),
    }

    def __reset(self):
        """ Reset all of the variables to their starting values.
        Not to be called except at initialization."""
        # Box2D-related
        self.points = []
        self.settings = fwSettings
        self.using_contacts = False
        self.stepCount = 0

        # Box2D-callbacks
        self.destructionListener = None
        self.renderer = None

    def __init__(self):
        super(FrameworkBase, self).__init__()

        self.__reset()

        # Box2D Initialization
        self.world = b2.b2World(gravity=(0, -10), doSleep=True)

        self.world.contactListener = self
        self.t_steps, self.t_draws = [], []

    def __del__(self):
        pass

    def Step(self, settings, action=None):
        """
        The main physics step.

        Takes care of physics drawing
        (callbacks are executed after the world.Step() )
        and drawing additional information.
        """
        assert action is None,\
            'action should only be used in subclass'

        self.stepCount += 1
        # Don't do anything if the setting's Hz are <= 0
        if settings.hz > 0.0:
            timeStep = 1.0 / settings.hz
        else:
            timeStep = 0.0

        # Set the flags based on what the settings show
        if self.renderer:
            self.renderer.flags = dict(
                drawShapes=settings.drawShapes,
                drawJoints=settings.drawJoints,
                drawAABBs=settings.drawAABBs,
                drawPairs=settings.drawPairs,
                drawCOMs=settings.drawCOMs,
                convertVertices=isinstance(self.renderer, b2.b2DrawExtended)
            )

        # Set the other settings that aren't contained in the flags
        self.world.warmStarting = settings.enableWarmStarting
        self.world.continuousPhysics = settings.enableContinuous
        self.world.subStepping = settings.enableSubStepping

        # Reset the collision points
        self.points = []

        # Tell Box2D to step
        t_step = time()
        self.world.Step(timeStep, settings.velocityIterations,
                        settings.positionIterations)
        t_step = time()-t_step

        # Update the debug draw settings so that the vertices will be properly
        # converted to screen coordinates
        t_draw = time()
        if self.renderer:
            self.renderer.StartDraw()

        self.world.DrawDebugData()

        if self.renderer:


            # Draw each of the contact points in different colors.
            if self.settings.drawContactPoints:
                for point in self.points:
                    if point['state'] == b2.b2_addState:
                        self.renderer.DrawPoint(self.renderer.to_screen(
                            point['position']), settings.pointSize,
                                                self.colors['contact_add'])
                    elif point['state'] == b2.b2_persistState:
                        self.renderer.DrawPoint(self.renderer.to_screen(
                            point['position']), settings.pointSize,
                                                self.colors['contact_persist'])

            if settings.drawContactNormals:
                for point in self.points:
                    p1 = self.renderer.to_screen(point['position'])
                    p2 = self.renderer.axisScale * point['normal'] + p1
                    self.renderer.DrawSegment(p1, p2,
                                              self.colors['contact_normal'])

            self.renderer.EndDraw()
            t_draw = time()-t_draw

            t_draw = max(b2.b2_epsilon, t_draw)
            t_step = max(b2.b2_epsilon, t_step)


            self.t_draws.append(1.0/t_draw)
            self.t_steps.append(1.0/t_step)


    def SimulationLoop(self, action):
        """
        The main simulation loop. Don't override this, override Step instead.
        """

        # Reset the text line to start the text from the top
        self.textLine = self.TEXTLINE_START

        # Draw the name of the test running
        self.Print(self.name, (127, 127, 255))

        if self.description:
            # Draw the name of the test running
            for s in self.description.split('\n'):
                self.Print(s, (127, 255, 127))

        self.Step(self.settings, action)

    def PreSolve(self, contact, old_manifold):
        """
        This is a critical function when there are many contacts in the world.
        It should be optimized as much as possible.
        """
        if not (self.settings.drawContactPoints or
                self.settings.drawContactNormals or self.using_contacts):
            return
        elif len(self.points) > self.settings.maxContactPoints:
            return

        manifold = contact.manifold
        if manifold.pointCount == 0:
            return

        _, state2 = b2.b2GetPointStates(old_manifold, manifold)
        if not state2:
            return

        worldManifold = contact.worldManifold

        for i, _ in enumerate(state2):
            self.points.append(
                {
                    'fixtureA' : contact.fixtureA,
                    'fixtureB' : contact.fixtureB,
                    'position' : worldManifold.points[i],
                    'normal' : worldManifold.normal,
                    'state' : state2[i]
                })


framework_module = __import__('gps.agent.box2d.'+'%s_framework' %
                              (fwSettings.backend.lower()),
                              fromlist=['%sFramework'
                                        %fwSettings.backend.capitalize()])
Framework = getattr(framework_module, '%sFramework' %
                    fwSettings.backend.capitalize())


