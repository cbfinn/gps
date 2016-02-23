#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C++ version Copyright (c) 2006-2007 Erin Catto http://www.box2d.org
# Python version Copyright (c) 2010 kne / sirkne at gmail dot com
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
This file contains the Framework for the Box2D GUI
"""

import Box2D as b2
import pygame
import framework


GUIEnabled = False

class PygameDraw(b2.b2DrawExtended):
    """
    This debug draw class accepts callbacks from Box2D and
    handles all of the rendering.
    """
    surface = None
    axisScale = 10.0
    def __init__(self, test=None, **kwargs):
        b2.b2DrawExtended.__init__(self, **kwargs)
        self.flipX = False
        self.flipY = True
        self.convertVertices = True
        self.test = test

    def StartDraw(self):
        """
        Called by renderer before drawing.
        """
        self.zoom = self.test.viewZoom
        self.center = self.test.viewCenter
        self.offset = self.test.viewOffset
        self.screenSize = self.test.screenSize

    def EndDraw(self):
        """
        Called by renderer when finished drawing.
        """

        pass

    def DrawPoint(self, p, size, color):
        """
        Draw a single point at point p given a pixel size and color.
        """
        self.DrawCircle(p, size/self.zoom, color, drawwidth=0)

    def DrawAABB(self, aabb, color):
        """
        Draw a wireframe around the AABB with the given color.
        """
        points = [(aabb.lowerBound.x, aabb.lowerBound.y),
                  (aabb.upperBound.x, aabb.lowerBound.y),
                  (aabb.upperBound.x, aabb.upperBound.y),
                  (aabb.lowerBound.x, aabb.upperBound.y)]

        pygame.draw.aalines(self.surface, color, True, points)

    def DrawSegment(self, p1, p2, color):
        """
        Draw the line segment from p1-p2 with the specified color.
        """
        pygame.draw.aaline(self.surface, color.bytes, p1, p2)

    def DrawTransform(self, xf):
        """
        Draw the transform xf on the screen
        """
        p1 = xf.position
        p2 = self.to_screen(p1 + self.axisScale * xf.R.col1)
        p3 = self.to_screen(p1 + self.axisScale * xf.R.col2)
        p1 = self.to_screen(p1)

        pygame.draw.aaline(self.surface, (255, 0, 0), p1, p2)
        pygame.draw.aaline(self.surface, (0, 255, 0), p1, p3)

    def DrawCircle(self, center, radius, color, drawwidth=1):
        """
        Draw a wireframe circle given the center, radius, and color.
        """
        radius *= self.zoom
        if radius < 1:
            radius = 1
        else: radius = int(radius)

        pygame.draw.circle(self.surface, color.bytes, center, radius, drawwidth)

    def DrawSolidCircle(self, center, radius, axis, color):
        """
        Draw a solid circle given the center, radius, and color.
        """
        radius *= self.zoom
        if radius < 1:
            radius = 1
        else: radius = int(radius)

        pygame.draw.circle(self.surface, (color/2).bytes+[127],
                           center, radius, 0)
        pygame.draw.circle(self.surface, color.bytes, center, radius, 1)
        pygame.draw.aaline(self.surface, (255, 0, 0), center,
                           (center[0] - radius*axis[0], center[1] +
                            radius*axis[1]))

    def DrawPolygon(self, vertices, color):
        """
        Draw a wireframe polygon given the screen vertices with the given color.
        """
        if not vertices:
            return

        if len(vertices) == 2:
            pygame.draw.aaline(self.surface, color.bytes, vertices[0], vertices)
        else:
            pygame.draw.polygon(self.surface, color.bytes, vertices, 1)

    def DrawSolidPolygon(self, vertices, color):
        """
        Draw a filled polygon given the screen vertices with the given color.
        """
        if not vertices:
            return

        if len(vertices) == 2:
            pygame.draw.aaline(self.surface, color.bytes, vertices[0],
                               vertices[1])
        else:
            pygame.draw.polygon(self.surface, (color/2).bytes+[127],
                                vertices, 0)
            pygame.draw.polygon(self.surface, color.bytes, vertices, 1)

class PygameFramework(framework.FrameworkBase):
    """
    This class is the framework for running the simulation
    """

    def __reset(self):
        # Screen/rendering-related
        self._viewZoom = 10.0
        self._viewCenter = None
        self._viewOffset = None
        self.screenSize = None
        self.rMouseDown = False
        self.textLine = 30
        self.font = None
        self.fps = 0

        # GUI-related (PGU)
        self.gui_app = None
        self.gui_table = None

    def __init__(self):
        super(PygameFramework, self).__init__()

        self.__reset()
        print('Initializing pygame framework...')
        # Pygame Initialization
        pygame.init()
        caption = "Python Box2D Testbed - " + self.name
        pygame.display.set_caption(caption)

        self.screen = pygame.display.set_mode((640, 480))
        self.screenSize = b2.b2Vec2(*self.screen.get_size())

        self.renderer = PygameDraw(surface=self.screen, test=self)
        self.world.renderer = self.renderer


        self.font = pygame.font.Font(None, 15)

        self.viewCenter = (0, 20.0)
        self.groundbody = self.world.CreateBody()

    def setCenter(self, value):
        """
        Updates the view offset based on the center of the screen.
        """
        self._viewCenter = b2.b2Vec2(*value)
        self._viewCenter *= self._viewZoom
        self._viewOffset = self._viewCenter - self.screenSize/2

    def setZoom(self, zoom):
        """
        Tells the display the zoom.
        """
        self._viewZoom = zoom

    viewZoom = property(lambda self: self._viewZoom, setZoom,
                        doc='Zoom factor for the display')
    viewCenter = property(lambda self: self._viewCenter/self._viewZoom,
                          setCenter, doc='Screen center in camera coordinates')
    viewOffset = property(lambda self: self._viewOffset,
                          doc='Offset of the top-left corner of the screen')


    def run(self):
        """
        Begins the draw loopn and tells the GUI to paint itself.
        """

        # If any of the test constructors update the settings, reflect
        # those changes on the GUI before running
        if GUIEnabled:
            self.gui_table.updateGUI(self.settings)
        self.clock = pygame.time.Clock()
        self.screen.fill((0, 0, 0))

            # Run the simulation loop
        self.SimulationLoop([0, 0, 0])

        if GUIEnabled and self.settings.drawMenu:
            self.gui_app.paint(self.screen)

        pygame.display.flip()
        self.clock.tick(self.settings.hz)
        self.fps = self.clock.get_fps()

    def run_next(self, action):
        """
        Updates the screen and tells the GUI to paint itself.
        """
        self.screen.fill((0, 0, 0))

        # Run the simulation loop
        self.SimulationLoop(action)
        if GUIEnabled and self.settings.drawMenu:
            self.gui_app.paint(self.screen)

        pygame.display.flip()
        self.clock.tick(self.settings.hz)
        self.fps = self.clock.get_fps()



    def Step(self, settings):
        """
        Updates the simulation
        """
        if GUIEnabled:
            self.gui_table.updateSettings(self.settings)

        super(PygameFramework, self).Step(settings)

        if GUIEnabled:
            self.gui_table.updateGUI(self.settings)

    def ConvertScreenToWorld(self, x, y):
        """
        Converts the display screen to the simulation's coordinates.
        """
        return b2.b2Vec2((x + self.viewOffset.x) / self.viewZoom,
                         ((self.screenSize.y - y + self.viewOffset.y)
                          / self.viewZoom))

    def DrawStringAt(self, x, y, s, color=(229, 153, 153, 255)):
        """
        Draw some text, str, at screen coordinates (x, y).
        """
        self.screen.blit(self.font.render(s, True, color), (x, y))

    def Print(self, s, color=(229, 153, 153, 255)):
        """
        Draw some text at the top status lines
        and advance to the next line.
        """
        self.screen.blit(self.font.render(s, True, color), (5, self.textLine))
        self.textLine += 15

