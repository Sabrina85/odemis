# -*- coding: utf-8 -*-
'''
Created on 29 Mar 2012

@author: Éric Piel

Copyright © 2012 Éric Piel, Delmic

This file is part of Odemis.

Odemis is free software: you can redistribute it and/or modify it under the terms 
of the GNU General Public License version 2 as published by the Free Software 
Foundation.

Odemis is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with 
Odemis. If not, see http://www.gnu.org/licenses/.
'''

# Provides various components which are actually not connected to a physical one.
# It's mostly for replacing components which are present but not controlled by
# software, or for testing.

from __future__ import division


import logging
from odemis import model
from odemis.model import isasync
from odemis.model._futures import CancellableThreadPoolExecutor
import random
import time


class Light(model.Emitter):
    """
    Simulated bright light component. Just pretends to be always on with wide
    spectrum emitted (white).
    """
    def __init__(self, name, role, **kwargs):
        model.Emitter.__init__(self, name, role, **kwargs)
        
        self._shape = ()
        self.power = model.FloatContinuous(0., [0., 100.], unit="W")
        self.power.subscribe(self._updatePower)
        # just one band: white
        # emissions is list of 0 <= floats <= 1. Always 1.0: cannot lower it.
        self.emissions = model.ListVA([1.0], unit="", setter=lambda x: [1.0])
        self.spectra = model.ListVA([(380e-9, 160e-9, 560e-9, 960e-9, 740e-9)],
                                     unit="m", readonly=True) # list of 5-tuples of floats
        
    def getMetadata(self):
        metadata = {}
        metadata[model.MD_IN_WL] = (380e-9, 740e-9)
        metadata[model.MD_LIGHT_POWER] = self.power.value
        return metadata
    
    def _updatePower(self, value):
        if value == 0:
            logging.info("Light is off")
        else:
            logging.info("Light is on")


class Stage(model.Actuator):
    """
    Simulated stage component. Just pretends to be able to move all around.
    """
    def __init__(self, name, role, axes, ranges=None, **kwargs):
        """
        axes (set of string): names of the axes
        """
        assert len(axes) > 0
        if ranges is None:
            ranges = {}

        axes_def = {}
        self._position = {}
        init_speed = {}
        for a in axes:
            rng = ranges.get(a, [-0.1, 0.1])
            axes_def[a] = model.Axis(unit="m", range=rng, speed=[0., 10.])
            # start at the centre
            self._position[a] = (rng[0] + rng[1]) / 2
            init_speed[a] = 10.0 # we are super fast!

        model.Actuator.__init__(self, name, role, axes=axes_def, **kwargs)

        # RO, as to modify it the client must use .moveRel() or .moveAbs()
        self.position = model.VigilantAttribute(
                                    self._applyInversionAbs(self._position),
                                    unit="m", readonly=True)
        
        self.speed = model.MultiSpeedVA(init_speed, [0., 10.], "m/s")
    
    def _updatePosition(self):
        """
        update the position VA
        """
        # it's read-only, so we change it via _value
        self.position._value = self._applyInversionAbs(self._position)
        self.position.notify(self.position.value)
        
    @isasync
    def moveRel(self, shift):
        shift = self._applyInversionRel(shift)
        maxtime = 0
        for axis, change in shift.items():
            if not axis in shift:
                raise ValueError("Axis '%s' doesn't exist." % str(axis))
            self._position[axis] += change
            if (self._position[axis] < self.axes[axis].range[0] or
                self._position[axis] > self.axes[axis].range[1]):
                logging.warning("moving axis %s to %f, outside of range %r", 
                                axis, self._position[axis], self.axes[axis].range)
            else: 
                logging.info("moving axis %s to %f", axis, self._position[axis])
            maxtime = max(maxtime, abs(change) / self.speed.value[axis])
        
        self._updatePosition()
        # TODO queue the move and pretend the position is changed only after the given time
        return model.InstantaneousFuture()
        
    @isasync
    def moveAbs(self, pos):
        pos = self._applyInversionAbs(pos)
        maxtime = 0
        for axis, new_pos in pos.items():
            if not axis in pos:
                raise ValueError("Axis '%s' doesn't exist." % str(axis))
            change = self._position[axis] - new_pos
            self._position[axis] = new_pos
            logging.info("moving axis %s to %f", axis, self._position[axis])
            maxtime = max(maxtime, abs(change) / self.speed.value[axis])
         
        # TODO queue the move
        self._updatePosition()
        return model.InstantaneousFuture()
    
    def stop(self, axes=None):
        logging.warning("Stopping all axes: %s", ", ".join(self.axes))


PRESSURE_VENTED = 100e3 # Pa
PRESSURE_PUMPED = 5e3 # Pa
class Chamber(model.Actuator):
    """
    Simulated chamber component. Just pretends to be able to change pressure
    """
    def __init__(self, name, role, **kwargs):
        """
        Initialises the component
        """
        # TODO: or just provide .targetPressure (like .targetTemperature) ?
        # Or maybe provide .targetPosition: position that would be reached if
        # all the requested move were instantly applied?
        # TODO: support multiple pressures (low vacuum, high vacuum)
        axes = {"pressure": model.Axis(unit="Pa", 
                                       choices={PRESSURE_VENTED: "vented",
                                                PRESSURE_PUMPED: "vacuum"})}
        model.Actuator.__init__(self, name, role, axes=axes, **kwargs)
        # For simulating moves
        self._position = PRESSURE_PUMPED # last official position
        self._goal = PRESSURE_PUMPED
        self._time_goal = 0 # time the goal was/will be reached
        self._time_start = 0 # time the move started
        
        # RO, as to modify it the client must use .moveRel() or .moveAbs()
        self.position = model.VigilantAttribute(
                                    {"pressure": self._position},
                                    unit="Pa", readonly=True)
        # Almost the same as position, but gives the current position
        self.pressure = model.VigilantAttribute(self._position,
                                    unit="Pa", readonly=True)

        # will take care of executing axis move asynchronously
        self._executor = CancellableThreadPoolExecutor(max_workers=1) # one task at a time

    def terminate(self):
        if self._executor:
            self.stop()
            self._executor.shutdown()
            self._executor = None

    def _updatePosition(self):
        """
        update the position VA and .pressure VA
        """
        # Compute the current pressure
        now = time.time()
        if self._time_goal < now: # done
            # goal ±5%
            pos = self._goal * random.uniform(0.95, 1.05)
        else:
            # TODO make it logarithmic
            ratio = (now - self._time_start) / (self._time_goal - self._time_start)
            pos = self._position + (self._goal - self._position) * ratio
        
        # it's read-only, so we change it via _value
        self.pressure._value = pos
        self.pressure.notify(pos)

        # .position contains the last known/valid position
        # it's read-only, so we change it via _value
        self.position._value = {"pressure": self._position}
        self.position.notify(self.position.value)

    @isasync
    def moveRel(self, shift):
        # TODO: have this the default implementation
        logging.warning("Shouldn't change pressure via .moveRel")

        # convert into an absolute move
        pos = {}
        for a, v in shift.items:
            pos[a] = self.position.value[a] + v

        return self.moveAbs(pos)

    @isasync
    def moveAbs(self, pos):
        if not pos:
            return model.InstantaneousFuture()

        # TODO: we need a common _checkMoveAbsInput()
        for axis, new_pos in pos.items():
            if axis in self.axes:
                if not new_pos in self.axes[axis].choices:
                    raise ValueError("Axis '%s' does allow position %s." % (axis, new_pos))
            else:
                raise ValueError("Axis '%s' doesn't exist." % (axis,))

        return self._executor.submit(self._changePressure, pos["pressure"])

    def _changePressure(self, p):
        """
        Synchronous change of the pressure
        p (float): target pressure
        """
        # TODO: allow to cancel during the change
        now = time.time()
        duration = 5 # s
        self._time_start = now
        self._time_goal = now + duration # s
        self._goal = p

        time.sleep(duration)
        self._position = p
        self._updatePosition()

    def stop(self, axes=None):
        self._executor.cancel()
        logging.warning("Stopped pressure change")

# vim:tabstop=4:shiftwidth=4:expandtab:spelllang=en_gb:spell:
