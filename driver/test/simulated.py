# -*- coding: utf-8 -*-
'''
Created on 20 Apr 2012

@author: Éric Piel

Copyright © 2012 Éric Piel, Delmic

This file is part of Open Delmic Microscope Software.

Delmic Acquisition Software is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version.

Delmic Acquisition Software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with Delmic Acquisition Software. If not, see http://www.gnu.org/licenses/.
'''
from driver import simulated
import model
import unittest

class ActuatorTest(object):
    """
    This abstract class should be able to test any type of actuator
    """
    # inheriting class should provide:
    # actuator_type (object): class of the actuator
    # actuator_args (tuple): argument to instantiate an actuator

    def test_scan(self):
        """
        Check that we can do a scan network. It can pass only if we are
        connected to at least one controller.
        """
        if not hasattr(self.actuator_type, "scan"):
            # nothing to test
            return
        devices = self.actuator_type.scan()
        self.assertGreater(len(devices), 0)
        
        for name, kwargs in devices:
            print "opening ", name
            dev = self.actuator_type(name, "actuator", None, **kwargs)
            self.assertTrue(dev.selfTest(), "Actuator self test failed.")

    def test_selfTest(self):
        dev = self.actuator_type(*self.actuator_args)
        self.assertTrue(dev.selfTest(), "Actuator self test failed.")

    def test_metadata(self):
        dev = self.actuator_type(*self.actuator_args)
        md = dev.getMetadata()
        self.assertIsInstance(md, dict, "Metadata should be a dict")
        if hasattr(dev, "position"):
            self.assertGreaterEqual(len(md), 1, "Metadata doesn't contain position information")
    
    def test_simple(self):
        dev = self.actuator_type(*self.actuator_args)
        self.assertGreaterEqual(len(dev.axes), 1, "Actuator has no axis")
        self.assertIsInstance(dev.ranges, dict, "range is not a dict")
        self.assertIsInstance(dev.speed, model.VigilantAttribute, "range is not a VigilantAttribute")
        self.assertIsInstance(dev.speed.value, dict, "speed value is not a dict")
        
    def test_moveAbs(self):
        # It's optional
        if not hasattr(self.actuator_type, "moveAbs"):
            # nothing to test
            return
        dev = self.actuator_type(*self.actuator_args)
        move = {}
        # move to the centre
        for axis in dev.axes:
            move[axis] = (dev.ranges[axis][0] + dev.ranges[axis][1]) / 2
        f = dev.moveAbs(move)
        f.result() # wait
        self.assertDictEqual(move, dev.position, "Actuator didn't move to the requested position")

    def test_moveRel(self):
        dev = self.actuator_type(*self.actuator_args)

        if hasattr(dev, "position"):
            prev_pos = dev.position
            
        move = {}
        # move by 1%
        for axis in dev.axes:
            move[axis] = dev.ranges[axis][1] * 0.01
            
        if hasattr(dev, "position"):
            expected_pos = {}
            for axis in dev.axes:
                expected_pos[axis] = prev_pos[axis] + move[axis]
            
        f = dev.moveRel(move)
        f.result() # wait
        if hasattr(dev, "position"):
            self.assertDictEqual(expected_pos, dev.position, "Actuator didn't move to the requested position")

    def test_stop(self):
        dev = self.actuator_type(*self.actuator_args)
        dev.stop()
        for axis in dev.axes:
            dev.stop(axis)
            
        move = {}
        for axis in dev.axes:
            move[axis] = dev.ranges[axis][1] * 0.01
        f = dev.moveRel(move)
        dev.stop()
        # TODO use the time of a long move to see if it took less 
        
    def test_speed(self):
        # For moves big enough, a 1m/s move should take approximately 100 times less time
        # than a 0.01m/s move 
        expected_ratio = 100
        delta_ratio = 2 # no unit 
        
        # TODO
#        # fast move
#        dev = self.actuator_type(*self.actuator_args)
#        stage.speed.value = {"x":1, "y":1}
#        move = {'x':100e-6, 'y':100e-6}
#        start = time.time()
#        stage.moveRel(move)
#        stage.waitStop()
#        dur_fast = time.time() - start
#        
#        stage.speed.value = {"x":1.0/expected_ratio, "y":1.0/expected_ratio}
#        move = {'x':-100e-6, 'y':-100e-6}
#        start = time.time()
#        stage.moveRel(move)
#        stage.waitStop()
#        dur_slow = time.time() - start
#        
#        ratio = dur_slow / dur_fast
#        print "ratio of", ratio 
#        if ratio < expected_ratio / 2 or ratio > expected_ratio * 2:
#            self.fail("Speed not consistent: ratio of " + str(ratio) + 
#                         "instead of " + str(expected_ratio) + ".")
        
class Stage2DTest(unittest.TestCase, ActuatorTest):
    
    actuator_type = simulated.Stage2D
    # name, role, children (must be None)
    actuator_args = ("stage", "test", {"x", "y"}, None)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()