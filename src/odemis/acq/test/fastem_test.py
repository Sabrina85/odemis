# -*- coding: utf-8 -*-
"""
Created on 10th February 2021

@author: Sabrina Rossberger, Thera Pals

Copyright © 2021 Sabrina Rossberger, Delmic

This file is part of Odemis.

Odemis is free software: you can redistribute it and/or modify it under the terms
of the GNU General Public License version 2 as published by the Free Software
Foundation.

Odemis is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
Odemis. If not, see http://www.gnu.org/licenses/.
"""

from __future__ import division

from concurrent.futures._base import CancelledError
import logging
import numpy
from odemis import model
import odemis
from odemis.acq import acqmng
from odemis.util import test
import os
import time
import unittest
from unittest.case import skip

from odemis.acq.leech import ProbeCurrentAcquirer
import odemis.acq.path as path
import odemis.acq.stream as stream
from odemis.acq.acqmng import SettingsObserver

logging.getLogger().setLevel(logging.DEBUG)

CONFIG_PATH = os.path.dirname(odemis.__file__) + "/../../install/linux/usr/share/odemis/"
FASTEM_CONFIG = CONFIG_PATH + "sim/FASTEM-sim.odm.yaml"  # TODO


# class Fake0DDetector(model.Detector):
#     """
#     Imitates a probe current detector, but you need to send the data yourself (using
#     comp.data.notify(d)
#     """
#     def __init__(self, name):
#         model.Detector.__init__(self, name, "fakedet", parent=None)
#         self.data = Fake0DDataFlow()
#         self._shape = (float("inf"),)
#
#
# class Fake0DDataFlow(model.DataFlow):
#     """
#     Mock object just sufficient for the ProbeCurrentAcquirer
#     """
#     def get(self):
#         da = model.DataArray([1e-12], {model.MD_ACQ_DATE: time.time()})
#         return da


# @skip("simple")
class FASTEMTestCase(unittest.TestCase):
    # We don't need the whole GUI, but still a working backend is nice

    # backend_was_running = False

    @classmethod
    def setUpClass(cls):
        # try:
        #     test.start_backend(SECOM_CONFIG)
        # except LookupError:
        #     logging.info("A running backend is already found, skipping tests")
        #     cls.backend_was_running = True
        #     return
        # except IOError as exp:
        #     logging.error(str(exp))
        #     raise

        # create some streams connected to the backend
        cls.microscope = model.getMicroscope()
        cls.ccd = model.getComponent(role="diagnostic-ccd")
        cls.ebeamscanner = model.getComponent(role="e-beam")

        # TODO stage

        cls.asm = model.getComponent(role="asm")
        cls.mppc = model.getComponent(role="mppc")
        cls.multibeamscanner = model.getComponent(role="multibeam")
        cls.descanner = model.getComponent(role="descanner")

        # TODO set VAs to "good" values

    @classmethod
    def tearDownClass(cls):
        pass
        # if cls.backend_was_running:
        #     return
        # test.stop_backend()

    def setUp(self):
        pass
        # if self.backend_was_running:
        #     self.skipTest("Running backend found")

    def test_acquire_singleField(self):
        """Acquire a single field image."""

        f = acqmng.acquire(self.mppc)
        data, e = f.result()
        self.assertIsInstance(data[0], model.DataArray)
        self.assertIsNone(e)

        thumb = acqmng.computeThumbnail(st, f)
        self.assertIsInstance(thumb, model.DataArray)

        # let's do it a second time, "just for fun"
        f = acqmng.acquire(st.getProjections())
        data, e = f.result()
        self.assertIsInstance(data[0], model.DataArray)
        self.assertIsNone(e)

        thumb = acqmng.computeThumbnail(st, f)
        self.assertIsInstance(thumb, model.DataArray)

    def test_metadata(self):
        """
        Check if extra metadata are saved
        """
        settings_obs = SettingsObserver(model.getComponents())
        self.ccd.binning.value = (1, 1)  # make sure we don't save the right metadata by accident
        detvas = {'exposureTime', 'binning', 'gain'}
        s1 = stream.FluoStream("fluo2", self.ccd, self.ccd.data,
                               self.light, self.light_filter, detvas=detvas)
        s2 = stream.BrightfieldStream("bf", self.ccd, self.ccd.data, self.light, detvas=detvas)

        # Set different binning values for each stream
        s1.detBinning.value = (2, 2)
        s2.detBinning.value = (4, 4)
        st = stream.StreamTree(streams=[s1, s2])
        f = acqmng.acquire(st.getProjections(), settings_obs=settings_obs)
        data, e = f.result()
        for s in data:
            self.assertTrue(model.MD_EXTRA_SETTINGS in s.metadata, "Stream %s didn't save extra metadata." % s)
        self.assertEqual(data[0].metadata[model.MD_EXTRA_SETTINGS][self.ccd.name]['binning'], [(2, 2), 'px'])
        self.assertEqual(data[1].metadata[model.MD_EXTRA_SETTINGS][self.ccd.name]['binning'], [(4, 4), 'px'])

    def test_progress(self):
        """
        Check we get some progress updates
        """
        # create a little complex streamTree
        st = stream.StreamTree(streams=[
            self.streams[0],
            stream.StreamTree(streams=self.streams[1:3])
        ])
        self.start = None
        self.end = None
        self.updates = 0

        f = acqmng.acquire(st.getProjections())
        f.add_update_callback(self.on_progress_update)

        data, e = f.result()
        self.assertIsInstance(data[0], model.DataArray)
        self.assertIsNone(e)
        self.assertGreaterEqual(self.updates, 3)  # at least one update per stream

    def test_cancel(self):
        """
        try a bit the cancelling possibility
        """
        # create a little complex streamTree
        st = stream.StreamTree(streams=[
            self.streams[2],
            stream.StreamTree(streams=self.streams[0:2])
        ])
        self.start = None
        self.end = None
        self.updates = 0
        self.done = False

        f = acqmng.acquire(st.getProjections())
        f.add_update_callback(self.on_progress_update)
        f.add_done_callback(self.on_done)

        time.sleep(0.5)  # make sure it's started
        self.assertTrue(f.running())
        f.cancel()

        self.assertRaises(CancelledError, f.result, 1)
        self.assertGreaterEqual(self.updates, 1)  # at least one update at cancellation
        self.assertLessEqual(self.end, time.time())
        self.assertTrue(self.done)
        self.assertTrue(f.cancelled())

    def on_done(self, future):
        self.done = True

    def on_progress_update(self, future, start, end):
        self.start = start
        self.end = end
        self.updates += 1


if __name__ == "__main__":
    unittest.main()
