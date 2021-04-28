#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19 Apr 2021

@author: Philip Winkler, Thera Pals, Sabrina Rossberger

Copyright © 2021 Philip Winkler, Delmic

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

import threading
from collections import OrderedDict
import collections
from concurrent.futures import CancelledError
import logging

import time
import copy

from odemis.util import TimeoutError
from odemis import model


class FastEMROA(object):
    """ Representation of a FastEM ROA (region of acquisition). """

    def __init__(self, name, coordinates, roc):
        """
        :param name: (str) name of the ROA
        :param coordinates: (float, float, float, float) l, t, r, b coordinates in m
        :param roc: (FastEMROC) corresponding region of calibration
        """
        self.name = model.StringVA(name)
        self.coordinates = model.TupleContinuous(coordinates,
                                                 range=((-1, -1, -1, -1), (1, 1, 1, 1)),
                                                 cls=(int, float),
                                                 unit='m')
        self.roc = model.VigilantAttribute(roc)

# TODO @Philip How do we check if the ROC is already acquired or not?
# TODO How is the same ROC assigned to many ROAs?
class FastEMROC(object):
    """ Representation of a FastEM ROC (region of calibration). """

    def __init__(self, name, coordinates):
        """
        :param name: (str) name of the ROC
        :param coordinates: (float, float, float, float) l, t, r, b coordinates in m
        """
        self.name = model.StringVA(name)
        self.coordinates = model.TupleContinuous(coordinates,
                                                 range=((-1, -1, -1, -1), (1, 1, 1, 1)),
                                                 cls=(int, float),
                                                 unit='m')
        self.parameters = None  # calibration object with all relevant parameters # TODO @Philip what is this for?


# The executor is a single object, independent of how many times the module (fastem.py) is loaded.
_executor = model.CancellableThreadPoolExecutor(max_workers=1)


def acquire(scanner, descanner, detector, roa, path):  # TODO scanner or better pass directly the dwell time?
    """
    Start an acquisition task for a given detector.

    Note: It is highly recommended to not have any other acquisition going on. TODO? @Eric still true?

    TODO add scanner, descanner
    :param detector: The detector object to be used for collecting the image data.
    :param roa: (FastEMROA) The acquisition region to be acquired (megafield).
    :param path: (str) The path and filename where the acquisition will be stored on the server (external storage).
    @ TODO Philip really also filename - directory name = project name?
    :return: (ProgressiveFuture) Acquisition future object, which can be cancelled. The result of the future is a tuple
             that contains:
             (model.DataArray): the raw acquisition data
             (Exception or None): exception raised during the acquisition @TODO check if true

    """
    # TODO should this be set by the GUI @Philip @Eric in the loop over the projects maybe?
    detector.dataContent.value = "empty"

    # TODO: pass path through attribute on ROA instead of second argument?
    # create a future
    f = model.ProgressiveFuture()

    # create a task that acquires the megafield image
    task = AcquisitionTask(scanner, descanner, detector, roa, path, f)

    f.task_canceller = task.cancel  # let the future cancel the task

    # connect the future to the task and run it in a thread @TODO @Eric does this comment make sense?
    # when f.result() is called, the task.run is executed @TODO @Eric correct?
    _executor.submitf(f, task.run)

    # return the future object
    return f


# TODO how does the GUI know the total time for all ROAs when it can only create one acquisition task per time?
# TODO if we would move this function to the AcquistionTask class how will the GUI be able to access the progress?
def estimateROATime(scanner, detector, num_fields):
    """
    Computes the approximate time it will take to run the roa (megafield) acquisition.
    TODO add scanner
    :param detector: (model.detector) The detector object to be used for collecting the image data.
    :param num_fields: (int) The number of single fields needed to cover the full ROA (megafield).
    :return (0 <= float): The estimated time for the roa (megafield) acquisition in s.
    """
    # TODO add descanner.physicalFlybackTime
    # number of pixels in an overscanned cell image
    num_pixels_cell = detector.cellCompleteResolution.value[0] * detector.cellCompleteResolution.value[1]
    # dwell time per pixel * size of an overscanned cell image * number of field images in megafield
    tot_time = scanner.dwell_time.value * num_fields * num_pixels_cell

    return tot_time


class AcquisitionTask(object):
    """
    The acquisition task for a single region of acquisition (ROA, megafield). An ROA consists of multiple single field
    images.
    """
    def __init__(self, scanner, descanner, detector, roa, path, future):
        self._scanner = scanner
        self._descanner = descanner
        self._detector = detector
        self._roa = roa  # region of acquisition object
        self._roa_coordinates = roa.coordinates.value  # TODO pass VA or value of VA?
        self._roa_name = roa.name.value
        self._roc = roa.roc.value  # region of calibration object
        self._roc_coordinates = self._roc.coordinates.value
        self._roc_name = self._roc.name.value
        self._path = path
        self._future = future

        # calculate how many field images needed to cover the full roa
        # TODO calculate number fields or check out BAssims code for tiledacq
        # TODO calculate number of single fields/field indicies to be acquired here already instead of in acquire function?
        self._field_indices = None  # TODO calculate ((0,0), (0,1), (1,0),....) nested tuple of field indices
        # TODO pass self._roa_coordinates

        # get the estimated time for the roa
        self.total_roa_time = estimateROATime(self._scanner, self._detector, len(self._field_indices))

        # the list of field image indices that still need to be acquired {(0,0), (0,1), (1,0), ...}
        self._fields_left = set(self._field_indices)  # just for progress update  # TODO better naming "left"

        # keep track if future was cancelled or not
        self._cancelled = False

        # threading event, which keeps track of when image data has been received from the detector
        self._data_received = threading.Event()

    def run(self):
        """
        Runs the acquisition of one ROA (megafield).
        :returns:
            (list of DataArrays): A list of the raw image data. Each data array (entire field, thumbnail, or zero array)
            represents one single field image within the roa (megafield).
            (Exception or None): Exception raised during the acquisition. If some single field image data has already
             been acquired, exceptions are not raised, but returned.
        :raise:
            Exception: If it failed before any single field images were acquired or if acquisition was cancelled.
        """

        exceptions = None

        # No need to set the start time of the future: it's automatically done when setting its state to running.
        self._future.set_progress(end=time.time() + self.total_roa_time)
        logging.info("Starting acquisition of mega field, with expected duration of %f s", self.total_roa_time)

        try:

            dataflow = self._detector.data
            dataflow.subscribe(self.image_received)

            # acquire the single field images
            megafield = self.acquire_roa(dataflow)

            dataflow.unsubscribe(self.image_received)

            # In case the acquisition was cancelled, before the future returned (f.result), raise cancellation error
            # after unsubscribing from the dataflow
            if self._cancelled:
                self._future.cancel()
                raise CancelledError()

        except CancelledError:
            raise

        except Exception as ex:
            # If no acquisition yet => just raise the exception.
            # If image data was already acquired, just log a warning.
            # TODO this seems not to be relevant for us, as already acquired data is offloaded
            # check if any field images have already been acquired
            if len(self._fields_left) == len(self._field_indices):
                raise
            logging.warning("Exception during roa acquisition (after some data has been already acquired).",
                            exc_info=True)
            exceptions = ex

        finally:
            # Don't hold references to the megafield once the acquisition is finished
            self._field_indices = None
            self._fields_left.clear()

        return megafield, exceptions

    def acquire_roa(self, dataflow):
        """
        Acquires the single field images resembling the roa (megafield).
        :return: (list of DataArrays): A list of the raw image data. Each data array (entire field, thumbnail,
                                       or zero array) represents one single field image within the roa (megafield).
        """
        # dictionary containing the single field images with index as key: e.g. {(0,1): DataArray}
        megafield = {}
        self.total_field_time = None  # TODO mppc.cellCompleteResolution * scanner.dwell_time + descanner.physicalFlybackTime
        timeout = self.total_field_time  # TODO + margin?

        # acquire all single field images
        # (are also automatically offload to the external storage)
        for field_idx in self._field_indices:

            # reset the event that waits for the image being received (puts flag to false)
            self._data_received.clear()

            # TODO move stage!

            # acquire the next field image
            dataflow.next(field_idx)

            field_image = None  # TODO @Eric how do we get the image data (in the callback? listener?)
            # update the dictionary with the new field image received
            megafield[field_idx] = field_image

            self._fields_left.discard(field_idx)

            # wait until single field image data was received (flag is set true if image_received was called)
            if not self._data_received.wait(timeout):
                raise TimeoutError  # TODO put message

            # cancel the acquisition and return to the run function and handle the cancellation there
            if self._cancelled:  # TODO canceler sends fake event (data received) @Eric what was meant here?
                return

            # update the time left for the acquisition
            expected_time = len(self._fields_left) * self.total_field_time
            self._future.set_progress(end=time.time() + expected_time)

        return megafield

    def image_received(self, dataflow, data):  # TODO what to do with the df and da? Needed?
        """
        Function called by dataflow when data has been received from the detector.
        :param datafow: (model.DataFlow) The dataflow on the detector.
        :param data: (model.DataArray) The data array containing the image data.
        """

        # when data was received notify the threading event, which keeps track of whether data was received
        self._data_received.set()

    def calculate_fields(self):
        """
        Calculates the number of single field images needed to cover the RAO (region of acquisition. Determines the
        corresponding indices of the field images in the matrix covering the ROA. If the ROA cannot be resembled
        by an integer number of single field images, the number of single field images is increased to cover the
        full region. An ROA is a rectangle.
        :return: (nested tuple (row, col)) The tuples need to be ordered so that the single field images resembling the
                 ROA are acquired first rows then columns.
        TODO verify the order of row/col
        """
        self._roa
        field_indices = None  # TODO ((0,0), (0,1), (0,2), ...., (1,0), ...)

        return field_indices  # TODO

    def cancel(self, f):
        """
        Cancels the ROA acquisition.
        :param f: (future) The ROA (megafield) future.
        :return: (boolean) True if cancelled, False if too late to cancel as future is already finished.
        """
        # put the cancel flag
        self._cancelled = True

        # Report if it's too late for cancellation (and the f.result() will return)
        if not self._fields_left:
            return False

        return True

