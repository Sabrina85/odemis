# -*- coding: utf-8 -*-
"""
Created on 10th February 2021

@author: Sabrina Rossberger

Copyright © 2021 Sabrina Rossberger, Delmic

This file is part of Odemis.

Odemis is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License version 2 as published by the Free
Software Foundation.

Odemis is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
Odemis. If not, see http://www.gnu.org/licenses/.

"""

# Everything related to high-level image acquisition on the FASTEM microscope.


from __future__ import division

from collections import OrderedDict
import collections
from concurrent.futures import CancelledError
import logging

from odemis import model
from odemis.acq import _futures
from odemis.acq.stream import FluoStream, SEMCCDMDStream, SEMMDStream, SEMTemporalMDStream, \
    OverlayStream, OpticalStream, EMStream, ScannedFluoStream, ScannedFluoMDStream, \
    ScannedRemoteTCStream, ScannedTCSettingsStream
from odemis.util import img, fluo, executeAsyncTask
import time
import copy
from odemis.model import prepare_to_listen_to_more_vas


# This is the "manager" of an acquisition. It gets the detector component.
# You are in charge of ensuring that no other acquisition is going on at the same time.
# The manager creates a separate thread to run the acquisition. It
# returns a special "ProgressiveFuture" which is a Future object that can be
# stopped while already running, and reports from time to time progress on its
# execution.

def acquire(mppc, roi, settings_obs=None):
    """ Start an acquisition task for a given detector (mppc).

    Note: It is highly recommended to not have any other acquisition going on. TODO?

    :param mppc: The detector object to be used for collecting the image data.
    :param settings_obs: [SettingsObserver or None] class that contains a list of all VAs
        that should be saved as metadata
    :return: (ProgressiveFuture) an object that represents the task, allow to
        know how much time before it is over and to cancel it. It also permits
        to receive the result of the task, which is a tuple:
            (list of model.DataArray): the raw acquisition data
            (Exception or None): exception raised during the acquisition
    """

    # create a future
    future = model.ProgressiveFuture()

    # create a task
    task = AcquisitionTask(mppc, future, settings_obs)
    future.task_canceller = task.cancel  # let the future cancel the task

    # connect the future to the task and run in a thread
    executeAsyncTask(future, task.run)

    # return the interface to manipulate the task
    return future


def estimateTime(mppc, asm):  # TODO needed?
    """
    Computes the approximate time it will take to run the acquisition.
    :param mppc: The detector object to be used for collecting the image data.
    :return (0 <= float): estimated time in s.
    """
    tot_time = dwell_time * len(field_images)

    return tot_time


class AcquisitionTask(object):
    """
    TODO
    """
    def __init__(self, mppc, roi, future, settings_obs=None):
        self._mppc = mppc
        self._roi = roi
        self._future = future
        self._settings_obs = settings_obs

        # get the estimated time
        total_time = estimateAcquisitionTime()

        self._fields_left = set(self._fields)  # just for progress update TODO use instead list of field images?
        self._current_field = None
        self._current_future = None
        self._cancelled = False

    def run(self):
        """
        Runs the acquisition
        :returns:
            (list of DataArrays): all the raw data acquired
            (Exception or None): exception raised during the acquisition
        :raise:
            Exception: if it failed before any result were acquired
        """
        # TODO put pseudo code here!
        exceptions = None
        assert(self._current_field is None)  # Task should be used only once
        expected_time = estimateAcquisitionTime()
        # no need to set the start time of the future: it's automatically done
        # when setting its state to running.
        self._future.set_progress(end=time.time() + expected_time)

        logging.info("Starting acquisition of mega field, with expected duration of %f s", expected_time)

        # Keep order so that the DataArrays (single field images) are returned in the order they were
        # acquired. Not absolutely needed, but nice for the user in some cases.
        single_field_images = OrderedDict()  # stream -> list of raw images  TODO (x,y) -> single field image?? or projects -> rois?
        try:
            if not self._settings_obs:
                logging.warning("Acquisition task has no SettingsObserver, not saving extra "
                                "metadata.")
            # for field in self._field_images:  # TODO do ROI or project loop here?
            # Get the future of the acquisition
            f = acquire()

            # self._current_future = f
            # self._current_field = field
            # self._fields_left.discard(field)

            # in case acquisition was cancelled, before the future was set
            if self._cancelled:
                f.cancel()
                raise CancelledError()

            # If it's a ProgressiveFuture, listen to the time update TODO always progressive future?
            # try:
            f.add_update_callback(self._on_progress_update)
            # except AttributeError:
            #     pass  # not a ProgressiveFuture, fine

            # Wait for the acquisition to be finished.
            # Will pass down exceptions, included in case it's cancelled
            das = f.result()
            if not isinstance(das, collections.Iterable):
                logging.warning("Future of %s didn't return a list of DataArrays, but %s", field, das)
                das = []

            # Add extra settings to metadata
            if self._settings_obs:
                settings = self._settings_obs.get_all_settings()
                for da in das:
                    da.metadata[model.MD_EXTRA_SETTINGS] = copy.deepcopy(settings)

            # single_field_images[field] = das

            # update the time left
            # expected_time -= self._fieldTimes[field]
            # self._future.set_progress(end=time.time() + expected_time)  # TODO if we update the time here, we also need the loop over the field images here?

        except CancelledError:
            raise
        except Exception as ex:
            # If no acquisition yet => just raise the exception,
            # otherwise, the results already acquired might already be useful
            if not field_images:
                raise
            logging.warning("Exception during acquisition (after some data already acquired)", exc_info=True)
            exceptions = ex
        finally:
            pass
            # Don't hold references to the field once it's over
            # self._field = []
            # self._fields_left.clear()
            # self._fieldTimes = {}
            # self._current_field = None
            # self._current_future = None

        # Update metadata using OverlayStream (if there was one)
        # self._adjust_metadata(raw_images)  TODO maybe useful for adding MD

        # merge all the raw data (= list of DataArrays) into one long list
        # mega_field = sum(single_field_images.values(), [])
        mega_field = das

        return mega_field, exceptions

    def acquire(self):

        # calculate how many field images needed to cover full roi
        field_indices = calculate_field_indices()
    
        megafield = {}
        
        self._mppc.dataContent.value = "empty"  # TODO move

        dataflow = self._mppc.data
        dataflow.subscribe(self.image_received)

        # acquire all single field images and offload them to the external storage
        for field_idx in field_indices:
            dataflow.next(field_idx)

        dataflow.unsubscribe(self.image_received)

        return megafield

    def image_received(self):  # TODO what to put here? or where? usually listener would be the live stream
        pass

    def calculate_fields(self):
        self._roi
        return [(0, 0), (0, 1)]

    def _adjust_metadata(self, raw_data):
        """
        Update/adjust the metadata of the raw data received based on global
        information.
        raw_data (dict Stream -> list of DataArray): the raw data for each stream.
          The raw data is directly updated, and even removed if necessary.
        """
        # Update the pos/pxs/rot metadata from the fine overlay measure.
        # The correction metadata is in the metadata of the only raw data of
        # the OverlayStream.
        opt_cor_md = None
        sem_cor_md = None
        for s, data in list(raw_data.items()):
            if isinstance(s, OverlayStream):
                if opt_cor_md or sem_cor_md:
                    logging.warning("Multiple OverlayStreams found")
                opt_cor_md = data[0].metadata
                sem_cor_md = data[1].metadata
                del raw_data[s] # remove the stream from final raw data

        # Even if no overlay stream was present, it's worthy to update the
        # metadata as it might contain correction metadata from basic alignment.
        for s, data in raw_data.items():
            if isinstance(s, OpticalStream):
                for d in data:
                    img.mergeMetadata(d.metadata, opt_cor_md)
            elif isinstance(s, EMStream):
                for d in data:
                    img.mergeMetadata(d.metadata, sem_cor_md)

        # add the stream name to the image if nothing yet
        for s, data in raw_data.items():
            for d in data:
                if model.MD_DESCRIPTION not in d.metadata:
                    d.metadata[model.MD_DESCRIPTION] = s.name.value

    def _on_progress_update(self, f, start, end):
        """
        Called when the current future has made a progress (and so it should
        provide a better time estimation).
        """
        # If the acquisition is cancelled or failed, we might receive updates
        # from the sub-future a little after. Let's not make a fuss about it.
        if self._future.done():
            return

        # There is a tiny chance that self._current_future is already set to
        # None, but the future isn't officially ended yet. Also fine.
        if self._current_future != f and self._current_future is not None:
            logging.warning("Progress update not from the current future: %s instead of %s",
                            f, self._current_future)
            return

        total_end = end + sum(self._streamTimes[s] for s in self._streams_left)
        self._future.set_progress(end=total_end)

    def cancel(self, future):
        """
        cancel the acquisition
        """
        # put the cancel flag
        self._cancelled = True

        if self._current_future is not None:
            cancelled = self._current_future.cancel()
        else:
            cancelled = False

        # Report it's too late for cancellation (and so result will come)
        if not cancelled and not self._streams_left:
            return False

        return True


HIDDEN_VAS = ['children', 'dependencies', 'affects', 'alive', 'state', 'ghosts']


class SettingsObserver(object):
    """
    Class that listens to all settings, so they can be easily stored as metadata
    at the end of an acquisition.
    """

    def __init__(self, components):
        """
        components (set of HwComponents): component which should be observed
        """
        self._all_settings = {}
        self._components = components  # keep a reference to the components, so they are not garbage collected
        self._va_updaters = []  # keep a reference to the subscribers so they are not garbage collected

        for comp in components:
            self._all_settings[comp.name] = {}
            vas = model.getVAs(comp).items()
            prepare_to_listen_to_more_vas(len(vas))

            for va_name, va in vas:
                if va_name in HIDDEN_VAS:
                    continue
                # Store current value of VA (calling .value might take some time)
                self._all_settings[comp.name][va_name] = [va.value, va.unit]
                # Subscribe to VA, update dictionary on callback
                def update_settings(value, comp_name=comp.name, va_name=va_name):
                    self._all_settings[comp_name][va_name][0] = value
                self._va_updaters.append(update_settings)
                va.subscribe(update_settings)

    def get_all_settings(self):
        return copy.deepcopy(self._all_settings)
