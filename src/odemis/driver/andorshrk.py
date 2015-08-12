# -*- coding: utf-8 -*-
'''
Created on 17 Feb 2014

@author: Éric Piel

Copyright © 2014-2015 Éric Piel, Delmic

This file is part of Odemis.

Odemis is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License version 2 as published by the Free Software Foundation.

Odemis is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with Odemis. If not, see http://www.gnu.org/licenses/.
'''
from __future__ import division

from ctypes import *
import ctypes
import logging
import math
from odemis import model
import odemis
from odemis.driver import andorcam2
from odemis.model import isasync, CancellableThreadPoolExecutor, HwError
from odemis.util import driver
import os
import time


# Constants from ShamrockCIF.h
ACCESSORYMIN = 0  # changed in the latest version (from 1->2)
ACCESSORYMAX = 1
FILTERMIN = 1
FILTERMAX = 6
# SHAMROCK_TURRETMIN 1
# SHAMROCK_TURRETMAX 3
GRATINGMIN = 1
# Note: the documentation mentions the width is in mm, but it's actually µm.
SLITWIDTHMIN = 10
SLITWIDTHMAX = 2500
# SHAMROCK_I24SLITWIDTHMAX 24000
SHUTTERMODEMIN = 0
SHUTTERMODEMAX = 1
# SHAMROCK_DET_OFFSET_MIN -240000
# SHAMROCK_DET_OFFSET_MAX 240000
# SHAMROCK_GRAT_OFFSET_MIN -20000
# SHAMROCK_GRAT_OFFSET_MAX 20000

SLIT_INDEX_MIN = 1
SLIT_INDEX_MAX = 4

INPUT_SLIT_SIDE = 1
INPUT_SLIT_DIRECT = 2
OUTPUT_SLIT_SIDE = 3
OUTPUT_SLIT_DIRECT = 4

FLIPPER_INDEX_MIN = 1
FLIPPER_INDEX_MAX = 2
PORTMIN = 0
PORTMAX = 1

INPUT_FLIPPER = 1
OUTPUT_FLIPPER = 2
DIRECT_PORT = 0
SIDE_PORT = 1

ERRORLENGTH = 64


class ShamrockError(Exception):
    def __init__(self, errno, strerror):
        self.args = (errno, strerror)
        self.errno = errno
        self.strerror = strerror

    def __str__(self):
        return self.args[1]


class ShamrockDLL(CDLL):
    """
    Subclass of CDLL specific to Andor Shamrock library, which handles error
    codes for all the functions automatically.
    It works by setting a default _FuncPtr.errcheck.
    """

    def __init__(self):
        if os.name == "nt":
            # FIXME: might not fly if parent is not a WinDLL => use __new__()
            WinDLL.__init__(self, "libshamrockcif.dll") # TODO check it works
        else:
            # libandor.so must be loaded first. If there is a camera, that has
            # already been done, but if not, we need to do it here. It's not a
            # problem to do it multiple times.
            self._dllandor = CDLL("libandor.so.2", RTLD_GLOBAL)
            # Global so that its sub-libraries can access it
            CDLL.__init__(self, "libshamrockcif.so.2", RTLD_GLOBAL)

    def at_errcheck(self, result, func, args):
        """
        Analyse the return value of a call and raise an exception in case of
        error.
        Follows the ctypes.errcheck callback convention
        """
        # everything returns DRV_SUCCESS on correct usage, _except_ GetTemperature()
        if result not in ShamrockDLL.ok_code:
            errmsg = create_string_buffer(ERRORLENGTH)
            self.ShamrockGetFunctionReturnDescription(result, errmsg, len(errmsg))
            raise ShamrockError(result,
                                "Call to %s failed with unknown error %d: %s" %
                                (func.__name__, result, errmsg.value))
        return result

    def __getitem__(self, name):
        try:
            func = super(ShamrockDLL, self).__getitem__(name)
        except Exception:
            raise AttributeError("Failed to find %s" % (name,))
        func.__name__ = name
        func.errcheck = self.at_errcheck
        return func

    ok_code = {
20202: "SHAMROCK_SUCCESS",
}
    err_code = {
20201: "SHAMROCK_COMMUNICATION_ERROR",
20266: "SHAMROCK_P1INVALID",
20267: "SHAMROCK_P2INVALID",
20268: "SHAMROCK_P3INVALID",
20269: "SHAMROCK_P4INVALID",
20275: "SHAMROCK_NOT_INITIALIZED",
20292: "SHAMROCK_NOT_AVAILABLE",
}


class HwAccessMgr(object):
    """
    Context manager that ensures that the CCD is not doing any acquisition
    while within the context.
    """
    def __init__(self, ccd):
        """
        ccd (AndorCam2 or None)
        """
        self._ccd = ccd

    def __enter__(self):
        if self._ccd is None:
            logging.debug("Not taking CCD lock")
        else:
            self._ccd.request_hw.append(None)  # let the acquisition thread know it should release the lock
            logging.debug("Requesting access to CCD")
            self._ccd.hw_lock.acquire()

    def __exit__(self, exc_type, exc_value, traceback):
        """
        returns True if the exception is to be suppressed (never)
        """
        if self._ccd is None:
            logging.debug("CCD lock doesn't need to be released")
        else:
            self._ccd.request_hw.pop()  # hw no more needed
            logging.debug("Released CCD lock")
            self._ccd.hw_lock.release()


class LedActiveMgr(object):
    """
    Context manager that signal that the leds (might) be on. The signal
    is a _low_ level on the TTL accessory output. When the leds are for sure
    off, the TTL level is set to _high_.
    This typically happens when the slits move, and can cause damage to some
    detectors.
    """
    def __init__(self, spec, line):
        """
        spec (Shamrock): spectrograph component
        line (1<=int<=2 or None): Accessory line number or None if nothing
         needs to be done.
        """
        self._spec = spec
        self._line = line
        self.__exit__(None, None, None)  # start by indicating the leds are off

    def __enter__(self):
        if self._line is None:
            return
        logging.debug("Indicating leds are on")
        # Force the protection independently of the protection VA state
        self._spec.SetAccessory(self._line, False)
        time.sleep(1e-3)  # wait 1 ms to make sure all the detectors are stopped

    def __exit__(self, exc_type, exc_value, traceback):
        """
        returns True if the exception is to be suppressed (never)
        """
        if self._line is None:
            return
        logging.debug("Indicating leds are off")
        # Unprotect iff the protection VA also allows it
        if not self._spec.protection.value:
            self._spec.SetAccessory(self._line, True)


# default names for the slits
SLIT_NAMES = {INPUT_SLIT_SIDE: "slit-in-side",  # Note: previously it was called "slit"
              INPUT_SLIT_DIRECT: "slit-in-direct",
              OUTPUT_SLIT_SIDE: "slit-out-side",
              OUTPUT_SLIT_DIRECT: "slit-out-direct",
             }

# The two values exported by the Odemis API for the flipper positions
FLIPPER_TO_PORT = {0: DIRECT_PORT,
                   math.radians(90): SIDE_PORT}

class Shamrock(model.Actuator):
    """
    Component representing the spectrograph part of the Andor Shamrock
    spectrometers.
    On Linux, the SR303i is supported since SDK 2.97, and the other ones,
    including the SR193i since SDK 2.99.
    The SR303i must be connected via the I²C cable on the iDus. With SDK 2.100+,
    it also work via the direct USB connection.
    Note: we don't handle changing turret.
    """
    def __init__(self, name, role, device, camera=None, accessory=None,
                 slits=None, bands=None, fstepsize=1e-6, children=None, **kwargs):
        """
        device (0<=int or "fake"): device number
        camera (None or AndorCam2): Needed if the connection is done via the
          I²C connector of the camera. In such case, no children should be
          provided.
        children (dict str -> Components): "ccd" should be the CCD used to acquire
          the spectrum, if the connection is directly via USB.
        inverted (None): it is not allowed to invert the axes
        accessory (str or None): if "slitleds", then a TTL signal will be set to
          high on line 1 whenever one of the slit leds might be turned on.
        slits (None or dict int -> str): names of each slit,
          for 1 to 4: in-side, in-direct, out-side, out-direct
        bands (None or dict 1<=int<=6 -> 2-tuple of floats > 0 or str):
          wavelength range or name of each filter for the filter wheel from 1->6.
          Positions without filters do not need to be defined.
        fstepsize (0<float): size of one step on the focus actuator. Not very
          important, mostly useful for providing to the user a rough idea of how
          much the image will change after a move.
        """
        # TODO: allow to specify the device via its sn, instead of the (absolutely
        # unreliable) device number?

        # From the documentation:
        # If controlling the shamrock through I²C it is important that both the
        # camera and spectrograph are being controlled through the same calling
        # program and that the DLLs used are contained in the same working
        # folder. The camera MUST be initialized before attempting to
        # communicate with the Shamrock.
        if kwargs.get("inverted", None):
            raise ValueError("Axis of spectrograph cannot be inverted")

        if device == "fake":
            self._dll = FakeShamrockDLL(camera)
            device = 0
        else:
            self._dll = ShamrockDLL()
        self._device = device

        try:
            self._camera = children["ccd"]
        except (TypeError, KeyError):  # no "ccd" child => camera?
            if camera is None:
                raise ValueError("Spectrograph needs a child 'ccd' or a camera")
            self._camera = camera
            self._is_via_camera = True
            self._hw_access = HwAccessMgr(camera)
        else:
            self._is_via_camera = False
            self._hw_access = HwAccessMgr(None)

        slits = slits or {}
        for i in slits:
            if not SLIT_INDEX_MIN <= i <= SLIT_INDEX_MAX:
                raise ValueError("Slit number must be between 1 and 4, but got %s" % (i,))
        self._slit_names = SLIT_NAMES.copy()
        self._slit_names.update(slits)

        try:
            self.Initialize()
        except ShamrockError:
            raise HwError("Failed to find Andor Shamrock (%s) as device %d" %
                          (name, device))
        try:
            nd = self.GetNumberDevices()
            if device >= nd:
                raise HwError("Failed to find Andor Shamrock (%s) as device %d" %
                              (name, device))

            if accessory is not None and not self.AccessoryIsPresent():
                raise ValueError("Accessory set to '%s', but no accessory connected"
                                 % (accessory,))
            if accessory == "slitleds":
                # To control the ttl signal from outside the component
                self.protection = model.BooleanVA(True, setter=self._setProtection)
                self._setProtection(True)
                self._led_access = LedActiveMgr(self, 0)
            else:
                self._led_access = LedActiveMgr(None, None)

            # for now, it's fixed (and it's unlikely to be useful to allow less than the max)
            max_speed = 1000e-9 / 5 # about 1000 nm takes 5s => max speed in m/s
            self.speed = model.MultiSpeedVA({"wavelength": max_speed},
                                            range=(max_speed, max_speed),
                                            unit="m/s",
                                            readonly=True)

            gchoices = self._getGratingChoices()

            # The actual limits are per grating. We cannot provide this much
            # info via the .axes attribute, so just lowest and largest
            # wavelength reachable
            wl_range = (float("inf"), float("-inf"))
            for g in gchoices:
                try:
                    wmin, wmax = self.GetWavelengthLimits(g)
                except ShamrockError:
                    logging.exception("Failed to find wavelength limit for grating %d", g)
                    continue
                wl_range = min(wl_range[0], wmin), max(wl_range[1], wmax)

            axes = {"wavelength": model.Axis(unit="m", range=wl_range,
                                             speed=(max_speed, max_speed)),
                    "grating": model.Axis(choices=gchoices)
                    }

            if self.FocusMirrorIsPresent():
                # TODO: how to get a conversion from steps to meter? => param?
                if not 0 < fstepsize <= 0.1:  # m
                    raise ValueError("fstepsize is %f but should be between 0 and 0.1m" % (fstepsize,))
                self._focus_step_size = fstepsize
                mx = self.GetFocusMirrorMaxSteps() * fstepsize
                axes["focus"] = model.Axis(unit="m",
                                           range=(fstepsize, mx))
                logging.info("Focus actuator added as 'focus'")
            else:
                logging.info("Focus actuator is not present")

            if self.FilterIsPresent():
                if bands is None:  # User gave no info => fallback to what the hardware knows
                    # TODO: way to detect that a position has no filter?
                    bands = dict((i, self.GetFilterInfo(i + 1)) for i in range(FILTERMAX))
                else:  # Check the content
                    try:
                        for pos, band in bands.items():
                            if not FILTERMIN <= pos <= FILTERMAX:
                                raise ValueError("Filter position should be between %d and "
                                                 "%d, but got %d." % (FILTERMIN, FILTERMAX, pos))
                            # To support "weird" filter, we accept strings
                            if isinstance(band, basestring):
                                if not band.strip():
                                    raise ValueError("Name of filter %d is empty" % pos)
                            else:
                                driver.checkLightBand(band)
                    except Exception:
                        logging.exception("Failed to parse bands %s", bands)
                        raise
                # If the current position is not among the known positions =>
                # add this position
                b = self.GetFilter()
                if b not in bands:
                    bands[b] = self.GetFilterInfo(b)

                axes["band"] = model.Axis(choices=bands)
                logging.info("Filter wheel added as 'band'")
            else:
                if bands is not None:
                    raise ValueError("Device %d has no filter wheel, but 'bands'"
                                     " argument provided." % (bands,))
                logging.info("Filter wheel is not present")

            # add slits which are available
            for i, slitn in self._slit_names.items():
                if self.AutoSlitIsPresent(i):
                    axes[slitn] = model.Axis(unit="m",
                                             range=(SLITWIDTHMIN * 1e-6, SLITWIDTHMAX * 1e-6)
                                             )
                    logging.info("Slit %d added as %s", i, slitn)
                else:
                    logging.info("Slit %d (%s) is not present", i, slitn)

            # TODO: allow more clever shutter management? eg, only opened when acquiring?
            if self.ShutterIsPresent():
                self.SetShutter(1)  # 1 = open
                logging.info("Opening shutter (all the time)")
            else:
                logging.info("No shutter is present")

            # TODO: allow to define the name of the axis? or anyway, we can use
            # MultiplexActuator to rename the axis?
            if self.FlipperMirrorIsPresent(OUTPUT_FLIPPER):
                # The position values are arbitrary, but these are the one we
                # typically use in Odemis for switching between two positions
                axes["flip-out"] = model.Axis(unit="rad",
                                              choices=set(FLIPPER_TO_PORT.keys())
                                              )
                logging.info("Adding out mirror flipper as flip-out")
            else:
                logging.info("Out mirror flipper is not present")

            # provides a ._axes
            model.Actuator.__init__(self, name, role, axes=axes, **kwargs)

            # set HW and SW version
            self._swVersion = "%s" % (odemis.__version__)
            # TODO: EEPROM contains name of the device, but there doesn't seem to be any function for getting it?!
            sn = self.GetSerialNumber()
            fl, ad, ft = self.EepromGetOpticalParams()
            self._hwVersion = ("%s (s/n: %s, focal length: %d mm)" %
                               ("Andor Shamrock", sn, round(fl * 1000)))

            # will take care of executing axis move asynchronously
            self._executor = CancellableThreadPoolExecutor(max_workers=1) # one task at a time

            # RO, as to modify it the client must use .moveRel() or .moveAbs()
            self.position = model.VigilantAttribute({}, readonly=True)
            self._updatePosition()

        except Exception:
            self.Close()
            raise

    def _setProtection(self, value):
        """
        value (bool): True = TTL signal down (off), False = TTL signal up (on)
        """
        line = 0  # just a fixed line
        self.SetAccessory(line, not value)

        return value

    def Initialize(self):
        """
        Initialise the currently selected device
        """
        # Can take quite a lot of time due to the homing
        logging.debug("Initialising Andor Shamrock...") # ~20s
        if self._is_via_camera:
            path = self._camera._initpath
        else:
            path = ""
        self._dll.ShamrockInitialize(path)

    def Close(self):
        self._dll.ShamrockClose()

    def GetNumberDevices(self):
        """
        Returns (0<=int) the number of available Shamrocks
        """
        nodevices = c_int()
        self._dll.ShamrockGetNumberDevices(byref(nodevices))
        return nodevices.value

    def GetSerialNumber(self):
        """
        Returns the device serial number
        """
        serial = create_string_buffer(64) # hopefully always fit! (normally 6 bytes)
        self._dll.ShamrockGetSerialNumber(self._device, serial)
        return serial.value

    # Probably not needed, as ShamrockGetCalibration returns everything already
    # computed
    def EepromGetOpticalParams(self):
        """
        Returns (tuple of 3 floats): Focal Length (m), Angular Deviation (degree) and
           Focal Tilt (degree) from the Shamrock device.
        """
        FocalLength = c_float()
        AngularDeviation = c_float()
        FocalTilt = c_float()
        self._dll.ShamrockEepromGetOpticalParams(self._device,
                 byref(FocalLength), byref(AngularDeviation), byref(FocalTilt))

        return FocalLength.value, AngularDeviation.value, FocalTilt.value

    def SetGrating(self, grating):
        """
        grating (0<int<=3)
        """
        assert 1 <= grating <= 3

        # Seems currently the SDK sometimes fail with SHAMROCK_COMMUNICATION_ERROR
        # as in SetWavelength()
        with self._hw_access:
            retry = 0
            while True:
                try:
                    self._dll.ShamrockSetGrating(self._device, grating)
                except ShamrockError as (errno, strerr):
                    if errno != 20201 or retry >= 5: # SHAMROCK_COMMUNICATION_ERROR
                        raise
                    # just try again
                    retry += 1
                    logging.info("Failed to set wavelength, will try again")
                    time.sleep(0.1 * retry)
                else:
                    break

    def GetGrating(self):
        """
        return (0<int<=3): current grating
        """
        with self._hw_access:
            grating = c_int()
            self._dll.ShamrockGetGrating(self._device, byref(grating))
        return grating.value

    def GetNumberGratings(self):
        """
        return (0<int<=3): number of gratings present
        """
        noGratings = c_int()
        self._dll.ShamrockGetNumberGratings(self._device, byref(noGratings))
        return noGratings.value

    def WavelengthReset(self):
        """
        Resets the wavelength to 0 nm.
        """
        # Same as ShamrockGotoZeroOrder()
        with self._hw_access:
            self._dll.ShamrockWavelengthReset(self._device)

    #ShamrockAtZeroOrder(self._device, int *atZeroOrder);

    def GetGratingInfo(self, grating):
        """
        grating (0<int<=3)
        return:
              lines (float): number of lines / m
              blaze (str): wavelength or mirror info, as reported by the device
                Note that some devices add a unit (eg, nm), and some don't.
                When it is a mirror, there is typically a "mirror" keyword.
              home (int): beginning of the grating in steps
              offset (int): offset to the grating in steps
        """
        assert 1 <= grating <= 3
        Lines = c_float() # in l/mm
        Blaze = create_string_buffer(64) # decimal of wavelength in nm
        Home = c_int()
        Offset = c_int()
        self._dll.ShamrockGetGratingInfo(self._device, grating,
                         byref(Lines), Blaze, byref(Home), byref(Offset))
        logging.debug("Grating %d is %f, %s, %d, %d", grating,
                      Lines.value, Blaze.value, Home.value, Offset.value)

        return Lines.value * 1e3, Blaze.value, Home.value, Offset.value

    def SetWavelength(self, wavelength):
        """
        Sets the required wavelength.
        wavelength (0<=float): wavelength in m
        """
        assert 0 <= wavelength <= 50e-6

        # Currently the SDK sometimes fail with 20201: SHAMROCK_COMMUNICATION_ERROR
        # when changing wavelength by a few additional nm. It _seems_ that it
        # works anyway (but not sure).
        # It seems that retrying a couple of times just works
        with self._hw_access:
            retry = 0
            while True:
                try:
                    # set in nm
                    self._dll.ShamrockSetWavelength(self._device, c_float(wavelength * 1e9))
                except ShamrockError as (errno, strerr):
                    if errno != 20201 or retry >= 5: # SHAMROCK_COMMUNICATION_ERROR
                        raise
                    # just try again
                    retry += 1
                    logging.info("Failed to set wavelength, will try again")
                    time.sleep(0.1)
                else:
                    break

    def GetWavelength(self):
        """
        Gets the current wavelength.
        return (0<=float): wavelength in m
        """
        with self._hw_access:
            wavelength = c_float() # in nm
            self._dll.ShamrockGetWavelength(self._device, byref(wavelength))
        return wavelength.value * 1e-9

    def GetWavelengthLimits(self, grating):
        """
        grating (0<int<=3)
        return (0<=float< float): min, max wavelength in m
        """
        assert 1 <= grating <= 3
        Min, Max = c_float(), c_float() # in nm
        self._dll.ShamrockGetWavelengthLimits(self._device, grating,
                                              byref(Min), byref(Max))
        return Min.value * 1e-9, Max.value * 1e-9

    def WavelengthIsPresent(self):
        """
        return (boolean): True if it's possible to change the wavelength
        """
        present = c_int()
        self._dll.ShamrockWavelengthIsPresent(self._device, byref(present))
        return (present.value != 0)

    def GetCalibration(self, npixels):
        """
        npixels (0<int): number of pixels on the sensor. It's actually the
          length of the list that is being returned. Note: on small center
          wavelength, the values might be meaningless, and multiple 0 nm can be
          returned.
        return (list of floats of length npixels): wavelength in m
        """
        assert(0 < npixels)
        # Warning: if npixels <= 7, very weird/large values are returned (with SDK 2.100).
        # Probably because GetPixelCalibrationCoefficients() returns also very
        # strange polynomial.
        if npixels <= 7:
            logging.warning("Requested calibration info for %d pixels, which is known to fail", npixels)
        logging.debug("Requesting calibration info for %d px", npixels)
        # TODO: this is pretty slow, and could be optimised either by using a
        # numpy array or returning directly the C array. We could also just
        # allocate one array at the init, and reuse it.
        CalibrationValues = (c_float * npixels)()
        self._dll.ShamrockGetCalibration(self._device, CalibrationValues, npixels)
        # Note: it just applies the polynomial, so you can end up with negative
        # values => clamp them to 0.
        return [max(0, v * 1e-9) for v in CalibrationValues]

    def GetPixelCalibrationCoefficients(self):
        """
        return (4 floats)
        """
        a, b, c, d = c_float(), c_float(), c_float(), c_float()
        self._dll.ShamrockGetPixelCalibrationCoefficients(self._device, byref(a), byref(b), byref(c), byref(d))
        return a.value, b.value, c.value, d.value

    def GetCCDLimits(self, port):
        """
        No idea what it really does
        return (float, float): low/high wavelength in m
        """
        low = c_float()
        high = c_float()
        self._dll.ShamrockGetCCDLimits(self._device, port, byref(low), byref(high))
        return low.value * 1e-9, high.value * 1e-9

    def SetPixelWidth(self, width):
        """
        Defines the size of each pixel (horizontally).
        Needed to get correct information from GetCalibration()
        width (float): size of a pixel in m
        """
        # set in µm
        self._dll.ShamrockSetPixelWidth(self._device, c_float(width * 1e6))

    def SetNumberPixels(self, npixels):
        """
        Defines how many pixels (around the center) are used.
        Needed to get correct information from GetCalibration()
        npixels (int): number of pixels on the attached sensor
        """
        self._dll.ShamrockSetNumberPixels(self._device, npixels)

#self._dll.ShamrockGetPixelWidth(self._device, float* Width)
#self._dll.ShamrockGetNumberPixels(self._device, int* NumberPixels)

    # Focus mirror management
    def SetFocusMirror(self, steps):
        """
        Relative move on the focus
        steps (int): relative numbers of steps to do
        """
        assert isinstance(steps, int)
        # The documentation states focus is >=0, but SR193 only accepts >0
        self._dll.ShamrockSetFocusMirror(self._device, steps)

    def GetFocusMirror(self):
        """
        Get the current position of the focus
        return (0<=int<=maxsteps): absolute position (in steps)
        """
        focus = c_int()
        self._dll.ShamrockGetFocusMirror(self._device, byref(focus))
        return focus.value

    def GetFocusMirrorMaxSteps(self):
        """
        Get the maximum position of the focus
        return (0 <= int): absolute max position (in steps)
        """
        focus = c_int()
        self._dll.ShamrockGetFocusMirrorMaxSteps(self._device, byref(focus))
        return focus.value

    def FocusMirrorIsPresent(self):
        present = c_int()
        self._dll.ShamrockFocusMirrorIsPresent(self._device, byref(present))
        return (present.value != 0)

    # Filter wheel support
    def SetFilter(self, pos):
        """
        Absolute move on the filter wheel
        pos (1<=int<=6): new position
        """
        assert(FILTERMIN <= pos <= FILTERMAX)
        self._dll.ShamrockSetFilter(self._device, pos)

    def GetFilter(self):
        """
        Return the current absolute position of the filter wheel
        return (1<=int<=6): current filter
        """
        pos = c_int()
        self._dll.ShamrockGetFilter(self._device, byref(pos))
        return pos.value

    def GetFilterInfo(self, pos):
        """
        pos (int): filter number
        return (str): the text associated to the given filter
        """
        info = create_string_buffer(64)  # TODO: what's a good size? The SDK doc says nothing
        self._dll.ShamrockGetFilterInfo(self._device, pos, info)
        return info.value

    def FilterIsPresent(self):
        present = c_int()
        self._dll.ShamrockFilterIsPresent(self._device, byref(present))
        return (present.value != 0)

    # Slits management
    def SetAutoSlitWidth(self, index, width):
        """
        index (1<=int<=4): Slit number
        width (0<float): slit opening width in m
        """
        assert(SLIT_INDEX_MIN <= index <= SLIT_INDEX_MAX)
        width_um = c_float(width * 1e6)

        with self._hw_access:
            with self._led_access:
                self._dll.ShamrockSetAutoSlitWidth(self._device, index, width_um)

    def GetAutoSlitWidth(self, index):
        """
        index (1<=int<=4): Slit number
        return (0<float): slit opening width in m
        """
        assert(SLIT_INDEX_MIN <= index <= SLIT_INDEX_MAX)
        width_um = c_float()
        with self._hw_access:
            self._dll.ShamrockGetAutoSlitWidth(self._device, index, byref(width_um))
        return width_um.value * 1e-6

    def AutoSlitReset(self, index):
        """
        index (1<=int<=4): Slit number
        """
        assert(SLIT_INDEX_MIN <= index <= SLIT_INDEX_MAX)
        with self._hw_access:
            with self._led_access:
                self._dll.ShamrockAutoSlitReset(self._device, index)

    def AutoSlitIsPresent(self, index):
        """
        Finds if a specified slit is present.
        index (1<=int<=4): Slit number
        return (bool): True if slit is present
        """
        assert(SLIT_INDEX_MIN <= index <= SLIT_INDEX_MAX)
        present = c_int()
        self._dll.ShamrockAutoSlitIsPresent(self._device, index, byref(present))
        return (present.value != 0)

    # Shutter management
    def SetShutter(self, mode):
        assert(SHUTTERMODEMIN <= mode <= SHUTTERMODEMAX)
        with self._hw_access:
            self._dll.ShamrockSetShutter(self._device, mode)

    def ShamrockGetShutter(self):
        mode = c_int()

        with self._hw_access:
            self._dll.ShamrockGetShutter(self._device, byref(mode))
        return mode.value

    def IsModePossible(self, mode):
        possible = c_int()

        with self._hw_access:
            self._dll.ShamrockIsModePossible(self._device, byref(possible))
        return (possible.value != 0)

    def ShutterIsPresent(self):
        present = c_int()

        with self._hw_access:
            self._dll.ShamrockShutterIsPresent(self._device, byref(present))
        return (present.value != 0)

# Mirror flipper management
    def SetFlipperMirror(self, flipper, port):
        assert(FLIPPER_INDEX_MIN <= flipper <= FLIPPER_INDEX_MAX)
        assert(0 <= port <= 1)

        with self._hw_access:
            self._dll.ShamrockSetFlipperMirror(self._device, flipper, port)

    def GetFlipperMirror(self, flipper):
        assert(FLIPPER_INDEX_MIN <= flipper <= FLIPPER_INDEX_MAX)
        port = c_int()

        with self._hw_access:
            self._dll.ShamrockGetFlipperMirror(self._device, flipper, byref(port))
        return port.value

# def ShamrockFlipperMirrorReset(int device, int flipper);

    def FlipperMirrorIsPresent(self, flipper):
        assert(FLIPPER_INDEX_MIN <= flipper <= FLIPPER_INDEX_MAX)
        present = c_int()
        self._dll.ShamrockFlipperMirrorIsPresent(self._device, flipper, byref(present))
        return (present.value != 0)

    # "Accessory" port control (= 2 TTL lines)
    def SetAccessory(self, line, val):
        """
        line (0 <= int <= 1): line number
        val (boolean): True = On, False = Off
        """
        assert(ACCESSORYMIN <= line <= ACCESSORYMAX)
        if val:
            state = 1
        else:
            state = 0
        self._dll.ShamrockSetAccessory(self._device, line, state)

        # HACK: the Andor driver has a problem and sets the spectrograph in a
        # bad state after setting the accessory to True. This puts it back in a
        # good state.
        self.GetGrating()

    # def ShamrockGetAccessoryState(int device,int Accessory, int *state);
    def GetAccessoryState(self, line):
        """
        line (0 <= int <= 1): line number
        return (boolean): True = On, False = Off
        """
        assert(ACCESSORYMIN <= line <= ACCESSORYMAX)
        state = c_int()
        self._dll.ShamrockGetAccessoryState(self._device, line, byref(state))
        return (state.value != 0)

    def AccessoryIsPresent(self):
        present = c_int()
        self._dll.ShamrockAccessoryIsPresent(self._device, byref(present))
        return (present.value != 0)

    # Helper functions
    def _getGratingChoices(self):
        """
        return (dict int -> string): grating number to description
        """
        ngratings = self.GetNumberGratings()
        gchoices = {}
        for g in range(1, ngratings + 1):
            try:
                lines, blaze, home, offset = self.GetGratingInfo(g)
                if lines == 0 or "mirror" in blaze.lower():
                    logging.debug("Considering grating %d a mirror", g)
                    gchoices[g] = "mirror"
                else:
                    gchoices[g] = "%.0f l/mm (blaze: %s)" % (lines * 1e-3, blaze)
            except ShamrockError:
                logging.exception("Failed to get grating info for %d", g)
                gchoices[g] = "unknown"

        return gchoices

    # high-level methods (interface)
    def _updatePosition(self):
        """
        update the position VA
        """
        # TODO: support "axes" to limit the axes to update
        pos = {"wavelength": self.GetWavelength(),
               "grating": self.GetGrating()
              }

        if "focus" in self.axes:
            pos["focus"] = self.GetFocusMirror() * self._focus_step_size

        if "band" in self.axes:
            pos["band"] = self.GetFilter()

        for i, name in self._slit_names.items():
            if name in self.axes:
                pos[name] = self.GetAutoSlitWidth(i)

        if "flip-out" in self.axes:
            val = self.GetFlipperMirror(OUTPUT_FLIPPER)
            userv = [k for k, v in FLIPPER_TO_PORT.items() if v == val][0]
            pos["flip-out"] = userv

        # it's read-only, so we change it via _value
        self.position._value = pos
        self.position.notify(self.position.value)

    def getPixelToWavelength(self):
        """
        Return the lookup table pixel number of the CCD -> wavelength observed.
        That's correct for the given current CCD binning, and current
        grating/wavelength.
        return (list of floats): pixel number -> wavelength in m
        """
        # FIXME: calling this during a move seems to hang => need a hw access lock
        # If wavelength is 0, report empty list to indicate it makes no sense
        if self.position.value["wavelength"] <= 1e-9:
            return []

        npixels = self._camera.resolution.value[0]

        self.SetNumberPixels(npixels)
        self.SetPixelWidth(self._camera.pixelSize.value[0] * self._camera.binning.value[0])
        # TODO: GetCalibration() return several values identical (eg, 0's if
        # cw is < 75 nm). Need to decide if we return something better (what?)
        # or check all the clients handle this corner case well.
        return self.GetCalibration(npixels)

    def getOpeningToWavelength(self, width):
        """
        Computes the range of the wavelength observed for a given slit opening
        width (in front of the detector).
        That is correct for the current grating/wavelength.
        width (float): opening width in m
        return (float, float): minimum/maximum wavelength observed
        """
        # Pretend we have a small CCD and look at the wavelength at the side
        # Note: In theory, we could just say we have 2 pixels, but the SDK doesn't
        # seem to put the center exactly at the center of the sensor (ie, it
        # seems pixel npixels/2 get the center wavelength), and the SDK doesn't
        # like resolutions < 8 anyway.
        self.SetNumberPixels(10)
        self.SetPixelWidth(width / 10)
        calib = self.GetCalibration(10)
        return calib[0], calib[-1]

    @isasync
    def moveRel(self, shift):
        """
        Move the stage the defined values in m for each axis given.
        shift dict(string-> float): name of the axis and shift in m
        returns (Future): future that control the asynchronous move
        """
        if not shift:
            return model.InstantaneousFuture()
        self._checkMoveRel(shift)

        # cannot convert it directly to an absolute move, because
        # several in a row must mean they accumulate. So we queue a
        # special task. That also means the range check is delayed until
        # the actual position is known.

        actions = []
        for axis, s in shift.items():  # order doesn't matter
            if axis == "wavelength":
                actions.append((self._doSetWavelengthRel, s))
            elif axis == "focus":
                actions.append((self._doSetFocusRel, s))
            elif axis == self._slit_names.values():
                sid = [k for k, v in self._slit_names.items() if v == axis][0]
                actions.append((self._doSetSlitRel, sid, s))

        f = self._executor.submit(self._doMultipleActions, actions)
        return f

    @isasync
    def moveAbs(self, pos):
        """
        Move the stage the defined values in m for each axis given.
        pos dict(string-> float): name of the axis and new position in m
        returns (Future): future that control the asynchronous move
        """
        if not pos:
            return model.InstantaneousFuture()
        self._checkMoveAbs(pos)

        # If grating needs to be changed, change it first, then the wavelength
        ordered_axes = ("grating", "wavelength", "band", "focus", "flip-out") + tuple(self._slit_names.values())
        actions = []
        for axis in ordered_axes:
            try:
                p = pos[axis]
            except KeyError:
                continue
            if axis == "grating":
                actions.append((self._doSetGrating, p))
            elif axis == "wavelength":
                actions.append((self._doSetWavelengthAbs, p))
            elif axis == "band":
                actions.append((self._doSetFilter, p))
            elif axis == "focus":
                actions.append((self._doSetFocusAbs, p))
            elif axis == "flip-out":
                actions.append((self._doSetFlipper, OUTPUT_FLIPPER, p))
            elif axis in self._slit_names.values():
                sid = [k for k, v in self._slit_names.items() if v == axis][0]
                actions.append((self._doSetSlitAbs, sid, p))

        f = self._executor.submit(self._doMultipleActions, actions)
        return f

    def _doMultipleActions(self, actions):
        """
        Run multiple actions sequentially (as long as they don't raise exceptions
        actions (tuple of tuple(callable, *args)): ordered actions defined by the
          callable and the arguments
        """
        for a in actions:
            func, args = a[0], a[1:]
            func(*args)

    def _doSetWavelengthRel(self, shift):
        """
        Change the wavelength by a value
        """
        pos = self.GetWavelength() + shift
        # it's only now that we can check the absolute position is wrong
        minp, maxp = self.axes["wavelength"].range
        if not minp <= pos <= maxp:
            raise ValueError("Position %f of axis '%s' not within range %f→%f" %
                             (pos, "wavelength", minp, maxp))

        # don't complain if the user asked for non reachable wl: he couldn't know
        minp, maxp = self.GetWavelengthLimits(self.GetGrating())
        pos = min(max(minp, pos), maxp)

        self.SetWavelength(pos)
        self._updatePosition()

    def _doSetWavelengthAbs(self, pos):
        """
        Change the wavelength to a value
        """
        # don't complain if the user asked for non reachable wl: he couldn't know
        minp, maxp = self.GetWavelengthLimits(self.GetGrating())
        rpos = min(max(minp, pos), maxp)
        if rpos != pos:
            logging.info("Limiting wavelength to %f nm (requested %f nm)",
                         rpos * 1e9, pos * 1e9)

        self.SetWavelength(rpos)
        self._updatePosition()

    def _doSetGrating(self, g):
        """
        Setter for the grating VA.
        It will try to put the same wavelength as before the change of grating.
        Synchronous until the grating is finished (up to 20s)
        g (1<=int<=3): the new grating
        """
        self.SetGrating(g)
        # By default the Shamrock library keeps the same wavelength

        self._updatePosition()

    def _doSetFocusRel(self, shift):
        # it's only now that we can check the goal (absolute) position is wrong
        shift_st = int(round(shift / self._focus_step_size))
        steps = self.GetFocusMirror() + shift_st  # absolute pos
        if not 0 < steps <= self.GetFocusMirrorMaxSteps():
            rng = self.axes["focus"].range
            raise ValueError(u"Position %f of axis 'focus' not within range %f→%f" %
                             (steps * self._focus_step_size, rng[0], rng[1]))

        self.SetFocusMirror(shift_st)  # needs relative value
        self._updatePosition()

    def _doSetFocusAbs(self, pos):
        steps = int(round(pos / self._focus_step_size))
        shift_st = steps - self.GetFocusMirror()
        self.SetFocusMirror(shift_st)  # needs relative value
        self._updatePosition()

    def _doSetFilter(self, pos):
        self.SetFilter(pos)
        self._updatePosition()

    def _doSetSlitRel(self, sid, shift):
        """
        Change the slit width by a value
        sid (int): slit ID
        shift (float): change in opening size in m
        """
        width = self.GetAutoSlitWidth(sid) + shift
        # it's only now that we can check the absolute position is wrong
        n = self._slit_names[sid]
        rng = self.axes[n].range
        if not rng[0] <= width <= rng[1]:
            raise ValueError(u"Position %f of axis '%s' not within range %f→%f" %
                             (width, n, rng[0], rng[1]))

        self.SetAutoSlitWidth(sid, width)
        self._updatePosition()

    def _doSetSlitAbs(self, sid, width):
        """
        Change the slit width to a value
        sid (int): slit ID
        width (float): new position in m
        """
        self.SetAutoSlitWidth(sid, width)
        self._updatePosition()

    def _doSetFlipper(self, flipper, pos):
        """
        Change the flipper position to one of the two positions
        """
        v = FLIPPER_TO_PORT[pos]
        self.SetFlipperMirror(flipper, v)
        self._updatePosition()

    def stop(self, axes=None):
        """
        stops the motion
        Warning: Only not yet-executed moves can be cancelled, this hardware
          doesn't support stopping while a move is going on.
        """
        self._executor.cancel()

    def terminate(self):
        if self._executor:
            self.stop()
            self._executor.shutdown()
            self._executor = None

        if self._device is not None:
            logging.debug("Shutting down the spectrograph")
            self.Close()
            self._device = None

#     def __del__(self):
#         self.terminate()

    def selfTest(self):
        """
        Check whether the connection to the spectrograph works.
        return (boolean): False if it detects any problem
        """
        try:
            if 0 <= self.GetWavelength() <= 10e-6:
                return True
        except Exception:
            logging.exception("Self test failed")

        return False

    @staticmethod
    def scan():
        dll = ShamrockDLL()
        # TODO: for now it will only find the Shamrocks connected directly via
        # USB, the I²C connections are not detected.
        # => also try to find every AndorCam2 and connect via them?
        dll.ShamrockInitialize("")
        nodevices = c_int()
        dll.ShamrockGetNumberDevices(byref(nodevices))
        logging.info("Scanning %d Andor Shamrock devices", nodevices.value)
        dev = []
        for i in range(nodevices.value):
            dev.append(("Andor Shamrock", # TODO: add serial number
                        {"device": i})
                      )

        return dev


# Only for testing/simulation purpose
# Very rough version that is just enough so that if the wrapper behaves correctly,
# it returns the expected values. Copied from andorcam2

def _deref(p, typep):
    """
    p (byref object)
    typep (c_type): type of pointer
    Use .value to change the value of the object
    """
    # This is using internal ctypes attributes, that might change in later
    # versions. Ugly!
    # Another possibility would be to redefine byref by identity function:
    # byref= lambda x: x
    # and then dereferencing would be also identity function.
    return typep.from_address(addressof(p._obj))

def _val(obj):
    """
    return the value contained in the object. Needed because ctype automatically
    converts the arguments to c_types if they are not already c_type
    obj (c_type or python object)
    """
    if isinstance(obj, ctypes._SimpleCData):
        return obj.value
    else:
        return obj


class FakeShamrockDLL(object):
    """
    Fake ShamrockDLL. It basically simulates a spectrograph connected.
    """

    def __init__(self, ccd=None):
        # gratings: l/mm, blaze, home, offset, min wl, max wl
        self._gratings = [(299.9, "300.0", 1000, -200, 0.0, 5003.6),
                          # (601.02, "500.0", 10000, 26, 0.0, 1578.95),
                          (0.0, "Mirror", 10000, 26, 0.0, 0.0),
                          (1200.1, "500.0", 30000, -65, 0.0, 808.65)]

        self._cw = 300.2 # current wavelength (nm)
        self._cg = 1 # current grating (1->3)
        self._pw = 0 # pixel width
        self._np = 0 # number of pixels

        # focus
        self._focus_pos = 25  # steps
        self._focus_max = 500  # steps

        # filter wheel
        self._filter = 1  # current position
        # filter info: pos - 1 -> str
        self._filters = ("Filter 1",
                         "Filter 2",
                         "Filter 3",
                         "",
                         "Filter 5",
                         "",
                         )

        # slits: int (id) -> float (position in µm)
        self._slits = {1: 10.3,
                       3: 1000,
                      }
        # flippers: int (id) -> int (port number, 0 or 1)
        self._flippers = {2: 0}

        # accessory: 2 lines -> int (0 or 1)
        self._accessory = [0, 0]

        # just for simulating the limitation of the iDus
        self._ccd = ccd

    def _check_hw_access(self):
        """
        Simulate hw connection failure if the CCD is acquiring, like the
        SR303i via the I²C connection of the iDus
        """
        if self._ccd and self._ccd.GetStatus() == andorcam2.AndorV2DLL.DRV_ACQUIRING:
            raise ShamrockError(20201, ShamrockDLL.err_code[20201])

    def ShamrockInitialize(self, path):
        pass

    def ShamrockClose(self):
        self._cw = None # should cause failure if calling anything else

    def ShamrockGetNumberDevices(self, p_nodevices):
        nodevices = _deref(p_nodevices, c_int)
        nodevices.value = 1

    def ShamrockGetSerialNumber(self, device, serial):
        serial.value = "SR303fake"

    def ShamrockEepromGetOpticalParams(self, device, p_fl, p_ad, p_ft):
        fl = _deref(p_fl, c_float)
        ad = _deref(p_ad, c_float)
        ft = _deref(p_ft, c_float)
        fl.value = 0.301  # m
        ad.value = 2.3 # °
        ft.value = -2.1695098876953125  # °

    def ShamrockSetGrating(self, device, grating):
        self._check_hw_access()
        new_g = _val(grating)
        time.sleep(min(1, abs(new_g - self._cg)) * 5) # very bad estimation
        self._cg = new_g

    def ShamrockGetGrating(self, device, p_grating):
        self._check_hw_access()
        grating = _deref(p_grating, c_int)
        grating.value = self._cg

    def ShamrockGetNumberGratings(self, device, p_nogratings):
        nogratings = _deref(p_nogratings, c_int)
        nogratings.value = len(self._gratings)

    def ShamrockWavelengthReset(self, device):
        self._check_hw_access()
        time.sleep(abs(self._cw) / 1000)
        self._cw = 0

    def ShamrockGetGratingInfo(self, device, grating,
                               p_lines, s_blaze, p_home, p_offset):
        lines = _deref(p_lines, c_float)
        home = _deref(p_home, c_int)
        offset = _deref(p_offset, c_int)
        info = self._gratings[_val(grating) - 1][0:4]
        lines.value, s_blaze.value, home.value, offset.value = info

    def ShamrockSetWavelength(self, device, wavelength):
        self._check_hw_access()
        # TODO: raise if outside of the grating range
        new_wl = _val(wavelength)
        time.sleep(abs(self._cw - new_wl) / 1000)
        self._cw = new_wl

    def ShamrockGetWavelength(self, device, p_wavelength):
        self._check_hw_access()
        wavelength = _deref(p_wavelength, c_float)
        wavelength.value = self._cw

    def ShamrockGetWavelengthLimits(self, device, grating, p_min, p_max):
        minwl, maxwl = _deref(p_min, c_float), _deref(p_max, c_float)
        minwl.value, maxwl.value = self._gratings[_val(grating) - 1][4:6]

    def ShamrockWavelengthIsPresent(self, device, p_present):
        present = _deref(p_present, c_int)
        present.value = 1 # yes!

    def ShamrockGetCalibration(self, device, calibval, npixels):
        center = (self._np - 1) / 2 # pixel containing center wl
        px_wl = self._pw / 50 # in nm
        minwl = self._gratings[self._cg - 1][4]
        for i in range(npixels):
            # return stupid values (that look slightly correct)
            calibval[i] = max(minwl, self._cw + (i - center) * px_wl)

    def ShamrockSetPixelWidth(self, device, width):
        self._pw = _val(width)

    def ShamrockSetNumberPixels(self, device, npixels):
        self._np = _val(npixels)

    def ShamrockSetFocusMirror(self, device, focus):
        if 0 <= self._focus_pos + focus <= self._focus_max:
            self._focus_pos += focus
            time.sleep(abs(focus) / 100)  # 100 steps/s
        else:
            raise ShamrockError(20267, ShamrockDLL.err_code[20267])

    def ShamrockGetFocusMirror(self, device, p_focus):
        focus = _deref(p_focus, c_int)
        focus.value = self._focus_pos

    def ShamrockGetFocusMirrorMaxSteps(self, device, p_steps):
        steps = _deref(p_steps, c_int)
        steps.value = self._focus_max

    def ShamrockFocusMirrorReset(self, device):
        self._focus_pos = 0

    def ShamrockFocusMirrorIsPresent(self, device, p_present):
        present = _deref(p_present, c_int)
        present.value = 1 # yes !

    def ShamrockSetFilter(self, device, flter):
        if FILTERMIN <= flter <= FILTERMAX:
            dist = abs(self._filter - flter)
            time.sleep(dist)  # 1s / position
            # TODO: sleep based on most direct move
            self._filter = flter
        else:
            raise ShamrockError(20268, ShamrockDLL.err_code[20268])

    def ShamrockGetFilter(self, device, p_filter):
        flter = _deref(p_filter, c_int)
        flter.value = self._filter

    def ShamrockGetFilterInfo(self, device, flter, s_info):
        s_info.value = self._filters[flter - 1]

#     def ShamrockSetFilterInfo(self, device,int Filter, char* Info):
#     def ShamrockFilterReset(self, device):

    def ShamrockFilterIsPresent(self, device, p_present):
        present = _deref(p_present, c_int)
        if self._filters:
            present.value = 1
        else:
            present.value = 0

    def ShamrockAutoSlitIsPresent(self, device, index, p_present):
        present = _deref(p_present, c_int)
        if _val(index) in self._slits:
            present.value = 1
        else:
            present.value = 0

    def ShamrockGetAutoSlitWidth(self, device, index, p_width):
        width = _deref(p_width, c_float)
        width.value = self._slits[_val(index)]

    def ShamrockSetAutoSlitWidth(self, device, index, width):
        w = _val(width)
        if SLITWIDTHMIN <= w <= SLITWIDTHMAX:
            oldwidth = self._slits[_val(index)]
            time.sleep(abs(oldwidth - w) / 500)
            self._slits[_val(index)] = w
        else:
            raise ShamrockError(20268, ShamrockDLL.err_code[20268])

    def ShamrockShutterIsPresent(self, device, p_present):
        present = _deref(p_present, c_int)
        present.value = 0  # no!

    def ShamrockFlipperMirrorIsPresent(self, device, flipper, p_present):
        present = _deref(p_present, c_int)
        if _val(flipper) in self._flippers:
            present.value = 1
        else:
            present.value = 0

    def ShamrockSetFlipperMirror(self, device, flipper, port):
        p = _val(port)
        f = _val(flipper)
        if PORTMIN <= p <= PORTMAX:
            oldport = self._flippers[f]
            time.sleep(abs(oldport - p))
            self._flippers[f] = p
        else:
            raise ShamrockError(20268, ShamrockDLL.err_code[20268])

    def ShamrockGetFlipperMirror(self, device, flipper, p_port):
        port = _deref(p_port, c_int)
        port.value = self._flippers[_val(flipper)]

    def ShamrockAccessoryIsPresent(self, device, p_present):
        present = _deref(p_present, c_int)
        present.value = 1  # yes!

    def ShamrockSetAccessory(self, device, line, state):
        l = _val(line)
        s = _val(state)
        if ACCESSORYMIN <= l <= ACCESSORYMAX:
            self._accessory[l] = s
        else:
            raise ShamrockError(20268, ShamrockDLL.err_code[20268])


class AndorSpec(model.Detector):
    """
    Spectrometer component, based on a AndorCam2 and a Shamrock
    """
    def __init__(self, name, role, children=None, daemon=None, **kwargs):
        """
        All the arguments are identical to AndorCam2, expected: 
        children (dict string->kwargs): name of child must be "shamrock" and the
          kwargs contains the arguments passed to instantiate the Shamrock component
        """
        # we will fill the set of children with Components later in ._children
        model.Detector.__init__(self, name, role, daemon=daemon, **kwargs)

        # Create the detector (ccd) child
        try:
            dt_kwargs = children["andorcam2"]
        except Exception:
            raise ValueError("AndorSpec excepts one child named 'andorcam2'")

        # We could inherit from it, but difficult to not mix up .binning, .shape
        # .resolution...
        self._detector = andorcam2.AndorCam2(parent=self, daemon=daemon, **dt_kwargs)
        self.children.value.add(self._detector)
        dt = self._detector

        # Copy and adapt the VAs and roattributes from the detector
                # set up the detector part
        # check that the shape is "horizontal"
        if dt.shape[0] <= 1:
            raise ValueError("Child detector must have at least 2 pixels horizontally")
        if dt.shape[0] < dt.shape[1]:
            logging.warning("Child detector is shaped vertically (%dx%d), "
                            "this is probably incorrect, as wavelengths are "
                            "expected to be along the horizontal axis",
                            dt.shape[0], dt.shape[1])
        # shape is same as detector (raw sensor), but the max resolution is always flat
        self._shape = tuple(dt.shape) # duplicate

        # The resolution and binning are derived from the detector, but with
        # settings set so that there is only one horizontal line.
        if dt.binning.range[1][1] < dt.resolution.range[1][1]:
            # without software binning, we are stuck to the max binning
            logging.info("Spectrometer %s will only use a %d px band of the %d "
                         "px of the sensor", name, dt.binning.range[1][1],
                         dt.resolution.range[1][1])

        resolution = (dt.resolution.range[1][0], 1) # max,1
        # vertically: 1, with binning as big as possible
        binning = (dt.binning.value[0],
                   min(dt.binning.range[1][1], dt.resolution.range[1][1]))

        min_res = (dt.resolution.range[0][0], 1)
        max_res = (dt.resolution.range[1][0], 1)
        self.resolution = model.ResolutionVA(resolution, [min_res, max_res],
                                             setter=self._setResolution)
        # 2D binning is like a "small resolution"
        self._binning = binning
        self.binning = model.ResolutionVA(self._binning, dt.binning.range,
                                          setter=self._setBinning)

        self._setBinning(binning) # will also update the resolution

        # TODO: update also the metadata MD_SENSOR_PIXEL_SIZE
        pxs = dt.pixelSize.value[0], dt.pixelSize.value[1] * dt.binning.value[1]
        self.pixelSize = model.VigilantAttribute(pxs, unit="m", readonly=True)
        # Note: the metadata has no MD_PIXEL_SIZE, but a MD_WL_LIST

        # TODO: support software binning by rolling up our own dataflow that
        # does data merging
        assert dt.resolution.range[0][1] == 1
        self.data = dt.data

        # duplicate every other VA and Event from the detector
        # that includes required VAs like .exposureTime
        for aname, value in model.getVAs(dt).items() + model.getEvents(dt).items():
            if not hasattr(self, aname):
                setattr(self, aname, value)
            else:
                logging.debug("skipping duplication of already existing VA '%s'", aname)

        assert hasattr(self, "exposureTime")

        # Create the spectrograph (actuator) child
        try:
            sp_kwargs = children["shamrock"]
        except Exception:
            raise ValueError("AndorSpec excepts one child named 'shamrock'")

        self._spectrograph = Shamrock(parent=self, camera=self._detector,
                                      daemon=daemon, **sp_kwargs)
        self.children.value.add(self._spectrograph)

        self._spectrograph.position.subscribe(self._onPositionUpdate)
        self.resolution.subscribe(self._onResBinningUpdate)
        self.binning.subscribe(self._onResBinningUpdate, init=True)

    def _setBinning(self, value):
        """
        Called when "binning" VA is modified. It also updates the resolution so
        that the horizontal AOI is approximately the same. The vertical size
        stays 1.
        value (int): how many pixels horizontally and vertically
          are combined to create "super pixels"
        """
        prev_binning = self._binning
        self._binning = tuple(value) # duplicate

        # adapt horizontal resolution so that the AOI stays the same
        changeh = prev_binning[0] / self._binning[0]
        old_resolution = self.resolution.value
        assert old_resolution[1] == 1
        new_resh = int(round(old_resolution[0] * changeh))
        new_resh = max(min(new_resh, self.resolution.range[1][0]), self.resolution.range[0][0])
        new_resolution = (new_resh, 1)

        # setting resolution and binning is slightly tricky, because binning
        # will change resolution to keep the same area. So first set binning, then
        # resolution
        self._detector.binning.value = value
        self.resolution.value = new_resolution
        return value

    def _setResolution(self, value):
        """
        Called when the resolution VA is to be updated.
        """
        # only the width might change
        assert value[1] == 1

        # fit the width to the maximum possible given the binning
        max_size = int(self.resolution.range[1][0] // self._binning[0])
        min_size = int(math.ceil(self.resolution.range[0][0] / self._binning[0]))
        size = (max(min(value[0], max_size), min_size), 1)

        self._detector.resolution.value = size
        assert self._detector.resolution.value[1] == 1 # TODO: handle this by software mean

        return size

    def _onResBinningUpdate(self, value):
        """
        Called when the resolution or the binning changes
        """
        self._updateWavelengthList()

    def _onPositionUpdate(self, pos):
        """
        Called when the wavelength position or grating (ie, groove density)
          of the spectrograph is changed.
        """
        self._updateWavelengthList()

    def _updateWavelengthList(self):
        wll = self._spectrograph.getPixelToWavelength()
        md = {model.MD_WL_LIST: wll}
        self._detector.updateMetadata(md)

    def terminate(self):
        self._spectrograph.terminate()
        self._detector.terminate()

    def selfTest(self):
        return super(AndorSpec, self).selfTest() and self._spectrograph.selfTest()
