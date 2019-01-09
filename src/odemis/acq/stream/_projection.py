#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 24 Jan 2017

@author: Guilherme Stiebler

Copyright © 2017 Guilherme Stiebler, Éric Piel, Delmic

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

from __future__ import division

import threading
import weakref
import logging
import time
import math
import gc
import numpy

from odemis import model
from odemis.util import img, polar

from odemis.model import MD_POL_MODE, MD_POL_NONE


class DataProjection(object):

    def __init__(self, stream):
        '''
        stream (Stream): the Stream to project
        '''
        self.stream = stream
        self._im_needs_recompute = threading.Event()
        weak = weakref.ref(self)
        self._imthread = threading.Thread(target=self._image_thread,
                                          args=(weak,),
                                          name="Image computation")
        self._imthread.daemon = True
        self._imthread.start()

        # DataArray or None: RGB projection of the raw data
        self.image = model.VigilantAttribute(None)

    @staticmethod
    def _image_thread(wprojection):
        """ Called as a separate thread, and recomputes the image whenever it receives an event
        asking for it.

        Args:
            wprojection (Weakref to a DataProjection): the data projection to follow

        """

        try:
            projection = wprojection()
            name = "%s:%x" % (projection.stream.name.value, id(projection))
            im_needs_recompute = projection._im_needs_recompute
            # Only hold a weakref to allow the stream to be garbage collected
            # On GC, trigger im_needs_recompute so that the thread can end too
            wprojection = weakref.ref(projection, lambda o: im_needs_recompute.set())

            tnext = 0
            while True:
                del projection
                im_needs_recompute.wait()  # wait until a new image is available
                projection = wprojection()

                if projection is None:
                    logging.debug("Projection %s disappeared so ending image update thread", name)
                    break

                tnow = time.time()

                # sleep a bit to avoid refreshing too fast
                tsleep = tnext - tnow
                if tsleep > 0.0001:
                    time.sleep(tsleep)

                tnext = time.time() + 0.1  # max 10 Hz
                im_needs_recompute.clear()
                projection._updateImage()
        except Exception:
            logging.exception("image update thread failed")

        gc.collect()


class RGBSpatialProjection(DataProjection):

    def __init__(self, stream):
        '''
        stream (Stream): the Stream to project
        '''
        super(RGBSpatialProjection, self).__init__(stream)

        self.should_update = model.BooleanVA(False)
        self.name = stream.name
        self.image = model.VigilantAttribute(None)

        # Don't call at init, so don't set metadata if default value
        self.stream.tint.subscribe(self._onTint)
        self.stream.intensityRange.subscribe(self._onIntensityRange)

        # handle z stack
        if model.hasVA(stream, "zIndex"):
            self.zIndex = stream.zIndex
            self.zIndex.subscribe(self._onZIndex)

        if stream.raw and isinstance(stream.raw[0], model.DataArrayShadow):
            # The raw tiles corresponding to the .image, updated whenever .image is updated
            self._raw = (())  # 2D tuple of DataArrays
            raw = stream.raw[0]
            md = raw.metadata
            # get the pixel size of the full image
            ps = md[model.MD_PIXEL_SIZE]
            max_mpp = ps[0] * (2 ** raw.maxzoom)
            # sets the mpp as the X axis of the pixel size of the full image
            mpp_rng = (ps[0], max_mpp)
            self.mpp = model.FloatContinuous(max_mpp, mpp_rng, setter=self._set_mpp)

            full_rect = img._getBoundingBox(raw)
            l, t, r, b = full_rect
            rect_range = ((l, b, l, b), (r, t, r, t))
            self.rect = model.TupleContinuous(full_rect, rect_range)

            self.mpp.subscribe(self._onMpp)
            self.rect.subscribe(self._onRect)

            # initialize the projected tiles cache
            self._projectedTilesCache = {}
            # initialize the raw tiles cache
            self._rawTilesCache = {}

            # When True, the projected tiles cache should be invalidated
            self._projectedTilesInvalid = True

        self._shouldUpdateImage()

    @property
    def raw(self):
        if hasattr(self, "_raw"):
            return self._raw
        else:
            return self.stream.raw

    def _onIntensityRange(self, irange):
        logging.debug("Intensity range changed to %s", irange)
        self._shouldUpdateImageEntirely()

    def _onTint(self, value):
        self._shouldUpdateImageEntirely()

    def _shouldUpdateImageEntirely(self):
        """
        Indicate that the .image should be computed _and_ that all the previous
        tiles cached (and visible in the new image) have to be recomputed too
        """
        # set projected tiles cache as invalid
        self._projectedTilesInvalid = True
        self._shouldUpdateImage()

    def _onMpp(self, mpp):
        self._shouldUpdateImage()

    def _onRect(self, rect):
        self._shouldUpdateImage()

    def _set_mpp(self, mpp):
        ps0 = self.mpp.range[0]
        exp = math.log(mpp / ps0, 2)
        exp = round(exp)
        return ps0 * 2 ** exp

    def _projectXY2RGB(self, data, tint=(255, 255, 255)):
        """
        Project a 2D spatial DataArray into a RGB representation
        data (DataArray): 2D DataArray
        tint ((int, int, int)): colouration of the image, in RGB.
        return (DataArray): 3D DataArray
        """
        # TODO replace by local irange
        irange = self.stream._getDisplayIRange()
        rgbim = img.DataArray2RGB(data, irange, tint)
        rgbim.flags.writeable = False
        # Commented to prevent log flooding
        # if model.MD_ACQ_DATE in data.metadata:
        #     logging.debug("Computed RGB projection %g s after acquisition",
        #                    time.time() - data.metadata[model.MD_ACQ_DATE])
        md = self.stream._find_metadata(data.metadata)
        md[model.MD_DIMS] = "YXC" # RGB format
        return model.DataArray(rgbim, md)

    def _onZIndex(self, value):
        self._shouldUpdateImage()

    def _shouldUpdateImage(self):
        """
        Ensures that the image VA will be updated in the "near future".
        """
        # If the previous request is still being processed, the event
        # synchronization allows to delay it (without accumulation).
        self._im_needs_recompute.set()

    def getBoundingBox(self):
        ''' Get the bounding box of the whole image, whether it`s tiled or not.
        return (tuple of floats(l,t,r,b)): Tuple with the bounding box
        Raises:
            ValueError: If the .image member is not set
        '''
        if hasattr(self, 'rect'):
            rng = self.rect.range
            return (rng[0][0], rng[0][1], rng[1][0], rng[1][1])
        else:
            im = self.image.value
            if im is None:
                raise ValueError("Stream's image not defined")
            md = im.metadata
            pxs = md.get(model.MD_PIXEL_SIZE, (1e-6, 1e-6))
            pos = md.get(model.MD_POS, (0, 0))
            w, h = im.shape[1] * pxs[0], im.shape[0] * pxs[1]
            return [pos[0] - w / 2, pos[1] - h / 2, pos[0] + w / 2, pos[1] + h / 2]

    def _zFromMpp(self):
        """
        Return the zoom level based on the current .mpp value
        return (int): The zoom level based on the current .mpp value
        """
        md = self.stream.raw[0].metadata
        ps = md[model.MD_PIXEL_SIZE]
        return int(math.log(self.mpp.value / ps[0], 2))

    def _rectWorldToPixel(self, rect):
        """
        Convert rect from world coordinates to pixel coordinates
        rect (tuple containing x1, y1, x2, y2): Rect on world coordinates
        return (tuple containing x1, y1, x2, y2): Rect on pixel coordinates
        """
        das = self.stream.raw[0]
        md = das.metadata
        ps = md.get(model.MD_PIXEL_SIZE, (1e-6, 1e-6))
        pos = md.get(model.MD_POS, (0, 0))
        # Removes the center coordinates of the image. After that, rect will be centered on 0, 0
        rect = (
            rect[0] - pos[0],
            rect[1] - pos[1],
            rect[2] - pos[0],
            rect[3] - pos[1]
        )
        dims = md.get(model.MD_DIMS, "CTZYX"[-das.ndim::])
        img_shape = (das.shape[dims.index('X')], das.shape[dims.index('Y')])

        # Converts rect from physical to pixel coordinates.
        # The received rect is relative to the center of the image, but pixel coordinates
        # are relative to the top-left corner. So it also needs to sum half image.
        # The -1 are necessary on the right and bottom sides, as the coordinates of a pixel
        # are -1 relative to the side of the pixel
        # The '-' before ps[1] is necessary due to the fact that 
        # Y in pixel coordinates grows down, and Y in physical coordinates grows up
        return (
            int(round(rect[0] / ps[0] + img_shape[0] / 2)),
            int(round(rect[1] / (-ps[1]) + img_shape[1] / 2)),
            int(round(rect[2] / ps[0] + img_shape[0] / 2)) - 1,
            int(round(rect[3] / (-ps[1]) + img_shape[1] / 2)) - 1,
        )

    def _getTile(self, x, y, z, prev_raw_cache, prev_proj_cache):
        """
        Get a tile from a DataArrayShadow. Uses cache.
        The cache for projected tiles and the cache for raw tiles has always the same tiles
        x (int): X coordinate of the tile
        y (int): Y coordinate of the tile
        z (int): zoom level where the tile is
        prev_raw_cache (dictionary): raw tiles cache from the
            last execution of _updateImage
        prev_proj_cache (dictionary): projected tiles cache from the
            last execution of _updateImage
        return (DataArray, DataArray): raw tile and projected tile
        """
        # the key of the tile on the cache
        tile_key = "%d-%d-%d" % (x, y, z)

        # if the raw tile has been already cached, read it from the cache
        if tile_key in prev_raw_cache:
            raw_tile = prev_raw_cache[tile_key]
        elif tile_key in self._rawTilesCache:
            raw_tile = self._rawTilesCache[tile_key]
        else:
            # The tile was not cached, so it must be read from the file
            raw_tile = self.stream.raw[0].getTile(x, y, z)

        # if the projected tile has been already cached, read it from the cache
        if tile_key in prev_proj_cache:
            proj_tile = prev_proj_cache[tile_key]
        elif tile_key in self._projectedTilesCache:
            proj_tile = self._projectedTilesCache[tile_key]
        else:
            # The tile was not cached, so it must be projected again
            proj_tile = self._projectTile(raw_tile)

        # cache raw and projected tiles
        self._rawTilesCache[tile_key] = raw_tile
        self._projectedTilesCache[tile_key] = proj_tile
        return raw_tile, proj_tile

    def _projectTile(self, tile):
        """
        Project the tile
        tile (DataArray): Raw tile
        return (DataArray): Projected tile
        """
        dims = tile.metadata.get(model.MD_DIMS, "CTZYX"[-tile.ndim::])
        ci = dims.find("C")  # -1 if not found
        # is RGB
        if dims in ("CYX", "YXC") and tile.shape[ci] in (3, 4):
            # Just pass the RGB data on
            tile = img.ensureYXC(tile)
            tile.flags.writeable = False
            # merge and ensures all the needed metadata is there
            tile.metadata = self.stream._find_metadata(tile.metadata)
            tile.metadata[model.MD_DIMS] = "YXC" # RGB format
            return tile
        elif dims in ("ZYX",) and model.hasVA(self.stream, "zIndex"):
            tile = img.getYXFromZYX(tile, self.stream.zIndex.value)
            tile.metadata[model.MD_DIMS] = "ZYX"
        else:
            tile = img.ensure2DImage(tile)

        return self._projectXY2RGB(tile, self.stream.tint.value)

    def _getTilesFromSelectedArea(self):
        """
        Get the tiles inside the region defined by .rect and .mpp
        return (DataArray, DataArray): Raw tiles and projected tiles
        """
        # This custom exception is used when the .mpp or .rect values changes while
        # generating the tiles. If the values changes, everything needs to be recomputed
        class NeedRecomputeException(Exception):
            pass

        das = self.stream.raw[0]

        # store the previous cache to use in this execution
        prev_raw_cache = self._rawTilesCache
        prev_proj_cache = self._projectedTilesCache
        # Execute at least once. If mpp and rect changed in
        # the last execution of the loops, execute again
        need_recompute = True
        while need_recompute:
            z = self._zFromMpp()
            rect = self._rectWorldToPixel(self.rect.value)
            # convert the rect coords to tile indexes
            rect = [l / (2 ** z) for l in rect]
            rect = [int(math.floor(l / das.tile_shape[0])) for l in rect]
            x1, y1, x2, y2 = rect
            # the 4 lines below avoids that lots of old tiles
            # stays in instance caches
            prev_raw_cache.update(self._rawTilesCache)
            prev_proj_cache.update(self._projectedTilesCache)
            # empty current caches
            self._rawTilesCache = {}
            self._projectedTilesCache = {}

            raw_tiles = []
            projected_tiles = []
            need_recompute = False
            try:
                for x in range(x1, x2 + 1):
                    rt_column = []
                    pt_column = []

                    for y in range(y1, y2 + 1):
                        # the projected tiles cache is invalid
                        if self._projectedTilesInvalid:
                            self._projectedTilesCache = {}
                            prev_proj_cache = {}
                            self._projectedTilesInvalid = False
                            raise NeedRecomputeException()

                        # check if the image changed in the middle of the process
                        if self._im_needs_recompute.is_set():
                            self._im_needs_recompute.clear()
                            # Raise the exception, so everything will be calculated again,
                            # but using the cache from the last execution
                            raise NeedRecomputeException()

                        raw_tile, proj_tile = \
                                self._getTile(x, y, z, prev_raw_cache, prev_proj_cache)
                        rt_column.append(raw_tile)
                        pt_column.append(proj_tile)

                    raw_tiles.append(tuple(rt_column))
                    projected_tiles.append(tuple(pt_column))

            except NeedRecomputeException:
                # image changed
                need_recompute = True

        return tuple(raw_tiles), tuple(projected_tiles)

    def _updateImage(self):
        """ Recomputes the image with all the raw data available
        """
        # logging.debug("Updating image")
        raw = self.stream.raw
        if not raw:
            return

        try:
            if isinstance(raw[0], model.DataArrayShadow):
                # DataArrayShadow => need to get each tile individually
                self._raw, projected_tiles = self._getTilesFromSelectedArea()
                self.image.value = projected_tiles
            else:
                self.image.value = self._projectTile(raw[0])

        except Exception:
            logging.exception("Updating %s %s image", self.__class__.__name__, self.name.value)


class ARPolarimetryProjection(RGBSpatialProjection):

    def __init__(self, stream):
        '''
        stream (Stream): the Stream to project
        '''

        # polarization VA, degree of polarization VA (DOP)
        # check if any polarization analyzer data, (None) == no analyzer data (pol)
        if self.stream._pos.keys()[0][-1]:
            # use first entry in acquisition to populate VA (acq could have 1 or 6 pol pos)
            self.polarization = model.VAEnumerated(self.stream._pos.keys()[0][-1], choices=stream.polpos)

            # degree of polarization
            dop = {"non-polarized", "DOP", "DOLP", "DOCP"}
            self.degreePolarization = model.VAEnumerated("non-polarized", choices=dop)

        if self.stream._pos.keys()[0][-1]:
            self.polarization.subscribe(self._onPolarization)

        super(ARPolarimetryProjection, self).__init__(stream)

    def _project2Polar(self, pos):
        """
        Return the polar projection of the image at the given position.
        pos (float, float, string or None): position (must be part of the ._pos)
        returns DataArray: the polar projection
        """
        if pos in self._polar:
            polard = self._polar[pos]
        else:
            # Compute the polar representation
            data = self._pos[pos]
            try:
                if numpy.prod(data.shape) > (1280 * 1080):
                    # AR conversion fails with very large images due to too much
                    # memory consumed (> 2Gb). So, rescale + use a "degraded" type that
                    # uses less memory. As the display size is small (compared
                    # to the size of the input image, it shouldn't actually
                    # affect much the output.
                    logging.info("AR image is very large %s, will convert to "
                                 "azimuthal projection in reduced precision.",
                                 data.shape)
                    y, x = data.shape
                    if y > x:
                        small_shape = 1024, int(round(1024 * x / y))
                    else:
                        small_shape = int(round(1024 * y / x)), 1024
                    # resize
                    data = img.rescale_hq(data, small_shape)
                    dtype = numpy.float16
                else:
                    dtype = None  # just let the function use the best one

                # 2 x size of original image (on smallest axis) and at most
                # the size of a full-screen canvas
                size = min(min(data.shape) * 2, 1134)

                # TODO: First compute quickly a low resolution and then
                # compute a high resolution version.
                # TODO: could use the size of the canvas that will display
                # the image to save some computation time.

                # Get bg image, if existing. It must match the polarization (defaulting to MD_POL_NONE).
                bg_image = self._getBackground(data.metadata.get(MD_POL_MODE, MD_POL_NONE))

                if bg_image is None:
                    # Simple version: remove the background value
                    data0 = polar.ARBackgroundSubtract(data)
                else:
                    data0 = img.Subtract(data, bg_image)  # metadata from data

                # Warning: allocates lot of memory, which will not be free'd until
                # the current thread is terminated.
                polard = polar.AngleResolved2Polar(data0, size, hole=False, dtype=dtype)

                # TODO: don't hold too many of them in cache (eg, max 3 * 1134**2)
                self._polar[pos] = polard
            except Exception:
                logging.exception("Failed to convert to azimuthal projection")
                return data  # display it raw as fallback

        return polard

    def _getBackground(self, pol_mode):
        """
        Get background image from background VA
        :param pol_mode: metadata
        :return: (DataArray or None): the background image corresponding to the given polarization,
                 or None, if no background corresponds.
        """
        bg_data = self.background.value  # list containing DataArrays, DataArray or None

        if bg_data is None:
            return None

        if isinstance(bg_data, model.DataArray):
            bg_data = [bg_data]  # convert to list of bg images

        for bg in bg_data:
            # if no analyzer hardware, set MD_POL_MODE = "pass-through" (MD_POL_NONE)
            if bg.metadata.get(MD_POL_MODE, MD_POL_NONE) == pol_mode:
                # should be only one bg image with the same metadata entry
                return bg  # DataArray

        # Nothing found e.g. pol_mode = "rhc" but no bg image with "rhc"
        logging.debug("No background image with polarization mode %s ." % pol_mode)
        return None

    def _find_metadata(self, md):
        # For polar view, no PIXEL_SIZE nor POS
        return {}

    def _updateImage(self):
        """ Recomputes the image with all the raw data available for the current
        selected point.
        """
        if not self.raw:
            return

        pos = self.point.value
        try:
            if pos == (None, None):
                self.image.value = None
            else:
                if self._pos.keys()[0][-1]:
                    pol = self.polarization.value
                else:
                    pol = None
                polard = self._project2Polar(pos + (pol,))
                # update the histogram
                # TODO: cache the histogram per image
                # FIXME: histogram should not include the black pixels outside
                # of the circle. => use a masked array?
                # reset the drange to ensure that it doesn't depend on older data
                self._drange = None
                self._updateHistogram(polard)
                self.image.value = self._projectXY2RGB(polard)
        except Exception:
            logging.exception("Updating %s image", self.__class__.__name__)

    def _onPolarization(self, pos):
        self._shouldUpdateImage()

    def _setBackground(self, bg_data):
        """
        Called when the background is about to be changed
        :param bg_data: (None, DataArray or list of DataArrays) background image(s)
        :return: (None, DataArray or list of DataArrays)
        :raises: (ValueError) the background data is not compatible with the data
                 (ex: incompatible resolution (shape), encoding (data type), format (bits),
                 polarization of images).
        """
        if bg_data is None:
            # simple baseline background value will be subtracted
            return bg_data

        isDataArray = False
        if isinstance(bg_data, model.DataArray):
            bg_data = [bg_data]
            isDataArray = True

        bg_data = [img.ensure2DImage(d) for d in bg_data]

        for d in bg_data:
            # TODO check if MD_AR_POLE in MD? will fail in set_ar_background anyways,
            # but maybe nicer to check here already
            arpole = d.metadata[model.MD_AR_POLE]  # we expect the data has AR_POLE

            # TODO: allow data which is the same shape but lower binning by
            # estimating the binned image
            # Check the background data and all the raw data have the same resolution
            # TODO: how to handle if the .raw has different resolutions?
            for r in self.raw:
                if d.shape != r.shape:
                    raise ValueError("Incompatible resolution of background data "
                                     "%s with the angular resolved resolution %s." %
                                     (d.shape, r.shape))
                if d.dtype != r.dtype:
                    raise ValueError("Incompatible encoding of background data "
                                     "%s with the angular resolved encoding %s." %
                                     (d.dtype, r.dtype))
                try:
                    if d.metadata[model.MD_BPP] != r.metadata[model.MD_BPP]:
                        raise ValueError(
                            "Incompatible format of background data "
                            "(%d bits) with the angular resolved format "
                            "(%d bits)." %
                            (d.metadata[model.MD_BPP], r.metadata[model.MD_BPP]))
                except KeyError:
                    pass  # no metadata, let's hope it's the same BPP

                # check the AR pole is at the same position
                if r.metadata[model.MD_AR_POLE] != arpole:
                    logging.warning("Pole position of background data %s is "
                                    "different from the data %s.",
                                    arpole, r.metadata[model.MD_AR_POLE])

                if MD_POL_MODE in r.metadata:  # check if we have polarization analyzer hardware present
                    # check if we have at least one bg image with the corresponding MD_POL_MODE to the image data
                    if not any(bg_im.metadata[MD_POL_MODE] == r.metadata[MD_POL_MODE] for bg_im in bg_data):
                        raise ValueError("No AR background with polarization %s" % r.metadata[MD_POL_MODE])

        if isDataArray:
            return bg_data[0]
        else:  # list
            return bg_data

    def _onBackground(self, data):
        """Called after the background has changed"""
        # uncache all the polar images, and update the current image
        self._polar = {}
        super(ARPolarimetryProjection, self)._onBackground(data)

