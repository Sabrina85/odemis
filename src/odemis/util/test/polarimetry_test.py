# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 07:43:16 2018

@author: coene
"""

import numpy as np
import matplotlib.pyplot as plt
from odemis.util import polarimetry_functions as plf
from odemis.util import plotCL as pcl
from odemis.util import ARfunctions as arf  # TODO to replace
import h5py as h5
import unittest

# plt.close("all")
# TODO need to read mumat already beforehand, otherwise display takes too long

# read inverse mumat
# matrix = h5.File("mumat_inv.h5")
# mumat = matrix["mumat"]
#
# f = h5.File("20180814-6polwithtxt.h5")

# TODO need f.close somewhere later?


class TestPolarConversion(unittest.TestCase):
    """
    Test calculation of stokes parameters in sample plane for images acquired
    with a polarization analyzer.
    """

    # read inverse mumat
    matrix = h5.File("mumat_inv.h5")
    mumat = matrix["mumat"]

    f = h5.File("20180814-6polwithtxt.h5")

    det_x = 1024
    det_y = 1024

    # TODO mapping of mumat for wavelength: first entry (0) is 200nm ...
    # TODO consider starting and stepsize, use shape of mumat to calc
    wl = 750e-9
    wl_start = 700e-9
    wl_step = 10e-9
    wl_index = (wl - wl_start)/wl_step

    # read image
    # TODO read image: read multiple ebeam pos?
    # TODO Order in h5 might be different!


    keys = ["Acquisition%d" % i for i in range(2, 8)]
    det_data = []
    pol_pos = []
    for key in f.keys():
        if key in keys:
            det_data.append(np.squeeze(f[key]["ImageData"]['Image']))
            pol_pos.append(f[key]["PhysicalData"]["Polarization"].value[0])

    det_data = np.array(det_data)
    det_data = np.swapaxes(det_data, 0, 2)

    # calc stokes vector detector plane
    # TODO check how matrix looks like when we have multiple ebeam pos! Still pass only one ebeam pos to fct?
    Stokes_det1 = plf.StokesCalc_CCD(det_data, pol_pos)

    # TODO check if we can use our code for masking data now and improve runtime
    # make mask for angular data that sets angles that are not collected by the mirror to 0
    thetalist = np.linspace(0, np.pi/2, det_x)
    philist = np.linspace(0, 2*np.pi, det_y)
    mask = arf.ARmask_calc(thetalist, philist, 0)
    maskpar = np.transpose(np.tile(mask, (4, 1, 1)), (1, 2, 0))

    Stokes_det1_plot = Stokes_det1 * maskpar  # TODO check if we can use some AR fct we already have for masking

    # plot recalculated stokes vectors in detector plane.
    # s1 = 1 means Ey = 1, Ez = 0, s1 = -1 means Ey = 0 and Ez = 1
    # pcl.AR_polar(Stokes_det1_plot[:,:,0]/Stokes_det1_plot[:,:,0].max(),[0,1])
    pcl.AR_polar_stokes_norm(Stokes_det1_plot[:, :, 1]/Stokes_det1_plot[:, :, 0], 1)
    pcl.AR_polar_stokes_norm(Stokes_det1_plot[:, :, 2]/Stokes_det1_plot[:, :, 0], 1)
    pcl.AR_polar_stokes_norm(Stokes_det1_plot[:, :, 3]/Stokes_det1_plot[:, :, 0], 1)

    # apply inverse mueller and get stokes vector sample plane
    Stokes_sample1 = plf.MuellerCor_DetToSample_Inv(Stokes_det1, mumat[:, :, :, :, wl_index])

    # plot normalized stokes parameters in sample plane
    pcl.AR_polar_stokes_norm(Stokes_sample1[:, :, 1]/Stokes_det1_plot[:, :, 0], 1)
    pcl.AR_polar_stokes_norm(Stokes_sample1[:, :, 2]/Stokes_det1_plot[:, :, 0], 1)
    pcl.AR_polar_stokes_norm(Stokes_sample1[:, :, 3]/Stokes_det1_plot[:, :, 0], 1)


    # calculate spherical field components from stokes vectors in sample plane
    E_spher1 = plf.StokesVecToEpar(Stokes_sample1)
    E_spher1[np.isnan(E_spher1)] = 0
    pcl.AR_polar(np.abs(E_spher1[:, :, 0]), [0, 1])
    pcl.AR_polar(np.abs(E_spher1[:, :, 1]), [0, 1])

    plt.figure()
    plt.imshow(np.real(E_spher1[:, :, 0]))
    plt.figure()
    plt.imshow(np.imag(E_spher1[:, :, 0]))

    pcl.AR_phasedif_plot(E_spher1)  # TODO what useful for?

    # convert E fields to cartesian coordinates
    E_cart1 = plf.SpherToCar(E_spher1)
    E_cart1[np.isnan(E_cart1)] = 0
    pcl.AR_polar(np.abs(E_cart1[:, :, 0]/np.abs(E_cart1[:, :, 0]).max()), [0, 1])
    pcl.AR_polar(np.abs(E_cart1[:, :, 1]/np.abs(E_cart1[:, :, 0]).max()), [0, 1])
    pcl.AR_polar(np.abs(E_cart1[:, :, 2]/np.abs(E_cart1[:, :, 0]).max()), [0, 1])

    # TODO calc polarization ratios

    DOP = np.sqrt(Stokes_sample1[:, :, 1]**2 + Stokes_sample1[:, :, 2]**2 + Stokes_sample1[:, :, 3]**2) / \
          Stokes_sample1[:, :, 0]
    DOLP = np.sqrt(Stokes_sample1[:, :, 1]**2 + Stokes_sample1[:, :, 2]**2) / Stokes_sample1[:, :, 0]
    DOCP = np.abs(Stokes_sample1[:, :, 3] / Stokes_sample1[:, :, 0])
    nonpol = 1 - DOP

    POL = np.zeros([det_x, det_y, 4])
    POL[:, :, 0] = nonpol
    POL[:, :, 0] = DOP
    POL[:, :, 0] = DOLP
    POL[:, :, 0] = DOCP

    plt.figure()
    plt.imshow(POL[:, :, 0])
    plt.figure()
    plt.imshow(POL[:, :, 1])
    plt.figure()
    plt.imshow(POL[:, :, 2])
    plt.figure()
    plt.imshow(POL[:, :, 3])

    print "test"
    # TODO for each ebeam pos a matrix where each value corresponds to a emission angle of the sample??
    # TODO conversion to polar coordinates
    # see below --> postpone ask Toon


# ---------------------------------------------------------------------------------------------------------------

# define angular grid  # TODO is 501, 801 shape of detector? how to handle when binning and pixelsize? mapping fct?
det_x = 1024
det_y = 1024
# thetalist = np.linspace(0, np.pi/2, 501)
# philist = np.linspace(0, 2*np.pi, 801)
thetalist = np.linspace(0, np.pi/2, det_x)
philist = np.linspace(0, 2*np.pi, det_y)
phi, theta = np.meshgrid(philist, thetalist)

# # TODO create inverse Mueller matrix/lookuptable for different lambda
# # TODO which range? 300 to 800? stepsize necessary? or use some interpolation
# wl_start = 700e-9
# wl_stop = 800e-9
# wl_step = 10e-9
#
#
# wl_range = np.arange(wl_start, wl_stop, wl_step)
#
# mumat = np.zeros([len(thetalist), len(philist), 4, 4, len(wl_range)])  # 4: stokes params TODO which det vs sample plane
#
# # create mueller matrix (mumat) for wavelength range wl_range
# for index, wl in enumerate(wl_range):
#     # calculate optical constants of mirror at wavelength of interest
#     # TODO here are the files McPeak_n.csv and McPeak_k.csv or Rakic_n.csv and Rakic_k.csv used! check if we are allowed to use those!
#     # TODO what do these data sets show? wavelength range is large, interpolation is also used ...
#     nsurf, eps = plf.al_opticalconstants(wl, oset=1)
#     # calculate mueller matrices for every angle and wavelength
#     mumat[:, :, :, :, index] = plf.CalcMuellerMatrix(phi, theta, nsurf)
#
# mumat_inv = np.zeros([len(thetalist), len(philist), 4, 4, len(wl_range)])
#
# # TODO does not work properly
# # create inverse mueller matrix (mumat) for wavelength range wl_range
# for index, wl in enumerate(wl_range):
#     # calculate inverse mueller matrices for every angle and wavelength
#     for ii in range(det_x):
#         for jj in range(det_y):
#             mumat_inv[ii, jj, :, :, index] = np.linalg.inv(mumat[ii, jj, :, :, index])
#
# # export mumat
# f = h5.File("mumat_inv_large.h5", "w")
# f.create_dataset("mumat", data=mumat_inv, compression="gzip")
# f.close()
#
#
# ################################### start test with pre-computed mumat

# f = h5.File("mumat.h5")
# mumat = f["mumat1"]
#
# # TODO get the start and stop wl the mumat was computed for
# # TODO changed the detector size
#
# # wavelength of interest
# wl = 750e-9
# wl_start = 700e-9
# wl_step = 10e-9
# wl_index = (wl - wl_start)/wl_step
#
# # speed of light
# c = 2.99792458e8
#
# # optical constants dipole environment
# ngold = 0.098537 + 4.78j
# struct = np.array([1., ngold])
# # struct = np.array([1., 4])
#
# # dipole orienations in-plane y ^ in-plane x --> z o
# pdipvecs = np.array([0, 0, 1, 0, 0, 0])
# rdip = np.array([0, 0, 50e-9])
# wl = 750e-9
# omega = c*2*np.pi/wl
#
# # calculate theoretical electric or magnetic dipole fields.
# dif_power, EH_fields = PM_dip_c.EvaluateFarFieldDifferentialPowerPM(theta, phi, pdipvecs, rdip, omega, struct)
# dif_power[np.isnan(dif_power)] = 0
#
# # Plot theoretical angular intensity and field distribution
# pcl.AR_polar(dif_power/dif_power.max(),[0,1])
# pcl.AR_polar(np.abs(EH_fields[:,:,0])/np.abs(EH_fields[:,:,0]).max(),[0,1])
# pcl.AR_polar(np.abs(EH_fields[:,:,1])/np.abs(EH_fields[:,:,0]).max(),[0,1])
# pcl.AR_polar(np.abs(EH_fields[:,:,2])/np.abs(EH_fields[:,:,0]).max(),[0,1])
#
# # convert cartesian fields to spherical fields which form natural basis
# E_spher = plf.CarToSpher(EH_fields[:, :, 0:3]/1e13)
# pcl.AR_phasedif_plot(E_spher)
# pcl.AR_polar(np.abs(E_spher[:,:,0]),[0,1])
# pcl.AR_polar(np.abs(E_spher[:,:,1]),[0,1])

# # calculate stokes vectors in sample plane
# # TODO I thought this is what we want to calc from the detector plane? only for testing?
# # TODO needed to create Muellermatrix? how stokes_sample evaluated? known intensity distribution?
# Stokes_sample = plf.EtoStokesPar(E_spher)
#
# # plot stokes parameters in sample plane
# ## pcl.AR_polar(Stokes_sample[:,:,0]/Stokes_sample[:,:,0].max(),[0,1])
# # pcl.AR_polar_stokes_norm(Stokes_sample[:,:,1]/Stokes_sample[:,:,0].max(),1)
# # pcl.AR_polar_stokes_norm(Stokes_sample[:,:,2]/Stokes_sample[:,:,0].max(),1)
# # pcl.AR_polar_stokes_norm(Stokes_sample[:,:,3]/Stokes_sample[:,:,0].max(),1)
#
#
# # TODO check if we can use our code for masking data now and improve runtime
# # make mask for angular data that sets angles that are not collected by the mirror to 0
# mask = arf.ARmask_calc(thetalist, philist, 0)
# maskpar = np.transpose(np.tile(mask, (4, 1, 1)), (1, 2, 0))
# maskpar1 = np.transpose(np.tile(mask, (6, 1, 1)), (1, 2, 0))
# # maskpar[maskpar == 0] = -1000
#
# # TODO here input our 6 pol pos data and calc the 4 stokes params in detector plane, then apply inverse mumat
# # calculate stokes vectors in detector plane by multiplying with mueller matrix
# Stokes_det = plf.MuellerCor_SampleToDet(Stokes_sample, mumat[:, :, :, :, wl_index])
# Stokes_det_plot = Stokes_det  # *maskpar
#
# # TODO which are final images showing the detected image corrected by the muellermatrix in the sample plane
#
# # TODO show these plots to user? Check dependencies!
# # plot stokes vectors in detector plane. Can be compared to Osorio et al for vertical dipole orientation
# # pcl.AR_polar(Stokes_det_plot[:,:,0]/Stokes_det_plot[:,:,0].max(),[0,1])
# pcl.AR_polar_stokes_norm(Stokes_det_plot[:, :, 1]/Stokes_det_plot[:, :, 0], 1)
# pcl.AR_polar_stokes_norm(Stokes_det_plot[:, :, 2]/Stokes_det_plot[:, :, 0], 1)
# pcl.AR_polar_stokes_norm(Stokes_det_plot[:, :, 3]/Stokes_det_plot[:, :, 0], 1)
#
# # TODO check if same pos as we specify in yaml file
# # QWP and LP settings for polarimetry analysis
# QWangle = np.array([np.pi/2, 0, 3*np.pi/4, np.pi/4, 0., 0.])
# polangle = np.array([np.pi/2, 0, 3*np.pi/4, np.pi/4, np.pi/4, 3*np.pi/4])
#
#
# print(QWangle*180/np.pi, polangle*180/np.pi)
#
# S0_list = np.empty([theta.shape[0], theta.shape[1], 6], dtype="float64")
#
# # TODO 6 pol pos? S0?
# # calculate polarization filtered stokes parameters in detector plane.
# for pp in range(6):
#
#     Sout_filt = plf.Mueller_Analyzer(Stokes_det, QWangle[pp], polangle[pp])
#     #S0 would correspond to the actual measurement that would normally be performed
#     S0_list[:, :, pp] = Sout_filt[:, :, 0]
#
# S0_list_plot = S0_list*maskpar1
#
# # plot polarization filtered S0 patterns
# for gg in range(0, 6):
#
#         pcl.AR_polar(S0_list_plot[:, :, gg]/S0_list_plot.max(), [0, 1])
#
# ###############################################################################################################
# # recalculate stokes parameters in detector plane based on 6 filtered S0's. At this point a normal analysis would start  # TODO S0?
# Stokes_det1 = plf.StokesCalc_CCD(S0_list)
# Stokes_det1_plot = Stokes_det1*maskpar
#
# # plot recalculated stokes vectors in detector plane.
# # s1 = 1 means Ey = 1, Ez = 0, s1 = -1 means Ey = 0 and Ez = 1
# # pcl.AR_polar(Stokes_det1_plot[:,:,0]/Stokes_det1_plot[:,:,0].max(),[0,1])
# pcl.AR_polar_stokes_norm(Stokes_det1_plot[:, :, 1]/Stokes_det1_plot[:, :, 0], 1)
# pcl.AR_polar_stokes_norm(Stokes_det1_plot[:, :, 2]/Stokes_det1_plot[:, :, 0], 1)
# pcl.AR_polar_stokes_norm(Stokes_det1_plot[:, :, 3]/Stokes_det1_plot[:, :, 0], 1)
#
# # back correction to sample plane
# Stokes_sample1 = plf.MuellerCor_DetToSample(Stokes_det, mumat[:, :, :, :, wl_index])
#
# # plot normalized stokes parameters in sample plane
# ## pcl.AR_polar_stokes_norm(Stokes_sample1[:,:,1]/Stokes_det1_plot[:,:,0],1)
# ## pcl.AR_polar_stokes_norm(Stokes_sample1[:,:,2]/Stokes_det1_plot[:,:,0],1)
# ## pcl.AR_polar_stokes_norm(Stokes_sample1[:,:,3]/Stokes_det1_plot[:,:,0],1)
#
# # calculate spherical field components from stokes vectors
# E_spher1 = plf.StokesVecToEpar(Stokes_sample1)
# E_spher1[np.isnan(E_spher1)] = 0
# # pcl.AR_polar(np.abs(E_spher1[:,:,0]),[0,1])
# # pcl.AR_polar(np.abs(E_spher1[:,:,1]),[0,1])
#
# plt.figure()
# plt.imshow(np.real(E_spher[:, :, 0]))
#
# plt.figure()
# plt.imshow(np.real(E_spher1[:, :, 0]))
#
# plt.figure()
# plt.imshow(np.imag(E_spher[:, :, 0]))
#
# plt.figure()
# plt.imshow(np.imag(E_spher1[:, :, 0]))
#
# pcl.AR_phasedif_plot(E_spher1)
#
# # check whether reconstruction is accurate
# diff_fields = np.abs(E_spher1)-np.abs(E_spher)/(np.abs(E_spher1)+np.abs(E_spher)).sum()
#
# # convert to cartesian coordinates
# E_cart1 = plf.SpherToCar(E_spher1)
# E_cart1[np.isnan(E_cart1)] = 0
#
# pcl.AR_polar(np.abs(E_cart1[:, :, 0]/np.abs(E_cart1[:, :, 0]).max()), [0,1])
# pcl.AR_polar(np.abs(E_cart1[:, :, 1]/np.abs(E_cart1[:, :, 0]).max()), [0,1])
# pcl.AR_polar(np.abs(E_cart1[:, :, 2]/np.abs(E_cart1[:, :, 0]).max()), [0,1])
print "test"

# TODO need some assert test case checking back and forth calc images are the same
