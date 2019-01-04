#import polarimetry_master as pol_m
#import polarimetry_functions as pol_f
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import PM_dipole_code as PM_dip_c
# import polarimetry_functions as plf
# import plotCL as pcl
# import ARfunctions as arf  # TODO to replace
import h5py as h5


# calc stoke params in sample plane for an acquired image
class CalStokesParams():
    """

    """

    def __init__(self):
        # read mumat
        f = h5.File("mumat.h5")
        mumat = f["mumat1"]
        # TODO need f.close somewhere later?

        det_x = 1024
        det_y = 1024

        # TODO mapping of mumat for wavelength: first entry (0) is 200nm ...
        # TODO consider starting and stepsize, use shape of mumat to calc
        wl = 750e-9
        wl_start = 700e-9
        wl_step = 10e-9
        wl_index = (wl - wl_start)/wl_step

        det_data = self.readImageData()

        # TODO calc stokes vector detector plane
        # TODO check how matrix looks like when we have multiple ebeam pos! Still pass only one ebeam pos to fct?
        Stokes_det1 = self.StokesCalc_CCD(det_data, pol_pos)

        # TODO check if we can use our code for masking data now and improve runtime
        # make mask for angular data that sets angles that are not collected by the mirror to 0
        thetalist = np.linspace(0, np.pi/2, det_x)
        philist = np.linspace(0, 2*np.pi, det_y)
        mask = arf.ARmask_calc(thetalist, philist, 0)
        maskpar = np.transpose(np.tile(mask, (4, 1, 1)), (1, 2, 0))

        Stokes_det1_plot = Stokes_det1*maskpar  # TODO check if we can use some AR fct we already have for masking

        # plot recalculated stokes vectors in detector plane.
        # s1 = 1 means Ey = 1, Ez = 0, s1 = -1 means Ey = 0 and Ez = 1
        # pcl.AR_polar(Stokes_det1_plot[:,:,0]/Stokes_det1_plot[:,:,0].max(),[0,1])
        pcl.AR_polar_stokes_norm(Stokes_det1_plot[:, :, 1]/Stokes_det1_plot[:, :, 0], 1)
        pcl.AR_polar_stokes_norm(Stokes_det1_plot[:, :, 2]/Stokes_det1_plot[:, :, 0], 1)
        pcl.AR_polar_stokes_norm(Stokes_det1_plot[:, :, 3]/Stokes_det1_plot[:, :, 0], 1)

        # TODO apply inverse mueller calc and get stokes vector sample plane

        Stokes_sample = self.MuellerCor_DetToSample_Inv(Stokes_det1, mumat[:, :, :, :, wl_index])

        # plot normalized stokes parameters in sample plane
        pcl.AR_polar_stokes_norm(Stokes_sample[:, :, 1]/Stokes_det1_plot[:, :, 0], 1)
        pcl.AR_polar_stokes_norm(Stokes_sample[:, :, 2]/Stokes_det1_plot[:, :, 0], 1)
        pcl.AR_polar_stokes_norm(Stokes_sample[:, :, 3]/Stokes_det1_plot[:, :, 0], 1)


        # TODO calc polarization ratios

        DOP = np.sqrt(Stokes_sample[:, :, 1]**2 + Stokes_sample[:, :, 2]**2 + Stokes_sample[:, :, 3]**2) / \
              Stokes_sample[:, :, 0]
        DOLP = np.sqrt(Stokes_sample[:, :, 1]**2 + Stokes_sample[:, :, 2]**2) / Stokes_sample[:, :, 0]
        DOCP = np.abs(Stokes_sample[:, :, 3] / Stokes_sample[:, :, 0])
        nonpol = 1 - DOP

        POL = np.zeros([det_x, det_y, 4])
        POL[:, :, 0] = nonpol
        POL[:, :, 0] = DOP
        POL[:, :, 0] = DOLP
        POL[:, :, 0] = DOCP

        # TODO get electrical fields
        # see below --> postpone ask Toon

def readImageData(self):
    # TODO read image: read multiple ebeam pos?
    f = h5.File("20180814-6polwithtxt.h5")
    keys = ["Acquisition%d" % i for i in range(2, 8)]
    det_data = []
    pol_pos = []
    for key in f.keys():
        if key in keys:
            det_data.append(np.squeeze(f[key]["ImageData"]['Image']))
            pol_pos.append(f[key]["PhysicalData"]["Polarization"].value[0])

    det_data = np.array(det_data)
    det_data = np.swapaxes(det_data, 0, 2)
    return det_data

def StokesCalc_CCD(self, det_data, pol_pos=None):
    """
    det_data (array): 6 input images (detector data) recorded with different polarization analyzer positions.
                    Shape is (x, y, 6).
    pol_pos (list or None): list with polarization analyzer positions. If None, use default order of measurements.
    Function that calculates stokes parameters in the detector plane. Needs to be adapted to order of measurements
    of polarization analyzer position.
    Default is (for testcase): 0:vertical, 1:horizontal, 2: 135 (negdiag) 3: 45 (posdiag)  4:right handed 5:left handed.
    return (array): The 4 Stokes params in detector plane. Shape is (x, y, 4).
    """

    pos_dict = {}
    if pol_pos:
        for index, pos in enumerate(pol_pos):
            pos_dict[pos] = index
    else:  # default
        pos_dict["vertical"] = 0
        pos_dict["horizontal"] = 1
        pos_dict["negdiag"] = 2
        pos_dict["posdiag"] = 3
        pos_dict["rhc"] = 4
        pos_dict["lhc"] = 5

    # TODO get choices from yaml? now hardcoded pos. or use MD? MD_POL_HORIZONTAL etc.
    S0 = det_data[:, :, pos_dict["horizontal"]] + det_data[:, :, pos_dict["vertical"]]
    S1 = det_data[:, :, pos_dict["horizontal"]] - det_data[:, :, pos_dict["vertical"]]
    S2 = det_data[:, :, pos_dict["posdiag"]] - det_data[:, :, pos_dict["negdiag"]]
    S3 = det_data[:, :, pos_dict["lhc"]] - det_data[:, :, pos_dict["rhc"]]

    stokes = np.dstack((S0, S1, S2, S3))

    return stokes


def ARmask_calc(self, theta1, phi1, holein=1):
    # solve equality ar^2-1/(4a)=x.
    # c=1/(2*(a*cos(phi)*sin(theta)+sqrt(a^2*(cos(theta)^2+(cos(phi)^2+sin(phi)^
    # 2)*sin(theta)^2)))); The cos(phi)^2+sin(phi)^2=1 so we can omit that. Than
    # we have cos(theta)^2+sin(theta)^2 which also drops out. That leaves the
    # square root of a^2

    a = 0.1
    xcut = 10.75

    ##thetacutoff can actually be calculated
    holesize = 0.6
    holeheight = np.sqrt(2.5 / a)
    #    thetacutoffhole=np.arctan(holesize/(2*holeheight))*180/np.pi
    thetacutoffhole = 4
    dfoc = 0.5
    phi, theta = np.meshgrid(phi1, theta1)
    c = 1. / (2 * (a * np.cos(phi) * np.sin(theta) + a))

    z = np.cos(theta) * c
    x = np.sin(theta) * np.cos(phi) * c  # -1/(4.*a)
    # y=np.sin(theta)*np.sin(phi)*c

    mask = np.ones([np.size(theta, 0), np.size(theta, 1)])
    if holein == 1:
        mask[(-x > xcut) | (theta < (thetacutoffhole * np.pi / 180)) | (z < dfoc)] = 0

    else:
        mask[(-x > xcut) | (z < dfoc)] = 0

    return mask

# TODO generate inverse mumat for all wavelength?

def MuellerCor_DetToSample_Inv(stokes_det, mumat_inv):
    """Apply the Mueller correction to go from detector plane to sample plane.
    This requires the inverse of the Mueller matrix.
    stokes_det: stokes params for detector plane. Shape is (x, y, 4).
    mumat: mueller matrix for specific wavelength. Shape is (x, y, 4, 4)
    return: stokes params for sample plane. Shape is (x, y, 4, 4)
    """

    stokes_sample = np.empty([stokes_det.shape[0], stokes_det.shape[1], 4], dtype="float64")

    for ii in range(stokes_sample.shape[0]):
        for jj in range(stokes_sample.shape[1]):
            stokes_sample[ii, jj, :] = np.dot(mumat_inv[ii, jj, :, :], stokes_det[ii, jj, :])
    #           Stokes_sample1[ii,jj,:] = np.linalg.lstsq(mumat[ii,jj,:,:],Stokes_det1[ii,jj,:])[0]
    #        # these approaches seem equivalent. Check this. First one seems more straightforward.

    return stokes_sample