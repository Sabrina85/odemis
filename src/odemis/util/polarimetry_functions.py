# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 07:35:04 2018

@author: coene
"""
import numpy as np
import calctools as ct
import matplotlib.pyplot as plt


def CalcMuellerMatrix(phi, theta, nsurf):

    # function that calculates mueller matrix for all emission angles simultaneously.
    a = 0.1
    # input Stokes vector (fully p polarized)
   
    Sin1 = np.array([1,1,0,0])
    Sin2 = np.array([1,-1,0,0])
    
#    Sin1 = np.tile(np.array([1,1,0,0]),(theta.shape[0],theta.shape[1],1))
    # input Stokes vector (fully s polarized)
#    Sin2 = np.tile(np.array([1,-1,0,0]),(theta.shape[0],theta.shape[1],1))
      

    # Field output in detection plane for E_theta input
    E1 = mueller_matrixinput(Sin1,phi,theta,a,nsurf)
    # Field output in detection plane for E_phi input
    E2 = mueller_matrixinput(Sin2,phi,theta,a,nsurf)

    # resulting y and z field components in CCD plane for fully p and s polarized emitters. x should be zero so can be ignored
    Etz = E1[:,:,2]
    Ety = E1[:,:,1]
    Efz = E2[:,:,2]
    Efy = E2[:,:,1]
    
    # Mueller matrix elements as specified by Rodriguez-Herrera and Bruce, Optical Engineering 45, 053602 (2006)
    m11 = (Ety*np.conj(Ety) + Etz*np.conj(Etz) + Efy*np.conj(Efy) + Efz*np.conj(Efz))/2
    m12 = (Ety*np.conj(Ety) + Etz*np.conj(Etz) - Efy*np.conj(Efy) - Efz*np.conj(Efz))/2
    m13 = np.real(Ety*np.conj(Efy)) + np.real(Etz*np.conj(Efz))
    m14 = np.imag(Etz*np.conj(Efz)) + np.imag(Ety*np.conj(Efy))

    m21 = (Ety*np.conj(Ety) - Etz*np.conj(Etz) + Efy*np.conj(Efy) - Efz*np.conj(Efz))/2
    m22 = (Ety*np.conj(Ety) - Etz*np.conj(Etz) - Efy*np.conj(Efy) + Efz*np.conj(Efz))/2
    m23 = np.real(Ety*np.conj(Efy)) - np.real(Etz*np.conj(Efz))
    m24 = np.imag(Ety*np.conj(Efy)) - np.imag(Etz*np.conj(Efz))

    m31 = np.real(Ety*np.conj(Etz)) + np.real(Efy*np.conj(Efz))
    m32 = np.real(Ety*np.conj(Etz)) - np.real(Efy*np.conj(Efz))
    m33 = np.real(Ety*np.conj(Efz)) + np.real(Efy*np.conj(Etz))
    m34 = np.imag(Ety*np.conj(Efz)) - np.imag(Efy*np.conj(Etz))

    m41 = -np.imag(Ety*np.conj(Etz)) - np.imag(Efy*np.conj(Efz))
    m42 = -np.imag(Ety*np.conj(Etz)) + np.imag(Efy*np.conj(Efz))
    m43 = -np.imag(Ety*np.conj(Efz)) - np.imag(Efy*np.conj(Etz))
    m44 = np.real(Ety*np.conj(Efz)) - np.real(Efy*np.conj(Etz))

    # organize matrices in one master array
    muellermatcol1 = np.dstack((m11,m21,m31,m41))
    muellermatcol2 = np.dstack((m12,m22,m32,m42))
    muellermatcol3 = np.dstack((m13,m23,m33,m43))
    muellermatcol4 = np.dstack((m14,m24,m34,m44))
    
    # add np.real to change datatype. Complex part is already 0 so does not change matrix
    muellermattot = np.real(np.stack((muellermatcol1,muellermatcol2,muellermatcol3,muellermatcol4),axis=3))
    
    return muellermattot


def mueller_matrixinput(stokes,phi,theta,a,nsurf):
    
    #function that returns input for mueller matrix calculation. In particular it calculates how the sample plane electric field components are mapped to the detector plane fields. 
    #calculate electric field from     
    efield = StokesVecToE(stokes)
       
    phi1 = -(phi+np.pi)
    #change of phi coordinate to rotate reference frame properly with respect to mirror
    
    #unit vector specifying light emission direction
    emdir = np.dstack(((np.sin(theta)*np.cos(phi1)),(np.sin(theta)*np.sin(phi1)),(np.cos(theta))))

    # S and p emission direction in spherical coordinates. p is theta gradient of the
    # spherical coordinates. s is the phi gradient. Question here is why the theta term is not included here. Try both
    pdir = np.dstack(((np.cos(theta)*np.cos(phi1)),(np.cos(theta)*np.sin(phi1)),(-np.sin(theta))))
    sdir = -np.dstack(((-np.sin(phi1)), (np.cos(phi1)), np.zeros(np.shape(phi1))))
    # For some reason adding a minus sign to sdir fixes problems for in-plane dipole analysis. 
    #Check that this actually doesn't cause problems in polarimetry analysis
    
    #calculaion of parabola position, surface normal, and fresnel emission
    ppos = parabolaposP(emdir,phi,theta,a)
    normal = parabolanormalP(ppos,a)
    rs, rp = fresnellreflP(normal,emdir,nsurf)
    #s-direction in mirror surface reference frame calculated from cross product between surface normal and emission direction
    S = NormalizeVecP(np.cross(normal,emdir,axis=2))

    Ein = efield[0]*pdir + efield[1]*sdir
    Es1 = np.transpose(np.tile(matrix_dot(Ein,S),(3,1,1)),(1,2,0))*S
    Ep1 = Ein - Es1
    
    #calculate p-component, taking into account direction change of p-polarization
    Epnorm = NormalizeVecP(Ep1)
    EpAmp = np.transpose(np.tile(np.sqrt(matrix_dot(Ep1,Ep1)),(3,1,1)),(1,2,0))
    Ep = (2*np.transpose(np.tile(matrix_dot(normal,Epnorm),(3,1,1)),(1,2,0))*normal - Epnorm)*EpAmp*rp
    Es = Es1*rs
    #final electric field components
    Etot = Ep + Es

    return Etot

def MuellerCor_SampleToDet(stokes_sample, mumat):
    #calculate stokes vector in detector plane based on a stokes vector in the sample plane. For this correction we use the Mueller matrix as is
    
    #allocate memory for stokes vectors detector
    stokes_det = np.empty([stokes_sample.shape[0],stokes_sample.shape[1],4],dtype="float64")
    
    #this can probably be parallelized
    for ii in range(stokes_sample.shape[0]):
        for jj in range(stokes_sample.shape[1]):   
            stokes_det[ii,jj,:] = np.dot(mumat[ii,jj,:,:],stokes_sample[ii,jj,:])
    
    return stokes_det


# TODO lookuptable
def MuellerCor_DetToSample(stokes_det, mumat):
    """Apply the Mueller correction to go from detector plane to sample plane.
    This requires the inverse of the Mueller matrix.
    stokes_det: stokes params for detector plane. Shape is (x, y, 4).
    mumat: mueller matrix for specific wavelength. Shape is (x, y, 4, 4)
    return: stokes params for sample plane. Shape is (x, y, 4, 4)
    """

    stokes_sample = np.empty([stokes_det.shape[0], stokes_det.shape[1], 4], dtype="float64")

    # TODO get rid of for loops? use numpy?
    for ii in range(stokes_sample.shape[0]):
        for jj in range(stokes_sample.shape[1]):
            stokes_sample[ii, jj, :] = np.dot(np.linalg.inv(mumat[ii, jj, :, :]), stokes_det[ii, jj, :])
#           Stokes_sample1[ii,jj,:] = np.linalg.lstsq(mumat[ii,jj,:,:],Stokes_det1[ii,jj,:])[0]
#        # these approaches seem equivalent. Check this. First one seems more straightforward.

    return stokes_sample

def MuellerCor_DetToSample_Inv(stokes_det, mumat_inv):
    """Apply the Mueller correction to go from detector plane to sample plane.
    This requires the inverse of the Mueller matrix.
    stokes_det: stokes params for detector plane. Shape is (x, y, 4).
    mumat: mueller matrix for specific wavelength. Shape is (x, y, 4, 4)
    return: stokes params for sample plane. Shape is (x, y, 4, 4)
    """

    stokes_sample = np.empty([stokes_det.shape[0], stokes_det.shape[1], 4], dtype="float64")

    # TODO get rid of for loops? use numpy?
    for ii in range(stokes_sample.shape[0]):
        for jj in range(stokes_sample.shape[1]):
            stokes_sample[ii, jj, :] = np.dot(mumat_inv[ii, jj, :, :], stokes_det[ii, jj, :])
#           Stokes_sample1[ii,jj,:] = np.linalg.lstsq(mumat[ii,jj,:,:],Stokes_det1[ii,jj,:])[0]
#        # these approaches seem equivalent. Check this. First one seems more straightforward.

    return stokes_sample

def StokesVecToEpar(stokes):
    #function to convert stokes parameters as measured on the CCD/CMOS to electric fields
    #amplitudes of Ey and Ez
    E1a = np.sqrt((stokes[:,:,0]+stokes[:,:,1])/2)
    E2a = np.sqrt((stokes[:,:,0]-stokes[:,:,1])/2)

    #phase difference is equal to phasey-phasez, check whether this works properly. field amplitudes should be absolute though
    phasedif = np.arctan2(stokes[:,:,3],stokes[:,:,2])

    #phasedifference is global and should not matter for the result. We should
    #check this. In this case we set phasez to zero
    E1 = E1a*np.exp(1j*phasedif)
    E2 = E2a*np.exp(1j*0)
    efield = np.dstack((E1,E2))
        
    return efield

def StokesVecToE(stokes):
    # function to convert stokes parameters as measured on the CCD/CMOS to electric fields
    # amplitudes of Ey and Ez
    E1a = np.sqrt((stokes[0]+stokes[1])/2)
    E2a = np.sqrt((stokes[0]-stokes[1])/2)

    # phase difference is equal to phasey-phasez, check whether this works properly. field amplitudes should be absolute though
    phasedif = np.arctan2(stokes[3],stokes[2])

    # phasedifference is global and should not matter for the result. We should
    # check this. In this case we set phasez to zero
    E1 = E1a*np.exp(1j*phasedif)
    E2 = E2a*np.exp(1j*0)
    efield = np.hstack((E1,E2))
        
    return efield


def StokesCalc_CCD(det_data, pol_pos=None):
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


def EtoStokesPar(Efields):
    
    # Calculate stokes vector from two orthogonal field components for an array of electric fields. 
    E1 = Efields[:, :, 0]
    E2 = Efields[:, :, 1]
    #electric field phases computed from real and imaginary parts
    phase1 = np.arctan2(np.imag(E1), np.real(E1))
    phase2 = np.arctan2(np.imag(E2), np.real(E2))
    
    #Calculate stokes vectors from electric fields
    S0 = np.abs(E1)**2+np.abs(E2)**2
    S1 = np.abs(E1)**2-np.abs(E2)**2
    S2 = 2*np.abs(E1)*np.abs(E2)*np.cos(phase1-phase2)
    S3 = 2*np.abs(E1)*np.abs(E2)*np.sin(phase1-phase2)
    # S2 = 2*real(E1.*conj(E2));
    # S3 = -2*imag(E1.*conj(E2));
    # phase difference is equal to phasey-phasez
    sdata = np.dstack((S0, S1, S2, S3))
    
    return sdata

def EtoStokes(Efields):
    
    # Calculate stokes vector from two orthogonal field components. 
    E1 = Efields[0]
    E2 = Efields[1]

    phase1 = np.arctan2(np.imag(E1),np.real(E1))
    phase2 = np.arctan2(np.imag(E2),np.real(E2))

    S0 = np.abs(E1)**2+np.abs(E2)**2
    S1 = np.abs(E1)**2-np.abs(E2)**2
    S2 = 2*np.abs(E1)*np.abs(E2)*np.cos(phase1-phase2)
    S3 = 2*np.abs(E1)*np.abs(E2)*np.sin(phase1-phase2)
    # S2 = 2*real(E1.*conj(E2));
    # S3 = -2*imag(E1.*conj(E2));
    #phase difference is equal to phasey-phasez
    sdata = np.hstack((S0,S1,S2,S3))
    
    return sdata


def parabolaposP(emission,phi,theta,a):
    
    # calculate position of paraboloid mirror surface in space for a given emission direction and a
    
    #solve equality ar^2-1/(4a)=x. 
    #c=1/(2*(a*cos(phi)*sin(theta)+sqrt(a^2*(cos(theta)^2+(cos(phi)^2+sin(phi)^2)*sin(theta)^2)))); The cos(phi)^2+sin(phi)^2=1 so we can omit that. Than
    #we have cos(theta)^2+sin(theta)^2 which also drops out. That leaves the
    #square root of a^2
    
    c = np.transpose(np.tile(1./(2*(a*np.cos(phi)*np.sin(theta)+a)),(3,1,1)),(1,2,0))
    par = emission*c

    return par

def parabolanormalP(ppos,a):
    #function that calculates vectorial parabola normal (in parallel for many angles)
    #definitions of r and theta for the parabola. 
    r = np.sqrt(ppos[:,:,1]**2+ppos[:,:,2]**2)
    theta1 = np.arccos(ppos[:,:,1]/r)

    #To calculate the surface normal we calculate the gradient of the radius and of theta by symbolic differentiation of the parabola formula.
    # The parabola formula looks as follows r=[a*r^2 r*cos(theta1)  r*sin(theta1)].
    # We pick x to be along the optical axis of the parabola and  y transverse
    # to it. Z is the dimension perpendicular to the sample

    Gradr = np.dstack(((2*a*r),(np.cos(theta1)),(np.sin(theta1))))
    Gradtheta = np.dstack(((np.zeros(np.shape(r))),(-r*np.sin(theta1)),(r*np.cos(theta1))))
    #compute surface normal from cross product of gradients
    Normal = np.cross(Gradr,Gradtheta,axis=2)
    Normal = NormalizeVecP(Normal)
    #Par = Gradr+Gradtheta
    
    return Normal

def NormalizeVecP(vector_in):

    #Parallel unit vector normalization
    dotm = np.sqrt(matrix_dot(vector_in,vector_in))
    dotm1 = np.transpose(np.tile(dotm,(3,1,1)),(1,2,0))
    vector_out = vector_in/dotm1
    #remove NaN's from array, which come from the hard coded zeros in the CL data
    vector_out[np.isnan(vector_out)] = 0
 
    return vector_out


def matrix_dot(a,b):
    
    #rather trivial but still a nice function to have
    dotmat = np.sum(a.conj()*b,2)
       
    return dotmat

    
def fresnellreflP(normal,emin,nsurf):
    
    #calculate reflection coefficients for a given paraboloid normal and emission direction
    cosi = np.transpose(np.tile(-matrix_dot(emin,normal),(3,1,1)),(1,2,0)) #Cosine of angle of incidence
    ai = np.arccos(cosi) #angle of incidence
    
    nsurfcost = np.sqrt(nsurf**2 - np.sin(ai)**2)# Refractive index of surface multiplied by the cosine of the angle of the refracted ray
        # Fresnel relative amplitudes of s and p E-field components. (Jackson %
        # p. 305 & 306). The n1 falls out because this is one for vacuum
        
    rs = (cosi - nsurfcost) / (cosi + nsurfcost)
    rp = ((nsurf**2)*cosi - nsurfcost) / ((nsurf**2)*cosi + nsurfcost)
    
    #Direction of the reflected beam. Not necessary for calculation
    #Refl = -(2*np.transpose(np.tile(matrix_dot(normal,emin),(3,1,1)),(1,2,0))*normal-emin)

    return rs,rp

def Mueller_Analyzer(S_in,QWPangle,LPangle):

    # code that calculates the effect of going through an polarization analyzer
    # on the Stokes Vector
#    QWPangle = QWPangle*np.pi/180 
#    LPangle = LPangle*np.pi/180

    # mueller matrix for linear polarizer
    LP = 0.5*(np.array([[1, np.cos(2*LPangle), np.sin(2*LPangle), 0],
                        [np.cos(2*LPangle), np.cos(2*LPangle)**2, np.sin(2*LPangle)*np.cos(2*LPangle), 0],
                        [np.sin(2*LPangle), np.sin(2*LPangle)*np.cos(2*LPangle), np.sin(2*LPangle)**2, 0],
                        [0, 0, 0, 0]]))

    #mueller matrix for quarter wave plate. Can be derived from the general Mueller matrix for a retarder.
    QWP = np.array([[1, 0, 0, 0],
                    [0, np.cos(2*QWPangle)**2, np.cos(2*QWPangle)*np.sin(2*QWPangle), np.sin(2*QWPangle)],
                    [0, np.cos(2*QWPangle)*np.sin(2*QWPangle), np.sin(2*QWPangle)**2, -np.cos(2*QWPangle)],
                    [0, -np.sin(2*QWPangle), np.cos(2*QWPangle), 0]])

    Sout = np.empty(S_in.shape)
    
    # this can probably be compressed but it works
    for i in range(S_in.shape[0]):
        for j in range(S_in.shape[1]):
     
            # Let the polarizer mueller matrices operate on the Stokes vector
            Sout[i, j, :] = np.dot(LP, np.dot(QWP, S_in[i, j, :]))


    return Sout


def CarToSpher(Efields):
    
    #function to project cartesian fields onto polar coordinates in a
    #parallelized way. This way can also be implemented in the correction code

    phi1 = np.linspace(0, 2*np.pi,Efields.shape[1],dtype="complex128")
    theta1 = np.linspace(0, np.pi/2,Efields.shape[0], dtype="complex128")
    phi, theta = np.meshgrid(phi1, theta1)
    
#    phi1 = (phi-np.pi)
    
    #Calcualte zenithal and azimuthal unit vector, check azimuthal unit vector
    thetagrad = np.dstack((np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi),-np.sin(theta)))
    phigrad = np.dstack((-np.sin(phi),np.cos(phi), np.zeros(phi.shape)))

    #Project cartesian fields onto these unitvectors
    Etheta = matrix_dot(Efields,thetagrad)
    Ephi = matrix_dot(Efields,phigrad)
    ETF = np.dstack((Etheta,Ephi))
    
    return ETF
    
    
def SpherToCar(Efields):
    
    #function to project spherical fields onto cartesian fields in a
    #parallelized way. 

    phi1 = np.linspace(0,2*np.pi,Efields.shape[1])
    theta1 = np.linspace(0,np.pi/2,Efields.shape[0])
    phi,theta = np.meshgrid(phi1,theta1)

    #Calcualte zenithal and azimuthal unit vector
    thetagrad = np.dstack((np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi),-np.sin(theta)))
    phigrad = np.dstack((-np.sin(phi),np.cos(phi), np.zeros(phi.shape)))

    Et = np.transpose(np.tile(Efields[:,:,0],(3,1,1)),(1,2,0))
    Ef = np.transpose(np.tile(Efields[:,:,1],(3,1,1)),(1,2,0))

    #Project cartesian fields onto these unitvectors
    Etheta = Et*thetagrad
    Ephi= Ef*phigrad
    Exyz = Etheta + Ephi
    
    return Exyz   

def DegPol(stokes):
    
    # degree of polarization, degree of circular polarization, and unpolarized part. 
    # Also return polarized and unpolarized intensity distributions
    DOP = np.sqrt(stokes[:,:,1]**2+stokes[:,:,2]**2+stokes[:,:,3]**2)/stokes[:,:,0]
    DOCP = stokes[:,:,3]/stokes[:,:,0]
    UP = 1 - DOP
    I_pol = stokes[:,:,0]*DOP
    I_unpol = stokes[:,:,0] * UP
    
    return DOP, DOCP, UP, I_pol, I_unpol

def findelement(list,value):
   
    element = np.argmin(np.abs(list-value))
    
    return element  


def al_opticalconstants(wavelength, oset=1):
    """return optical constants of Al for a particular wavelength.
    Choice between two tabulated datasets (Rakic and McPeak from refractiveindex.info).
    Datasets are verified. Mcpeak is chosen as default as it is a bit more smooth.
    """
    if oset == 1:
    
        mcpeak_n = np.loadtxt('McPeak_n.csv', delimiter=',', skiprows=1)
        mcpeak_k = np.loadtxt('McPeak_k.csv', delimiter=',', skiprows=1)
        # wavelength in SI units.
        mcpeak_wl = mcpeak_n[:, 0]*1e-6
        mcpeak_ntot = mcpeak_n[:, 1]+1j*mcpeak_k[:, 1]
        mcpeak_interpwl = np.linspace(mcpeak_wl.min(), mcpeak_wl.max(), 2000)
        mcpeak_ntot_interp = np.interp(mcpeak_interpwl, mcpeak_wl, np.real(mcpeak_ntot)) \
                             + 1j*np.interp(mcpeak_interpwl, mcpeak_wl, np.imag(mcpeak_ntot))
        
        if (wavelength < mcpeak_wl.min()) | (wavelength > mcpeak_wl.max()):
            print("Warning: specified wavelength falls outside optical data range")
        
        n = mcpeak_ntot_interp[ct.findelement(mcpeak_interpwl,wavelength)] 
        eps = n**2
    
    if oset == 2:
    
        rakic_n = np.loadtxt('Rakic_n.csv', delimiter=',', skiprows=1)
        rakic_k = np.loadtxt('Rakic_k.csv', delimiter=',', skiprows=1)
        # wavelength in SI units
        rakic_wl = rakic_n[:,0]*1e-6
        rakic_ntot = rakic_n[:,1]+1j*rakic_k[:,1]
        # interpolate data
        rakic_interpwl = np.linspace(rakic_wl.min(), rakic_wl.max(), 2000)
        rakic_ntot_interp = np.interp(rakic_interpwl, rakic_wl, np.real(rakic_ntot)) \
                            + 1j*np.interp(rakic_interpwl, rakic_wl, np.imag(rakic_ntot))
      
        if (wavelength < rakic_wl.min()) | (wavelength > rakic_wl.max()):
            print("Warning: specified wavelength falls outside optical data range")
        
        n = rakic_ntot_interp[ct.findelement(rakic_interpwl, wavelength)]
        eps = n**2
    
    return n, eps

   
def al_opticalconstants_range(wavelength,oset=1):
 
    #return optical constants for a range of wavelengths. Choice between two tabulated datasets (Rakic and McPeak from refractiveindex.info). Datasets are verified. Mcpeak is chosen as default as it is a bit more smooth
    if oset==1:
    
        mcpeak_n = np.loadtxt('McPeak_n.csv',delimiter=',',skiprows=1)
        mcpeak_k = np.loadtxt('McPeak_k.csv',delimiter=',',skiprows=1)
        #wavelength in SI units. 
        mcpeak_wl = mcpeak_n[:,0]*1e-6
        mcpeak_ntot = mcpeak_n[:,1]+1j*mcpeak_k[:,1]
        ntot_interp = np.interp(wavelength,mcpeak_wl,np.real(mcpeak_ntot))+1j*np.interp(wavelength,mcpeak_wl,np.imag(mcpeak_ntot))
        
        if (wavelength.min() < mcpeak_wl.min()) | (wavelength.max() > mcpeak_wl.max()):
            print("Warning: specified wavelength falls outside optical data range")
        

        eps = ntot_interp**2
    
    if oset==2:
    
        rakic_n = np.loadtxt('Rakic_n.csv',delimiter=',',skiprows=1)
        rakic_k = np.loadtxt('Rakic_k.csv',delimiter=',',skiprows=1)
        #wavelength in SI units
        rakic_wl = rakic_n[:,0]*1e-6
        rakic_ntot = rakic_n[:,1]+1j*rakic_k[:,1]
        #interpolate data
        ntot_interp = np.interp(wavelength,rakic_wl,np.real(rakic_ntot))+1j*np.interp(wavelength,rakic_wl,np.imag(rakic_ntot))
      
        if (wavelength.min() < rakic_wl.min()) | (wavelength.max() > rakic_wl.max()):
            print("Warning: specified wavelength falls outside optical data range")
        
   
        eps = ntot_interp**2
    
    return ntot_interp, eps

  