# -*- coding: utf-8 -*-
"""
Created on Thu Mar 03 10:01:53 2016

@author: Toon
"""
from __future__ import division
import numpy as np
from scipy.spatial import Delaunay as DelaunayTriangulation
from scipy.interpolate import LinearNDInterpolator
import calctools as ct
import matplotlib.pyplot as plt

#background correction of AR pattern, can also be used for spec

def bgcor(data,bg):
    
    dataAR=data[0][0]
    bgAR=bg[0][0]
    ndimbg=np.ndim(bgAR)
    ndimdata=np.ndim(dataAR)
    
    
    if ndimbg==4 and ndimdata==4:
        
        bgav=np.sum(np.sum(bgAR,0),0)/(np.size(bgAR,0)*np.size(bgAR,1))
        bgavtot=np.tile(bgav,(np.size(dataAR,0),np.size(dataAR,1),1,1)) 
        data[0][0]=dataAR-bgavtot

    return data


def findangle(harsize,binning,deff,a,xcut,dfoc,thetacutoffhole=5):
    
    y=np.linspace(-harsize[1]/binning,harsize[1]/binning,harsize[1]*2/binning)*deff
    z=np.linspace(-harsize[0]/binning,harsize[0]/binning,harsize[0]*2/binning)*deff+5.#dfoc
#    z=np.linspace(-harsize[0]/binning,harsize[0]/binning,harsize[0]*2/binning)*deff+dfoc
#    y=np.linspace(-harsize,harsize,harsize*2/binning)*pixelsize
#    z=(np.linspace(-harsize,harsize,harsize*2/binning)+dfoc/pixelsize)*pixelsize

    y1,z1=np.meshgrid(y,z)
    r=np.sqrt(y1**2+z1**2)
    x=a*r**2-1/(4*a)

    #angular conversion
    theta = np.arccos(z1/np.sqrt(x**2+r**2))
    phi = np.arctan2(y1,-x)
    phi[phi<0]=phi[phi<0]+2*np.pi
#    phi=phi+np.pi
#    
#    plt.figure()
#    plt.imshow(z1)
##    
#    plt.figure()
#    plt.imshow(phi)
    # solid angle correction    
    omega=(deff**2)*(2*a*r**2 - x)/(x**2+r**2)**(3./2.)
    # Old formulation
#    omega = pixelsize**2 * binning**2 * (2*a*r**2 - x)/(x**2+r**2)**(3/2)
#    plt.figure()
#    plt.imshow(omega)
    #mask, not necessary at this point but easy to compute. Can be used to check Mask overlay
    mask=np.ones((harsize[0]*2/binning,harsize[1]*2/binning))
    mask[(x > xcut) | (theta < (thetacutoffhole*np.pi/180)) | (z1 < dfoc)]=0
    
    # highest cut position
    zcut = 1.83
    #max xpos at given zcut
    xpos = 2.5-0.1*zcut**2
#    mask[((phi < np.pi/2)  & (z1 < zcut*(np.abs(x)/xpos))) | ((phi > 3./2 *np.pi) & (z1 < zcut*(np.abs(x)/xpos)))]=0    
    
    
#    plt.figure()
#    plt.imshow(mask)
    datatot=[theta,phi,omega,mask]
    
    return datatot


def centerARdata(data,deff,harsize,binning,polepos):
    
    #pole positions as stored by odemis
    ARdata=data[0][0]
    ndims=np.ndim(ARdata)
#    polepos=data[0][3]
    #polepos in odemis is defined at hole. Convert this to mirror apex such that it is consistent witht the rest of the code
    polepos[0]=polepos[0]-np.round(4.5/deff)
    shifty= np.int(harsize[0]/binning-polepos[0])
    shiftx= np.int(harsize[1]/binning-polepos[1])
    
#    print(shifty)
#    print(shiftx)
   
    if ndims==4: 
      #center data such that angular conversion is done properly
      ARdata=np.roll(np.roll(ARdata,shifty,axis=2),shiftx,axis=3)
      
    
    # There was a fliplr in there before. It might not be necessary to use that
    data[0][0]=ARdata

    return data
    

def centerARdataTiff(data,deff,harsize,binning,polepos):
    
    #pole positions as stored by odemis
    ARdata=data[0][0]
    ndims=np.ndim(ARdata)
    #polepos in odemis is defined at hole. Convert this to mirror apex such that it is consistent witht the rest of the code
    shifty = np.int(harsize[0]/binning-polepos[0])
    shiftx = np.int(harsize[1]/binning-polepos[1])
    
       #center data such that angular conversion is done properly
    ARdata=np.roll(np.roll(ARdata,shifty,axis=0),shiftx,axis=1)
      
    
    # There was a fliplr in there before. It might not be necessary to use that
    data[0][0]=ARdata

    return data    


#still need to adjust the code for different size data, for now only 2D AR scans supported
def interp_data(data,ARmatrix,a,xcut,dfoc):
    
    
    theta=ARmatrix[0]
    phi=ARmatrix[1]
    omega=ARmatrix[2]
    #can be used for importing different size datasets, not implemented yet
    ndimdata=np.ndim(data[0][0]) 
    #datasize, can be increased for high-res. datasets
    phisize=360
    thetasize=90
    datasize1=np.shape(data[0][0])[0]
    datasize2=np.shape(data[0][0])[1]
    data_converted=np.zeros((datasize1, datasize2, thetasize, phisize))
    
    #compute new mask
    phi1=np.linspace(0, 2*np.pi, phisize)
    theta1=np.linspace(0, np.pi/2, thetasize)
    phi2,theta2=np.meshgrid(phi1, theta1)
    
    #length vector
    c=(2*(a*np.cos(phi2)*np.sin(theta2)+a))**-1
    #position vectors
    x=-np.sin(theta2)*np.cos(phi2)*c 
#    y=np.sin(theta2)*np.sin(phi2)*c
    z=np.cos(theta2)*c
    
    mask=np.ones((thetasize,phisize))
    mask[(x > xcut) | (theta2 < (4*np.pi/180)) | (z < dfoc)]=0   
    
    # can probably choose a selection here to speed up interpolation. This is a silly fix but it works. Prevents exptrapolation which leads to errors
    theta=np.tile(theta,(1,3))
    phi=np.append(np.append(phi-2*np.pi,phi,axis=1),phi+2*np.pi,axis=1)

    for ii in range(0, datasize1):
        for jj in range(0, datasize2):
    
         ARdata=np.squeeze(data[0][0][ii,jj,:,:])/omega
         ARdata=np.tile(ARdata,(1,3))

         triang = DelaunayTriangulation(np.array([theta, phi]).T)
         # create interpolation object
         interp = LinearNDInterpolator(triang, ARdata.flat)
         # create grid of positions for interpolation
         xi, yi = np.meshgrid(np.linspace(0, np.pi / 2, thetasize),
                                 np.linspace(0, 2 * np.pi, phisize))
         # interpolate
         qz = interp(xi, yi)
         # qz = qz.swapaxes(0, 1)[:, ::-1]  # rotate by 90°
         # qz[np.isnan(qz)] = 0  # remove NaNs created during interpolation
         # assert np.all(
         #     qz > -1)  # there should be no negative values, some very small due to interpolation are possible
         # qz[qz < 0] = 0

         data_converted[ii,jj,:,:]=qz*mask        
         
    data[0][3]=data_converted
    data[0][4]=mask
    data[0][5]=ARmatrix
    
    return data


def interp_data_tiff(data,ARmatrix,a,xcut,dfoc):
    
    theta=ARmatrix[0]
    phi=ARmatrix[1]
    omega=ARmatrix[2]
 #can be used for importing different size datasets, not implemented yet
#datasize, can be increased for high-res. datasets
    phisize=1001
    thetasize=1001
#  
    data_converted=np.zeros((thetasize,phisize))    
#    
#    #compute new mask
    phi1=np.linspace(0,2*np.pi,phisize)
    theta1=np.linspace(0,np.pi/2,thetasize)  
    phi2,theta2=np.meshgrid(phi1,theta1)
    
#length vector
    c=(2*(a*np.cos(phi2)*np.sin(theta2)+a))**-1
#position vectors
    x=-np.sin(theta2)*np.cos(phi2)*c 
#    y=np.sin(theta2)*np.sin(phi2)*c
    z = np.cos(theta2)*c
    mask = np.ones((thetasize,phisize))
    mask[(x > xcut) | (theta2 < (5*np.pi/180)) | (z < dfoc)]=0   
         
        
    xsize = np.size(phi,1) 
### can probably choose a selection here to speed up interpolation. This is a silly fix but it works. Prevents exptrapolation which leads to errors
#    theta=np.tile(theta,(1,3))
#    phi=np.append(np.append(phi-np.pi,phi,axis=1),phi,axis=1)
    
    #extension
    ext = 0
#    phi=phi-np.pi
#    phi=phi[:,(xsize-ext):(2*xsize+ext)]
#    theta=theta[:,(xsize-ext):(2*xsize+ext)]
##

    ARdata = data[0][0]/omega
    
    #still ugly but it works
    theta=np.tile(theta,(1,3))
    phi=np.append(np.append(phi-2*np.pi,phi,axis=1),phi+2*np.pi,axis=1)
    ARdata=np.tile(ARdata,(1,3))
   

    print(np.shape)
 #this fails for some reason when defining hole as center, investigate more
    triang = DelaunayTriangulation(phi.flat,theta.flat)
#    interp = triang.nn_interpolator(ARdata.flat, default_value=0)
    #use extrapolator to bridge last step. Problem arises because phi as calculated in find_angle does not go entirely to 2pi
    interp = triang.linear_interpolator(ARdata.flat, default_value=2000)
    qz = interp[0:np.pi/2:complex(0,thetasize),0:2*np.pi:complex(0,phisize)]        
    data_converted = qz*mask     

    return data_converted
    
    
def thetacut180(data,phicenter,width):
    
    # make angular cross cut at specified angle and angular width
    phisize=np.size(data,1)
    phi=np.linspace(0,360,phisize)
    element1=ct.findelement(phi,phicenter)
    #number of pixels per degree times the number of degrees over two
    rangephi=np.int((phisize/360)*(width/2))    
    
    if phicenter < 180:    
        
      element2=np.int(ct.findelement(phi,phicenter+180))
      
    else:
        
      element2=np.int(ct.findelement(phi,phicenter-180))
    
    print(element1,element2,rangephi)    
    
    #condition when phi is close to zero
    if (element1 > rangephi) & (element1 < (phisize-rangephi)): 
        
        cut1 = (np.sum(data[:,(element1-rangephi):(element1+rangephi)],1))/(rangephi*2)
  
        
    elif element1 < rangephi:
        
        #remainder
        lim1 = rangephi-element1
        intsize1 = lim1+element1+rangephi
#        print(lim1,intsize1)
        cut1 = (np.sum(data[:,0:(element1+rangephi)],1)+np.sum(data[:,(phisize-lim1):phisize],1))/intsize1
        
    elif  element1 >= phisize-rangephi:

        lim1 = rangephi-(phisize-element1)
        intsize1 = lim1+phisize-element1+rangephi
#        print(lim1,element1-rangephi,intsize1)
        cut1 = (np.sum(data[:,(element1-rangephi):phisize],1)+np.sum(data[:,0:lim1],1))/intsize1
        
    if (element2 > rangephi) & (element2 < phisize-rangephi):   
        
        cut2=(np.sum(data[:,(element2-rangephi):(element2+rangephi)],1))/(rangephi*2)
        
    elif element2 < rangephi:
        
        #remainder
       
        lim2 = rangephi-element2
        intsize2 = lim2+element2+rangephi
#        print(lim2,intsize2,phisize-lim2)
        cut2 = (np.sum(data[:,0:(element2+rangephi)],1)+np.sum(data[:,(phisize-lim2):phisize],1))/intsize2
        
    elif element2 >= phisize-rangephi:
        
        lim2 = rangephi-(phisize-element2)
        intsize2 = lim2+phisize-element2+rangephi
#        print(lim2,element2-rangephi,intsize2)
        cut1 = (np.sum(data[:,(element2-rangephi):phisize],1)+np.sum(data[:,0:lim2],1))/intsize2

    # should take care with orientation as this might flip depending on what phicenter you choose
    thetacut=np.concatenate((np.flipud(cut1),cut2),axis=0)
    
    return thetacut


def thetacut90(data,phicenter,width):
    
    # make angular cross cut at specified angle and angular width
    phisize=np.size(data,1)
    phi=np.linspace(0,360,phisize)
    element1=ct.findelement(phi,phicenter)
    #number of pixels per degree times the number of degrees over two
    rangephi=np.int((phisize/360)*(width/2))    
    
    cut1=np.sum(data[:,(element1-rangephi):(element1+rangephi)],1)

    return cut1
     
     
def ARmask_calc(theta1,phi1,holein=1):
    
    #solve equality ar^2-1/(4a)=x. 
    #c=1/(2*(a*cos(phi)*sin(theta)+sqrt(a^2*(cos(theta)^2+(cos(phi)^2+sin(phi)^
    #2)*sin(theta)^2)))); The cos(phi)^2+sin(phi)^2=1 so we can omit that. Than
    #we have cos(theta)^2+sin(theta)^2 which also drops out. That leaves the
    #square root of a^2 
    
    a = 0.1
    xcut = 10.75
    
    ##thetacutoff can actually be calculated
    holesize = 0.6
    holeheight = np.sqrt(2.5/a)
#    thetacutoffhole=np.arctan(holesize/(2*holeheight))*180/np.pi
    thetacutoffhole = 4
    dfoc = 0.5
    phi,theta=np.meshgrid(phi1,theta1)
    c = 1./(2*(a*np.cos(phi)*np.sin(theta)+a))

    z = np.cos(theta)*c
    x = np.sin(theta)*np.cos(phi)*c#-1/(4.*a)
    #y=np.sin(theta)*np.sin(phi)*c

    mask = np.ones([np.size(theta,0),np.size(theta,1)])
    if holein == 1:
        mask[(-x > xcut) | (theta < (thetacutoffhole*np.pi/180)) | (z < dfoc)] = 0
    
    else:
        mask[(-x > xcut) | (z < dfoc)] = 0
        
    return mask     
