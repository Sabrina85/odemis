 # -*- coding: utf-8 -*-
"""
Created on Tue Oct 06 15:12:31 2015

@author: Toon
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import calctools as ct
import normalize as nm
import extracms as col
from matplotlib import ticker
import colbar as colbar
import half_polar as half_pol
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

#hyperspectral plotting function, relies on dataformat provided by dataformat.


def CL_map(data,cw,bw,limits):

    specdata=data[0][0]
    
    # calculate order of magnitude and adjust colorbar ticks and label accordingly 
    order=np.log10(specdata.max())
    plotorder=np.int(order)
        
    wldata=data[0][1]*1e9
    wmax=ct.findelement(wldata,cw+bw/2)
    wmin=ct.findelement(wldata,cw-bw/2)
    
    #wavelength slice summed over the bandwidth    
    specmap=np.squeeze(np.sum(specdata[wmin:wmax,:,:],0))/10**plotorder
    
    #color limits
    lim1=specmap.min()+limits[0]*(specmap.max()-specmap.min())
    lim2=specmap.max()-(1-limits[1])*(specmap.max()-specmap.min())
    
#    Choice whether to plot in nm or um    
    if data[0][2].max() < 0.8e-6 or data[0][3].max() < 0.8e-6 :   
        spatial1=data[0][3]*1e9
        spatial2=data[0][2]*1e9
        xlabelstring='Electron beam pos. ($nm$)'    
        ylabelstring='Electron beam pos. ($nm$)'
    else:
        spatial1=data[0][3]*1e6
        spatial2=data[0][2]*1e6
        xlabelstring='Electron beam pos. ($\mu m$)'    
        ylabelstring='Electron beam pos. ($\mu m$)'

    plt.figure()
    cl_map=plt.imshow(specmap,cmap=col.inferno(),interpolation='none',
                      extent=(spatial1.min(),spatial1.max(),spatial2.min(),spatial2.max()))
    cl_map.set_clim(lim1,lim2)
    colb=colbar.add_colorbar(cl_map,20.5,0.8)
    colb.set_label('CL intensity (10$^{'+np.str(plotorder)+'}$)',size=14)
    colb.ax.tick_params(labelsize=14)
    tick_locator = ticker.MaxNLocator(nbins=5)
    colb.locator = tick_locator
    colb.update_ticks()    
    plt.xlabel(xlabelstring,fontsize=16)
    plt.ylabel(ylabelstring,fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('cw = ' + np.str(cw) + ' nm, bw = '+np.str(bw)+' nm',fontsize=12)
    
    return specmap


def Sem_map(data,semlimits):
    
    semdata=data[1][0]
    ordersem=np.log10(semdata.max())
    plotordersem=np.int(ordersem)
    semdata=semdata/10**plotordersem
    
    #color limits
    lim1sem=semdata.min()+semlimits[0]*(semdata.max()-semdata.min())
    lim2sem=semdata.max()-(1-semlimits[1])*(semdata.max()-semdata.min())
    
    
    if data[1][1].max() < 0.8e-6 or data[1][2].max() < 0.8e-6 :   
        spatial1sem=data[1][2]*1e9
        spatial2sem=data[1][1]*1e9
        xlabelstringsem='Electron beam pos. ($nm$)'    
        ylabelstringsem='Electron beam pos. ($nm$)'
    else:
        spatial1sem=data[1][2]*1e6
        spatial2sem=data[1][1]*1e6
        xlabelstringsem='Electron beam pos. ($\mu m$)'    
        ylabelstringsem='Electron beam pos. ($\mu m$)'

    plt.figure()
    sem_map=plt.imshow(semdata,cmap=cm.gray,interpolation='none',
                      extent=(spatial1sem.min(),spatial1sem.max(),spatial2sem.min(),spatial2sem.max()))
    sem_map.set_clim(lim1sem,lim2sem)
    colb=colbar.add_colorbar(sem_map,20.5,0.8)
    colb.set_label('SE signal (10$^{'+np.str(plotordersem)+'}$)',size=14)
    colb.ax.tick_params(labelsize=14)
    tick_locator = ticker.MaxNLocator(nbins=5)
    colb.locator = tick_locator
    colb.update_ticks()          
    plt.xlabel(xlabelstringsem,fontsize=16)
    plt.ylabel(ylabelstringsem,fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Concurrent SEM image',fontsize=12)
    plt.locator_params(axis = 'x', nbins = 6)  
    plt.locator_params(axis = 'y', nbins = 6)      
    
    return semdata


def Drift_map(data,driftlimits):

    if data[1][1].max() < 0.8e-6 or data[1][2].max() < 0.8e-6 :   
        spatial1drift=data[2]*1e9
        spatial2drift=data[1]*1e9
        xlabelstringdrift='Electron beam pos. ($nm$)'    
        ylabelstringdrift='Electron beam pos. ($nm$)'
    else:
        spatial1drift=data[2]*1e6
        spatial2drift=data[1]*1e6
        xlabelstringdrift='Electron beam pos. ($\mu m$)'    
        ylabelstringdrift='Electron beam pos. ($\mu m$)'
    
    for ii in range(0,3):
        
        driftdata=np.squeeze(data[0][ii,:,:])
        orderdrift=np.log10(driftdata.max())
        plotorderdrift=np.int(orderdrift)
        driftdata=driftdata/10**plotorderdrift
        #color limits
        lim1drift=driftdata.min()+driftlimits[0]*(driftdata.max()-driftdata.min())
        lim2drift=driftdata.max()-(1-driftlimits[1])*(driftdata.max()-driftdata.min())
    
        plt.figure()
        drift_map=plt.imshow(driftdata,cmap=cm.gray,interpolation='none',
                      extent=(spatial1drift.min(),spatial1drift.max(),spatial2drift.min(),spatial2drift.max()))
        drift_map.set_clim(lim1drift,lim2drift)
        colb=plt.colorbar()
        colb.set_label('SE signal (10$^{'+np.str(plotorderdrift)+'}$)',size=14)
        colb.ax.tick_params(labelsize=14)
        tick_locator = ticker.MaxNLocator(nbins=5)
        colb.locator = tick_locator
        colb.update_ticks()          
        plt.xlabel(xlabelstringdrift,fontsize=16)
        plt.ylabel(ylabelstringdrift,fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title('Drift image ' + np.str(ii),fontsize=12)
        plt.locator_params(axis = 'x', nbins = 6)  
        plt.locator_params(axis = 'y', nbins = 6)      
    
    return driftdata


def Survey_map(data,surveylimits):
    
    surveydata=data[2][0]
    ordersurvey=np.log10(surveydata.max())
    plotordersurvey=np.int(ordersurvey)
    surveydata=surveydata/10**plotordersurvey
    
    #colorlimits
    lim1survey=surveydata.min()+surveylimits[0]*(surveydata.max()-surveydata.min())
    lim2survey=surveydata.max()-(1-surveylimits[1])*(surveydata.max()-surveydata.min())

    if data[2][1].max() < 0.8e-6 or data[2][2].max() < 0.8e-6 :   
        spatial1survey=data[2][2]*1e9
        spatial2survey=data[2][1]*1e9
        xlabelstringsurvey='Electron beam pos. ($nm$)'    
        ylabelstringsurvey='Electron beam pos. ($nm$)'
    else:
        spatial1survey=data[2][2]*1e6
        spatial2survey=data[2][1]*1e6
        xlabelstringsurvey='Electron beam pos. ($\mu m$)'    
        ylabelstringsurvey='Electron beam pos. ($\mu m$)'

    plt.figure()
    survey_map=plt.imshow(surveydata,cmap=cm.gray,interpolation='none',
                      extent=(spatial1survey.min(),spatial1survey.max(),spatial2survey.min(),spatial2survey.max()))
    survey_map.set_clim(lim1survey,lim2survey)
    colb=colbar.add_colorbar(survey_map,20.5,0.8)
    colb.set_label('SE signal (10$^{'+np.str(plotordersurvey)+'}$)',size=14)
    colb.ax.tick_params(labelsize=14)
    tick_locator = ticker.MaxNLocator(nbins=5)
    colb.locator = tick_locator
    colb.update_ticks()          
    plt.xlabel(xlabelstringsurvey,fontsize=16)
    plt.ylabel(ylabelstringsurvey,fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('SEM Survey',fontsize=12)
    plt.locator_params(axis = 'x', nbins = 6)  
    plt.locator_params(axis = 'y', nbins = 6)  
   
    return surveydata


def PMT_map(data,PMTlimits,colorm=cm.gray):

    #PMTdata=data[2][0]
    PMTdata=data[0][0]    
    
    #colorlimits
    lim1PMT=PMTdata.min()+PMTlimits[0]*(PMTdata.max()-PMTdata.min())
    lim2PMT=PMTdata.max()-(1-PMTlimits[1])*(PMTdata.max()-PMTdata.min())

    if data[0][1].max() < 0.8e-6 or data[0][2].max() < 0.8e-6 : 
#    if data[2][1].max() < 0.8e-6 or data[2][2].max() < 0.8e-6 :
    
        spatial1PMT=data[0][2]*1e9
        spatial2PMT=data[0][1]*1e9
#        spatial1PMT=data[2][2]*1e9
#        spatial2PMT=data[2][1]*1e9
        xlabelstringPMT='Electron beam pos. ($nm$)'    
        ylabelstringPMT='Electron beam pos. ($nm$)'
    else:
        spatial1PMT=data[0][2]*1e6
        spatial2PMT=data[0][1]*1e6
#        spatial1PMT=data[2][2]*1e6
#        spatial2PMT=data[2][1]*1e6
        xlabelstringPMT='Electron beam pos. ($\mu m$)'    
        ylabelstringPMT='Electron beam pos. ($\mu m$)'

    plt.figure()
    PMT_map=plt.imshow(PMTdata,cmap=colorm,interpolation='none',
                      extent=(spatial1PMT.min(),spatial1PMT.max(),spatial2PMT.min(),spatial2PMT.max()))
    PMT_map.set_clim(lim1PMT,lim2PMT)
    plt.xlabel(xlabelstringPMT,fontsize=16)
    plt.ylabel(ylabelstringPMT,fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('PMT CL image',fontsize=12)
    plt.locator_params(axis = 'x', nbins = 6)  
    plt.locator_params(axis = 'y', nbins = 6)     
    
    return PMTdata


def PMT_map_noaxes(data,PMTlimits,colorm=cm.gray):
    
 
        PMTdata=data[0][0]
        
        if data[2][1].max() < 0.8e-6 or data[2][2].max() < 0.8e-6 :
        
            spatial1PMT=data[2][2]*1e9
            spatial2PMT=data[2][1]*1e9
 
        else:
            spatial1PMT=data[2][2]*1e6
            spatial2PMT=data[2][1]*1e6
            
            #colorlimits
        lim1PMT=PMTdata.min()+PMTlimits[0]*(PMTdata.max()-PMTdata.min())
        lim2PMT=PMTdata.max()-(1-PMTlimits[1])*(PMTdata.max()-PMTdata.min())
            
        plt.figure(figsize=(5, 5))
        PMT_map=plt.imshow(PMTdata,cmap=colorm,interpolation='none',
                      extent=(spatial1PMT.min(),spatial1PMT.max(),spatial2PMT.min(),spatial2PMT.max()))
        PMT_map.set_clim(lim1PMT,lim2PMT)
        plt.axis('off')
        PMT_map.axes.get_xaxis().set_visible(False)
        PMT_map.axes.get_yaxis().set_visible(False)
        
#    else:
#        PMTdata=data
#        
#            #colorlimits
#        lim1PMT=PMTdata.min()+PMTlimits[0]*(PMTdata.max()-PMTdata.min())
#        lim2PMT=PMTdata.max()-(1-PMTlimits[1])*(PMTdata.max()-PMTdata.min())
#        
#        plt.figure()
#        PMT_map=plt.imshow(PMTdata,cmap=cm.gray,interpolation='none',
#                      extent=(0,np.size(PMTdata,1),0,np.size(PMTdata,0)))
#        PMT_map.set_clim(lim1PMT,lim2PMT)
#        plt.axis('off')
#        PMT_map.axes.get_xaxis().set_visible(False)
#        PMT_map.axes.get_yaxis().set_visible(False)
#    

        return PMTdata  
    
        # Todo:
#    - include lines indicating position of the scan area with respect to survey. 
#    - Control number of tickmarks for esthetic reasons


#def RGB_map(data,wlims,lthreshold,uthreshold,noscale):


def RGB_map(data,wlims,lthreshold,uthreshold):
#function that generates a falsecolor RGB values. Input is data, wavelength
#list and wavelength limits. The threshold limits the effect of outliers on
#the color maps.
 
#    noscale=list(noscale)   
    wldata=data[0][1]*1e9    
    specdata=data[0][0]
    el1=ct.findelement(wldata,wlims[0])
    el2=ct.findelement(wldata,wlims[1])
    arraywidth=np.abs(el1-el2)
    mean1=np.sum(np.sum(np.sum(specdata)))/np.size(specdata)

#data thresholding
    specdata[np.greater_equal(specdata,uthreshold*mean1)]=uthreshold*mean1
    specdata[np.less_equal(specdata,lthreshold*mean1)]=0
#specdata[np.less_equal(specdata,lthreshold*mean1)]=lthreshold*mean1

#subdivide data in different color bins
    specdatb=np.squeeze(np.sum(specdata[el1:(el1+np.round(arraywidth/3)),:,:],0))
    specdatg=np.squeeze(np.sum(specdata[(el1+np.round(arraywidth/3)+1):(el1+np.round(2*arraywidth/3)),:,:],0))
    specdatr=np.squeeze(np.sum(specdata[(el1+1+np.round(2*arraywidth/3)):el2,:,:],0))

    
#concatenate color sets
    RGBmap = nm.norm1(np.concatenate((specdatr[...,None],specdatg[...,None],specdatb[...,None]),axis=2))
    
#    if noscale == 1:
#        
#        x1=list(np.shape(data[0][3]))
#        x2=list(np.shape(data[0][2]))
#        spatial1RGB=np.linspace(0,x1[0])
#        spatial2RGB=np.linspace(0,x2[0])
#        xlabelstringRGB='pixel'    
#        ylabelstringRGB='pixel'
#        
#        plt.figure()
#        plt.imshow(RGBmap,interpolation='none',
#                     extent=(spatial1RGB.min(),spatial1RGB.max(),spatial2RGB.min(),spatial2RGB.max()))
##   
#        plt.xlabel(xlabelstringRGB,fontsize=16)
#        plt.ylabel(ylabelstringRGB,fontsize=16)
#        plt.xticks(fontsize=16)
#        plt.yticks(fontsize=16)
#        plt.title('RGB map ' +np.str(wlims[0]) + '-' +np.str(wlims[1]) + ' nm',fontsize=12)
#        
#    else:   
    if data[1][1].max() < 0.8e-6 or data[1][2].max() < 0.8e-6 :   
        spatial1RGB=data[0][3]*1e9
        spatial2RGB=data[0][2]*1e9
        xlabelstringRGB='Electron beam pos. ($nm$)'    
        ylabelstringRGB='Electron beam pos. ($nm$)'
    else:
        spatial1RGB=data[0][3]*1e6
        spatial2RGB=data[0][2]*1e6
        xlabelstringRGB='Electron beam pos. ($\mu m$)'    
        ylabelstringRGB='Electron beam pos. ($\mu m$)'

    plt.figure()
    plt.imshow(RGBmap,interpolation='none',
                     extent=(spatial1RGB.min(),spatial1RGB.max(),spatial2RGB.min(),spatial2RGB.max()))
#   
    plt.xlabel(xlabelstringRGB,fontsize=16)
    plt.ylabel(ylabelstringRGB,fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
#    plt.title('RGB map ' +np.str(wlims[0]) + '-' +np.str(wlims[1]) + ' nm',fontsize=12)
    plt.locator_params(axis = 'x', nbins = 6)  
    plt.locator_params(axis = 'y', nbins = 6)
    plt.axis('off')
    
    return RGBmap
    

def spec1d(data,wl,lims,ylablestring='CL Intensity',xlabelstring='Wavelength (nm)',leg=[]):
# also takes multiple spectra. Should be in the form of a single matrix where wavelength is dim 0
# Todo program that it can take multiple wavelength lists

 ndims=np.ndim(data)

 if ndims==1: 
  
  plt.figure()
  plt.plot(wl,data,'b',linewidth=2)
  plt.axis([lims[0], lims[1], lims[2]*data.max(), lims[3]*data.max()])
  plt.xlabel(xlabelstring,fontsize=22)
  plt.ylabel(ylablestring,fontsize=22)
  plt.xticks(fontsize=22)
  plt.yticks(fontsize=22)  
  plt.locator_params(axis = 'x', nbins = 6)  
  plt.locator_params(axis = 'y', nbins = 5)  
  if np.size(leg)!= 0:
      plt.legend(leg,fontsize=22)
 
 else:
  nspec=np.shape(data)[1]
  blist=np.linspace(1,0,nspec)
  rlist=np.linspace(0,1,nspec)
 
  plt.figure()
  for gg in range(0,nspec):
   cindex=((rlist[gg],0,blist[gg]))   
   plt.plot(wl,data[:,gg],color=cindex,linewidth=2)
 
  plt.axis([lims[0], lims[1], lims[2]*data.max(), lims[3]*data.max()])
  plt.xlabel(xlabelstring,fontsize=22)
  plt.ylabel(ylablestring,fontsize=22)
  plt.xticks(fontsize=22)
  plt.yticks(fontsize=22)
  plt.locator_params(axis = 'x', nbins = 6)  
  plt.locator_params(axis = 'y', nbins = 5)      
  if np.size(leg)!= 0:
      plt.legend(leg,fontsize=22,loc=2)          
 return data


def spec1d_WF(data,wl,lims,offset,ylablestring='Normalized CL Intensity',xlabelstring='Wavelength (nm)'):
# also takes multiple spectra. Should be in the form of a single matrix
# Todo program that it can take multiple wavelength lists

 
 nspec=np.shape(data)[1]
 blist=np.linspace(1,0,nspec)
 rlist=np.linspace(0,1,nspec)
 lowlim=lims[2]*data.max()
 uplim=lims[3]*(data.max()+(nspec-1)*offset) 
 rat=(lims[1]-lims[0])/(uplim-lowlim)  
 
 plt.figure()
 for gg in range(0,nspec):
  cindex=((rlist[gg],0,blist[gg]))   
  plt.plot(wl,data[:,gg]+offset*gg*data.max(),color=cindex,linewidth=2)
 
 plt.axis([lims[0],lims[1],lowlim,uplim])
 plt.xlabel(xlabelstring,fontsize=18)
 plt.ylabel(ylablestring,fontsize=18)
 plt.xticks(fontsize=18)
 plt.yticks(fontsize=18) 
 plt.axes().set_aspect(rat*2)
 plt.locator_params(axis = 'x', nbins = 4)  
 plt.locator_params(axis = 'y', nbins = 6)    
   
 return data


def RGB_PMT(data1,data2,data3,lthreshold,uthreshold):
 #data should have same lengthscale and resolution!   
 specdatr=data1[2][0]
 specdatg=data2[2][0]
 specdatb=data3[2][0]
 
 meanr=specdatr.sum()/np.size(specdatr)
 meang=specdatg.sum()/np.size(specdatg)
 meanb=specdatb.sum()/np.size(specdatb)
 
# print meanr
# print specdatr.max()
# print meang
# print specdatg.max()
# print meanb 
# print specdatb.max()
 
 specdatr[np.greater_equal(specdatr,uthreshold*meanr)]=uthreshold*meanr
 specdatr[np.less_equal(specdatr,lthreshold*meanr)]=0
 specdatg[np.greater_equal(specdatg,uthreshold*meang)]=uthreshold*meang
 specdatg[np.less_equal(specdatg,lthreshold*meang)]=0
 specdatb[np.greater_equal(specdatb,uthreshold*meanb)]=uthreshold*meanb
 specdatb[np.less_equal(specdatb,lthreshold*meanb)]=0
 

 if data1[2][1].max() < 0.8e-6 or data1[2][2].max() < 0.8e-6 :   
    spatial1RGBPMT=data1[2][2]*1e9
    spatial2RGBPMT=data1[2][1]*1e9
    xlabelstringRGBPMT='Electron beam pos. ($nm$)'    
    ylabelstringRGBPMT='Electron beam pos. ($nm$)'
 else:
    spatial1RGBPMT=data1[2][2]*1e6
    spatial2RGBPMT=data1[2][1]*1e6
    xlabelstringRGBPMT='Electron beam pos. ($\mu m$)'    
    ylabelstringRGBPMT='Electron beam pos. ($\mu m$)'
 
 RGBmap=nm.norm1(np.concatenate((specdatr[...,None],specdatg[...,None],specdatb[...,None]),axis=2))
 plt.figure()
 plt.imshow(RGBmap,interpolation='none',
 extent=(spatial1RGBPMT.min(),spatial1RGBPMT.max(),spatial2RGBPMT.min(),spatial2RGBPMT.max()))
 plt.xlabel(xlabelstringRGBPMT,fontsize=16)
 plt.ylabel(ylabelstringRGBPMT,fontsize=16)
 plt.xticks(fontsize=16)
 plt.yticks(fontsize=16)
 plt.locator_params(axis = 'x', nbins = 6)  
 plt.locator_params(axis = 'y', nbins = 6) 
# plt.title('RGB map ' +np.str(wlims[0]) + '-' +np.str(wlims[1]) + ' nm',fontsize=12)

 return RGBmap   
    

def RGB_PMT_noaxes(data1,data2,data3,lthreshold,uthreshold):
 #data should have same lengthscale and resolution!   
 specdatr=data1[0][0]
 specdatg=data2[0][0]
 specdatb=data3[0][0]
 
 meanr=specdatr.sum()/np.size(specdatr)
 meang=specdatg.sum()/np.size(specdatg)
 meanb=specdatb.sum()/np.size(specdatb)
 
# print meanr
# print specdatr.max()
# print meang
# print specdatg.max()
# print meanb 
# print specdatb.max()
 
 specdatr[np.greater_equal(specdatr,uthreshold*meanr)]=uthreshold*meanr
 specdatr[np.less_equal(specdatr,lthreshold*meanr)]=0
 specdatg[np.greater_equal(specdatg,uthreshold*meang)]=uthreshold*meang
 specdatg[np.less_equal(specdatg,lthreshold*meang)]=0
 specdatb[np.greater_equal(specdatb,uthreshold*meanb)]=uthreshold*meanb
 specdatb[np.less_equal(specdatb,lthreshold*meanb)]=0
 

 if data1[2][1].max() < 0.8e-6 or data1[2][2].max() < 0.8e-6 :   
    spatial1RGBPMT=data1[0][2]*1e9
    spatial2RGBPMT=data1[0][1]*1e9
   
 else:
    spatial1RGBPMT=data1[0][2]*1e6
    spatial2RGBPMT=data1[0][1]*1e6
  
 
 RGBmap=nm.norm1(np.float64(np.concatenate((specdatr[...,None],specdatg[...,None],specdatb[...,None]),axis=2)))
 plt.figure(figsize=(5, 5))
 fig=plt.imshow(RGBmap,interpolation='none',
 extent=(spatial1RGBPMT.min(),spatial1RGBPMT.max(),spatial2RGBPMT.min(),spatial2RGBPMT.max()))
 plt.axis('off')
 fig.axes.get_xaxis().set_visible(False)
 fig.axes.get_yaxis().set_visible(False)

 return RGBmap       


#include mask to check pole position and magnification
def AR_raw(data,limits,mask=0):
    
    order=np.log10(data.max())
    plotorder=np.int(order)
    data=data/10**plotorder    
      
    #color limits
    lim1=data.min()+limits[0]*(data.max()-np.abs(data.min()))
    lim2=data.max()-(1-limits[1])*(data.max()-np.abs(data.min()))
    
    if lim1 < 0:
        lim1=0        

    xlabelstring='$x$ (pixels)'    
    ylabelstring='$z$ (pixels)'
    
    x=np.linspace(1,np.size(data,1)+1,np.size(data,1))
    z=np.linspace(1,np.size(data,0)+1,np.size(data,0))  
#    rat=z.max()/x.max()
    
    if mask==0:
     fig = plt.figure()
     #flip data to make consistent with axes again
     AR_map=plt.imshow(np.flipud(data),cmap=col.inferno(),interpolation='none',
                      extent=(x.min(),x.max(),z.min(),z.max()))
     AR_map.set_clim(lim1,lim2)
     #colb=plt.colorbar()
     #colb.set_label('CL intensity (10$^{'+np.str(plotorder)+'}$)',size=14)
     #colb.ax.tick_params(labelsize=14)
     #tick_locator = ticker.MaxNLocator(nbins=5)
     #colb.locator = tick_locator
     #colb.update_ticks()     
     plt.xlabel(xlabelstring,fontsize=16)
     plt.ylabel(ylabelstring,fontsize=16)
     plt.xticks(fontsize=16)
     plt.yticks(fontsize=16)
     plt.title('AR image',fontsize=12)
#     plt.axes().set_aspect(rat)

    return fig,AR_map


def AR_raw_noaxes(data,limits,mask=0,sizefig=(10,10)):
    
    order=np.log10(data.max())
    plotorder=np.int(order)
    data=data/10**plotorder    
    
    #define slit for plot
    hor1=np.ones(data.shape[0])*(data.shape[1]/2+5)
    hor2=np.ones(data.shape[0])*(data.shape[1]/2-5)
    ver=np.arange(data.shape[0])
    
    
    #color limits
    lim1=data.min()+limits[0]*(data.max()-np.abs(data.min()))
    lim2=data.max()-(1-limits[1])*(data.max()-np.abs(data.min()))
    
    if lim1 < 0:
        lim1=0        

    x=np.linspace(1,np.size(data,1)+1,np.size(data,1))
    z=np.linspace(1,np.size(data,0)+1,np.size(data,0))  
#    rat=z.max()/x.max()
    
    if mask==0:
     fig = plt.figure(figsize=sizefig)
     #flip data to make consistent with axes again
     lineplot = plt.plot(hor1,ver,'--w',hor2,ver,'--w',linewidth=2)
     cur_axes = plt.gca()
     cur_axes.axes.get_xaxis().set_visible(False)
     cur_axes.axes.get_yaxis().set_visible(False)
     plt.axis('off')
#     lineplotax =  
     AR_map=plt.imshow(np.flipud(data),cmap=col.inferno(),interpolation='none',
                      extent=(x.min(),x.max(),z.min(),z.max()))
     AR_map.set_clim(lim1,lim2)
#     plt.axes().set_aspect(rat)
     plt.axis('off')
     AR_map.axes.get_xaxis().set_visible(False)
     AR_map.axes.get_yaxis().set_visible(False)
     
#     ax = plt.Axes(fig1, [0., 0., 1., 1.])
#     ax.set_axis_off()
#     plt.axis('off')

    return fig,AR_map


def AR_map(data,limits,mask=0):
    
    order=np.log10(data.max())
    plotorder=np.int(order)
    data=data/10**plotorder    
      
    #color limits
    lim1=data.min()+limits[0]*(data.max()-np.abs(data.min()))
    lim2=data.max()-(1-limits[1])*(data.max()-np.abs(data.min()))
    
    if lim1 < 0:
        lim1=0        

    xlabelstring='$x$ (pixels)'    
    ylabelstring='$z$ (pixels)'
    
    x=np.linspace(1,np.size(data,1)+1,np.size(data,1))
    z=np.linspace(1,np.size(data,0)+1,np.size(data,0))  
#    rat=z.max()/x.max()
    
    if mask==0:
     fig = plt.figure()
     #flip data to make consistent with axes again
     AR_map=plt.imshow(np.flipud(data),cmap=col.EELSmap(),interpolation='none',
                      extent=(x.min(),x.max(),z.min(),z.max()))
     AR_map.set_clim(lim1,lim2)
     #colb=plt.colorbar()
     #colb.set_label('CL intensity (10$^{'+np.str(plotorder)+'}$)',size=14)
     #colb.ax.tick_params(labelsize=14)
     #tick_locator = ticker.MaxNLocator(nbins=5)
     #colb.locator = tick_locator
     #colb.update_ticks()     
     plt.xlabel(xlabelstring,fontsize=16)
     plt.ylabel(ylabelstring,fontsize=16)
     plt.xticks(fontsize=16)
     plt.yticks(fontsize=16)
#     plt.title('AR image',fontsize=12)
#     plt.axes().set_aspect(rat)

    return fig,AR_map


def AR_square(data,limits):
    
    order=np.log10(data.max())
    plotorder=np.int(order)
    data=data/10**plotorder    
    phi=np.linspace(0,360,np.size(data,0))
    theta=np.linspace(0,90,np.size(data,1))   
      
    #color limits
      #color limits
    if np.size(limits) == 1:
        
     lim1=0
     lim2=limits/10**plotorder
    
    if np.size(limits) == 2:  
        
     lim1=data.min()+limits[0]*(data.max()-data.min())
     lim2=data.max()-(1-limits[1])*(data.max()-data.min())
     
     if lim1 < 0:
        lim1=0     
    
    xlabelstring='$\phi$ (degrees)'    
    ylabelstring='$\\theta$ (degrees)'
    
    
    plt.figure(figsize=(10, 5))
       
  
    AR_map=plt.imshow(np.flipud(data),cmap=col.inferno(),interpolation='none',
                      extent=(phi.min(),phi.max(),theta.min(),theta.max()))
    AR_map.set_clim(lim1,lim2)
    colb=colbar.add_colorbar(AR_map,9.5,0.5)
    colb.set_label('CL intensity (10$^{'+np.str(plotorder)+'}$ cts/sr)',size=14)
    colb.ax.tick_params(labelsize=14)
    tick_locator = ticker.MaxNLocator(nbins=5)
    colb.locator = tick_locator
    colb.update_ticks()  
    
    plt.xlabel(xlabelstring,fontsize=16)
    plt.ylabel(ylabelstring,fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('AR image',fontsize=12)
    plt.locator_params(axis = 'x', nbins = 5)  
    plt.locator_params(axis = 'y', nbins = 4)

    return AR_map


def AR_polar(data,limits,overlay=0,sizefig=(10,10)):
    
    order=np.log10(data.max())
    plotorder=np.int(order)
    data=data/10**plotorder    
    #color limits
    if np.size(limits) == 1:
        
     lim1=0
     lim2=limits/10**plotorder
    
    if np.size(limits) == 2:
        
     lim1=data.min()+limits[0]*(data.max()-data.min())
     lim2=data.max()-(1-limits[1])*(data.max()-data.min())
     
     if lim1 < 0:
        lim1=0
    
    phi1=np.linspace(0,2*np.pi,np.size(data,1))
    theta1=np.linspace(0,np.pi/2,np.size(data,0))
    phi,theta=np.meshgrid(phi1,theta1)
    y=np.cos(phi)*theta
    x=np.sin(phi)*theta
    
    #add ring overlay
    npoints=500
    angle=np.linspace(0,2*np.pi,npoints)
    cx=np.cos(angle)
    cy=np.sin(angle)    
    ringpos=[1,2./3,1./3]    
    ringmaty=np.zeros([npoints,3])    
    ringmatx=np.zeros([npoints,3])

    for ii in range(0,3):        
        ringmaty[:,ii]=cy*y.max()*ringpos[ii]
        ringmatx[:,ii]=cx*x.max()*ringpos[ii]
    
    #add spider overlay
    nspider=300
    anglelist=np.linspace(0,150,6)*np.pi/180
    spidermatrix_x=np.zeros([nspider,np.size(anglelist)])
    spidermatrix_y=np.zeros([nspider,np.size(anglelist)])
    xmat=np.linspace(x.min(),x.max(),nspider)
    ymat=np.linspace(y.min(),y.max(),nspider)

    for jj in range(0,np.size(anglelist)):
     spidermatrix_x[:,jj]=xmat*np.cos(anglelist[jj])
     spidermatrix_y[:,jj]=ymat*np.sin(anglelist[jj])
    
    fig = plt.figure(figsize=sizefig)
#    polar=plt.pcolormesh(x,y,data,cmap=col.inferno())
    polar=plt.pcolormesh(x,y,data,cmap=col.EELSmap())
#    polar=plt.pcolormesh(x,y,data,cmap=cm.gray)
    polar.set_clim(lim1,lim2)
    plt.axis('equal')
    plt.axis('off')
    
    if overlay == 0:
        
     colb=colbar.add_colorbar(polar,20.5,-0.5)
     colb.set_label('CL intensity (10$^{'+np.str(plotorder)+'}$ cts/sr)',size=14)
     colb.ax.tick_params(labelsize=14)
     tick_locator = ticker.MaxNLocator(nbins=5)
     colb.locator = tick_locator
     colb.update_ticks()
     plt.plot(ringmaty,ringmatx,color=((0.6,0.6,0.6)),linewidth=2)      
     plt.plot(spidermatrix_x,spidermatrix_y,color=((0.6,0.6,0.6)),linewidth=2)

    return polar, fig
    

def AR_polar_stokes_norm(data, overlay=0):
    
    phi1 = np.linspace(0, 2*np.pi, np.size(data, 1))
    theta1 = np.linspace(0, np.pi/2, np.size(data, 0))
    phi, theta = np.meshgrid(phi1, theta1)
    y = np.cos(phi)*theta
    x = np.sin(phi)*theta
    
    # add ring overlay
    npoints = 500
    angle = np.linspace(0, 2*np.pi, npoints)
    cx = np.cos(angle)
    cy = np.sin(angle)
    ringpos=[1,2./3,1./3]    
    ringmaty=np.zeros([npoints,3])    
    ringmatx=np.zeros([npoints,3])

    for ii in range(0,3):        
        ringmaty[:,ii]=cy*y.max()*ringpos[ii]
        ringmatx[:,ii]=cx*x.max()*ringpos[ii]
    
    #add spider overlay
    nspider=300
    anglelist=np.linspace(0,150,6)*np.pi/180
    spidermatrix_x=np.zeros([nspider,np.size(anglelist)])
    spidermatrix_y=np.zeros([nspider,np.size(anglelist)])
    xmat=np.linspace(x.min(),x.max(),nspider)
    ymat=np.linspace(y.min(),y.max(),nspider)

    for jj in range(0,np.size(anglelist)):
     spidermatrix_x[:,jj]=xmat*np.cos(anglelist[jj])
     spidermatrix_y[:,jj]=ymat*np.sin(anglelist[jj])
    
    plt.figure()
    polar=plt.pcolormesh(x,y,data,cmap = col.mitcolors())
#    polar=plt.pcolormesh(x,y,data,cmap=cm.gray)
    polar.set_clim(-1,1)
    plt.axis('equal')
    plt.axis('off')
    
    if overlay == 0:
        
     colb=colbar.add_colorbar(polar,20.5,-0.5)
#     colb.set_label('CL intensity (10$^{'+np.str(plotorder)+'}$ cts/sr)',size=14)
     colb.ax.tick_params(labelsize=14)
     tick_locator = ticker.MaxNLocator(nbins=5)
     colb.locator = tick_locator
     colb.update_ticks()
     plt.plot(ringmaty,ringmatx,color=((0.6,0.6,0.6)),linewidth=2)      
     plt.plot(spidermatrix_x,spidermatrix_y,color=((0.6,0.6,0.6)),linewidth=2)

    return polar


def AR_phasedif_plot(data,overlay=0):
    
    phi1=np.linspace(0,2*np.pi,np.size(data,1))
    theta1=np.linspace(0,np.pi/2,np.size(data,0))
    phi,theta=np.meshgrid(phi1,theta1)
    y=np.cos(phi)*theta
    x=np.sin(phi)*theta
    
    phasedif = np.abs(np.arctan2(np.imag(data[:,:,0]),np.real(data[:,:,0]))-np.arctan2(np.imag(data[:,:,1]),np.real(data[:,:,1])))/np.pi
    phasedif[phasedif > 1] = 1 - (phasedif[phasedif > 1]-1)
    
    
    #add ring overlay
    npoints=500
    angle=np.linspace(0,2*np.pi,npoints)
    cx=np.cos(angle)
    cy=np.sin(angle)    
    ringpos=[1,2./3,1./3]    
    ringmaty=np.zeros([npoints,3])    
    ringmatx=np.zeros([npoints,3])

    for ii in range(0,3):        
        ringmaty[:,ii]=cy*y.max()*ringpos[ii]
        ringmatx[:,ii]=cx*x.max()*ringpos[ii]
    
    #add spider overlay
    nspider=300
    anglelist=np.linspace(0,150,6)*np.pi/180
    spidermatrix_x=np.zeros([nspider,np.size(anglelist)])
    spidermatrix_y=np.zeros([nspider,np.size(anglelist)])
    xmat=np.linspace(x.min(),x.max(),nspider)
    ymat=np.linspace(y.min(),y.max(),nspider)

    for jj in range(0,np.size(anglelist)):
     spidermatrix_x[:,jj]=xmat*np.cos(anglelist[jj])
     spidermatrix_y[:,jj]=ymat*np.sin(anglelist[jj])
    
    plt.figure()
    polar=plt.pcolormesh(x,y,phasedif,cmap = col.mitcolors())
#    polar=plt.pcolormesh(x,y,data,cmap=cm.gray)
    polar.set_clim(-1,1)
    plt.axis('equal')
    plt.axis('off')
    
    if overlay == 0:
        
     colb=colbar.add_colorbar(polar,20.5,-0.5)
#     colb.set_label('CL intensity (10$^{'+np.str(plotorder)+'}$ cts/sr)',size=14)
     colb.ax.tick_params(labelsize=14)
     tick_locator = ticker.MaxNLocator(nbins=5)
     colb.locator = tick_locator
     colb.update_ticks()
     plt.plot(ringmaty,ringmatx,color=((0.6,0.6,0.6)),linewidth=2)      
     plt.plot(spidermatrix_x,spidermatrix_y,color=((0.6,0.6,0.6)),linewidth=2)
    
    return polar


def thetacut_ARspectral_theory(data,el):
    
    #code to make crosscuts through theory. Also mirrors the data from one angular quadrant for completeness
 
    ardata=data[:,el]
    ardata1=nm.norm1(np.concatenate((np.flipud(ardata),ardata),axis=0))
    theta=np.linspace(180,0,np.size(ardata1))
    f1=plt.figure()
    a1 = half_pol.fractional_polar_axes(f1)
    a1.plot(theta,ardata1,'b',linewidth=2)
    plt.show(f1)

    return ardata


def thetacut_serial(data):
    
    #code to make crosscuts through theory. Also mirrors the data from one angular quadrant for completeness
 
    theta_pol = np.linspace(180,0,data.shape[0])
    f1=plt.figure()
    a1 = half_pol.fractional_polar_axes(f1)
    nspec= data.shape[1]
    blist=np.linspace(1,0,nspec)
    rlist=np.linspace(0,1,nspec)
    
    for pp in range(nspec):
        cindex=((rlist[pp],0,blist[pp])) 
        a1.plot(theta_pol,nm.norm1(data[:,pp]),color=cindex,linewidth=2)
    
    plt.show(f1)

    return 


def waterfall3D(X, Y, Z, lims,angles = [-60,30],xlabelstring='$\phi$ (degrees)',ylabelstring='$Energy (eV)',zlabelstring='CL intensity'):
    
#  slc = np.arange(0, 10, 1)
# I haven't quite figured out how to use the meshgrid function in numpy

  # Function to generate formats for facecolors
  cc = lambda arg: colorConverter.to_rgba(arg, alpha=0.3)
  # This is just wrong. There must be some way to use the meshgrid or why bother.
  verts = []
  for i in range(X.shape[0]):
    verts.append(list(zip(X[i], Z[i])))

#  xmin = np.floor(np.min(X))
#  xmax = np.ceil(np.max(X))
#  ymin = np.floor(np.min(Y))
#  ymax = np.ceil(np.max(Y))
  
  xmin = lims[0]   
  xmax = lims[1]
  ymin = lims[2]
  ymax = lims[3] 
  zmin = 0
#  zmin = np.floor(np.min(Z.real))
  zmax = np.ceil(np.max(np.abs(Z)))

  fig = plt.figure(figsize=(15, 10))
  ax = Axes3D(fig)
 
  poly = PolyCollection(verts, facecolors=[cc('b')])
  ax.add_collection3d(poly, zs=Y[:,0], zdir='y')
  ax.set_xlim(xmin,xmax)
  ax.set_ylim(ymin,ymax)
  ax.set_zlim(zmin,zmax)
  ax.azim = angles[0]
  ax.elev = angles[1]
#  plt.xlabel(xlabelstring,fontsize=16)
#  plt.ylabel(ylabelstring,fontsize=16)
#  ax.set_zlabel(zlabelstring,fontsize=16)
  plt.xticks(fontsize=12)
  plt.yticks(fontsize=12)
  for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(12)
  plt.show()
  

def AR_spec_mapraw(ARdata,wlist,ARlimits):

    order=np.log10(ARdata.max())
    plotorder=np.int(order)
    ARdata=ARdata/10**plotorder   
        
    #colorlimits
    lim1AR=ARdata.min()+ARlimits[0]*(ARdata.max()-ARdata.min())
    lim2AR=ARdata.max()-(1-ARlimits[1])*(ARdata.max()-ARdata.min())

    wlist1=wlist*1e9
    xlabelstringAR='Wavelength (nm)'    
    ylabelstringAR='CCD pixel'
    CCDpix=np.linspace(1,np.size(ARdata,0),np.size(ARdata,0))
#    rat=(wlist1.max()-wlist1.min())/(CCDpix.max()-CCDpix.min())
    wlmesh, pixelmesh = np.meshgrid(wlist1,CCDpix)

    plt.figure(figsize=(10, 10))
    ARmap=plt.pcolormesh(wlmesh,pixelmesh,ARdata,cmap=col.inferno())
    colb=colbar.add_colorbar(ARmap,10.5,0.8)
    colb.set_label('CL intensity (10$^{'+np.str(plotorder)+'}$ cts)',size=16)
    colb.ax.tick_params(labelsize=18)
    tick_locator = ticker.MaxNLocator(nbins=5)
    colb.locator = tick_locator
    colb.update_ticks()
    ARmap.set_clim(lim1AR,lim2AR)
    plt.axis([wlist1.min(), wlist1.max(), CCDpix.min(), CCDpix.max()])
    plt.xlabel(xlabelstringAR,fontsize=18)
    plt.ylabel(ylabelstringAR,fontsize=18)
    plt.xticks(fontsize=18) 
    plt.yticks(fontsize=18)
    plt.locator_params(axis = 'x', nbins = 4)  
    plt.locator_params(axis = 'y', nbins = 4)
    plt.title('AR spectral image raw',fontsize=18)
    
    return ARdata


def AR_spec_lensscan_mapraw(ARdata,pixelsize,xstepsize,ARlimits):

    order=np.log10(ARdata.max())
    plotorder=np.int(order)
    ARdata=ARdata/10**plotorder   
        
    #colorlimits
    lim1AR=ARdata.min()+ARlimits[0]*(ARdata.max()-ARdata.min())
    lim2AR=ARdata.max()-(1-ARlimits[1])*(ARdata.max()-ARdata.min())
         
    xlabelstringAR='Y direction (mm)'    
    ylabelstringAR='Z direction (mm)'
    CCDpixz = np.linspace(0,np.size(ARdata,0)-1,np.size(ARdata,0))*pixelsize
    CCDpixy = np.linspace(-(np.size(ARdata,1)-1)/2,(np.size(ARdata,1)-1)/2,np.size(ARdata,1))*xstepsize
    pixelmeshy, pixelmeshz = np.meshgrid(CCDpixy,CCDpixz)
#    rat = (pixelmeshz.max()-pixelmeshz.min())/(pixelmeshy.max()-pixelmeshy.min())

    plt.figure(figsize=(10, 5))
    ARmap=plt.pcolormesh(pixelmeshy,pixelmeshz,ARdata,cmap=col.EELSmap())
#    colb=colbar.add_colorbar(ARmap,10.5,0.8)
#    colb.set_label('CL intensity (10$^{'+np.str(plotorder)+'}$ cts)',size=16)
#    colb.ax.tick_params(labelsize=18)
#    tick_locator = ticker.MaxNLocator(nbins=5)
#    colb.locator = tick_locator
#    colb.update_ticks()
    ARmap.set_clim(lim1AR,lim2AR)
    plt.xlabel(xlabelstringAR,fontsize=18)
    plt.ylabel(ylabelstringAR,fontsize=18)
    plt.xticks(fontsize=18) 
    plt.yticks(fontsize=18)
    plt.locator_params(axis = 'x', nbins = 4)  
    plt.locator_params(axis = 'y', nbins = 4)
#    plt.title('AR spectral image raw',fontsize=18)
    plt.axis('equal')
    plt.axis([CCDpixy.min(), CCDpixy.max(), CCDpixz.min(), CCDpixz.max()])
#    plt.axes().set_aspect(rat*2)
    
    return ARdata

    
def AR_spec_processed_eV_angle(ARdata,eVlist,thetalistdeg,ARlimits):    
    
    order=np.log10(ARdata.max())
    plotorder=np.int(order)-1
    ARdata=ARdata/10**plotorder   
        
    #colorlimits
    lim1AR=ARdata.min()+ARlimits[0]*(ARdata.max()-ARdata.min())
    lim2AR=ARdata.max()-(1-ARlimits[1])*(ARdata.max()-ARdata.min())
   
    ylabelstringAR='Energy (eV)'    
    xlabelstringAR='Polar angle (degrees)'
    thetamesh, eVmesh = np.meshgrid(thetalistdeg,eVlist)
#    rat=(eVlist.max()-eVlist.min())/(thetalistdeg.max()-thetalistdeg.min())
#    thetalist1=np.linspace(-90,90,np.size(thetalist))
    
    plt.figure(figsize=(10, 10))
    ARmap=plt.pcolormesh(thetamesh,eVmesh,ARdata,cmap=col.EELSmap())
    colb=colbar.add_colorbar(ARmap,0.2,0.8)
    colb.set_label('CL intensity (10$^{'+np.str(plotorder)+'}$ eV$^{-1}$sr$^{-1}$)',size=20)
    colb.ax.tick_params(labelsize=20)
    tick_locator = ticker.MaxNLocator(nbins=5)
    colb.locator = tick_locator
    colb.update_ticks()
    ARmap.set_clim(lim1AR,lim2AR)
    plt.axis([-90, 90,eVlist.min(), eVlist.max()])
    plt.xlabel(xlabelstringAR,fontsize=20)
    plt.ylabel(ylabelstringAR,fontsize=20)
    plt.xticks(fontsize=20) 
    plt.yticks(fontsize=20)
    plt.locator_params(axis = 'x', nbins = 7)  
    plt.locator_params(axis = 'y', nbins = 6)
#    plt.axes().set_aspect(0.2)
#    plt.axis('square')
#    plt.axes().set_aspect(1./rat) 

    return ARdata


def AR_spec_processed_eV_momentum(ARdata,eVlist,thetalistdeg,ARlimits):    
    
    order=np.log10(ARdata.max())
    plotorder=np.int(order)-1
    ARdata=ARdata/10**plotorder   
     
    knot = eVlist*np.pi*2/1239.84
    
    theta_mesh, eV_mesh = np.meshgrid(thetalistdeg,eVlist)
    theta_mesh, knot_mesh = np.meshgrid(thetalistdeg, knot)
    
    kpar_mesh = np.sin(theta_mesh*np.pi/180)*knot_mesh*100
    
    #colorlimits
    lim1AR=ARdata.min()+ARlimits[0]*(ARdata.max()-ARdata.min())
    lim2AR=ARdata.max()-(1-ARlimits[1])*(ARdata.max()-ARdata.min())

    ylabelstringAR='Energy (eV)'    
    xlabelstringAR='k$_{x}$ (10$^{7}$ m$^{-1}$)'
    
    rat = (kpar_mesh.max()-kpar_mesh.min())/(eVlist.max()-eVlist.min())
#    thetalist1=np.linspace(-90,90,np.size(thetalist))
    
    plt.figure(figsize=(10, 10))
    ARmap=plt.pcolormesh(kpar_mesh,eV_mesh,ARdata,cmap=col.EELSmap())
    colb=colbar.add_colorbar(ARmap,9.5,0.8)
    colb.set_label('CL intensity (10$^{'+np.str(plotorder)+'}$ eV$^{-1}$sr$^{-1}$)',size=20)
    colb.ax.tick_params(labelsize=20)
    tick_locator = ticker.MaxNLocator(nbins=5)
    colb.locator = tick_locator
    colb.update_ticks()
    ARmap.set_clim(lim1AR,lim2AR)
    plt.axis([kpar_mesh.min(), kpar_mesh.max(),eVlist.min(), eVlist.max()])
    plt.xlabel(xlabelstringAR,fontsize=20)
    plt.ylabel(ylabelstringAR,fontsize=20)
    plt.xticks(fontsize=20) 
    plt.yticks(fontsize=20)
    plt.locator_params(axis = 'x', nbins = 7)  
    plt.locator_params(axis = 'y', nbins = 6)
#    plt.axes().set_aspect(0.2)
#    plt.axis('square')
#    plt.axes().set_aspect(1./rat) 

    return ARdata, kpar_mesh


def AR_spec_processed_eV_momentum_norm(ARdata,eVlist,thetalistdeg,a,ARlimits,aa=0):    
    
    order=np.log10(ARdata.max())
    plotorder=np.int(order)-1
    ARdata=ARdata/10**plotorder   
     
    knot = eVlist*np.pi*2/1239.84
    
    theta_mesh, eV_mesh = np.meshgrid(thetalistdeg,eVlist)
    theta_mesh, knot_mesh = np.meshgrid(thetalistdeg, knot)
    
    kpar_mesh = np.sin(theta_mesh*np.pi/180)*knot_mesh*a*1e9/np.pi
    
    #colorlimits
    lim1AR=ARdata.min()+ARlimits[0]*(ARdata.max()-ARdata.min())
    lim2AR=ARdata.max()-(1-ARlimits[1])*(ARdata.max()-ARdata.min())

    ylabelstringAR='Energy (eV)'    
    xlabelstringAR='k$_{x}$ ($\pi$/a)'
    
    rat = (kpar_mesh.max()-kpar_mesh.min())/(eVlist.max()-eVlist.min())
#    thetalist1=np.linspace(-90,90,np.size(thetalist))
    
    if aa == 0:
    
        plt.figure(figsize=(10, 10))
        ARmap=plt.pcolormesh(kpar_mesh,eV_mesh,ARdata,cmap=col.EELSmap())
        colb=colbar.add_colorbar(ARmap,9.5,0.8)
        colb.set_label('CL intensity (10$^{'+np.str(plotorder)+'}$ eV$^{-1}$sr$^{-1}$)',size=20)
        colb.ax.tick_params(labelsize=20)
        tick_locator = ticker.MaxNLocator(nbins=5)
        colb.locator = tick_locator
        colb.update_ticks()
        ARmap.set_clim(lim1AR,lim2AR)
        plt.axis([kpar_mesh.min(), kpar_mesh.max(),eVlist.min(), eVlist.max()])
        plt.xlabel(xlabelstringAR,fontsize=20)
        plt.ylabel(ylabelstringAR,fontsize=20)
        plt.xticks(fontsize=20) 
        plt.yticks(fontsize=20)
        plt.locator_params(axis = 'x', nbins = 7)  
        plt.locator_params(axis = 'y', nbins = 6)
        
    else:
        plt.figure(figsize=(10, 10))
        ARmap=plt.pcolormesh(kpar_mesh,eV_mesh,ARdata,cmap=col.EELSmap())
        plt.axis("off")
        ARmap.set_clim(lim1AR,lim2AR)
        
#    plt.axes().set_aspect(0.2)
#    plt.axis('square')
#    plt.axes().set_aspect(1./rat) 

    return ARdata, kpar_mesh


def AR_spec_processed_nm(ARdata,wllist,thetalistdeg,ARlimits,aa=0):    
    
    order=np.log10(ARdata.max())
    plotorder=np.int(order)-1
    ARdata=ARdata/10**plotorder   
        
    #colorlimits
    lim1AR=ARdata.min()+ARlimits[0]*(ARdata.max()-ARdata.min())
    lim2AR=ARdata.max()-(1-ARlimits[1])*(ARdata.max()-ARdata.min())
         
    wllist1=wllist
    xlabelstringAR='Wavelength (nm)'    
    ylabelstringAR='Polar angle (degrees)'
    wlmesh, thetamesh = np.meshgrid(wllist1,thetalistdeg)
#    rat=(eVlist.max()-eVlist.min())/(thetalistdeg.max()-thetalistdeg.min())
#    thetalist1=np.linspace(-90,90,np.size(thetalist))
    
    if aa == 0:
    
        plt.figure(figsize=(10, 10))
        ARmap=plt.pcolormesh(wlmesh,thetamesh,ARdata,cmap=col.EELSmap())
        colb=colbar.add_colorbar(ARmap,7.5,0.8)
        colb.set_label('CL intensity (10$^{'+np.str(plotorder)+'}$ nm$^{-1}$sr$^{-1}$)',size=20)
        colb.ax.tick_params(labelsize=20)
        tick_locator = ticker.MaxNLocator(nbins=5)
        colb.locator = tick_locator
        colb.update_ticks()
        ARmap.set_clim(lim1AR,lim2AR)
        plt.axis([wllist1.min(), wllist1.max(), -90, 90])
        plt.xlabel(xlabelstringAR,fontsize=20)
        plt.ylabel(ylabelstringAR,fontsize=20)
        plt.xticks(fontsize=20) 
        plt.yticks(fontsize=20)
        plt.locator_params(axis = 'x', nbins = 5)  
        plt.locator_params(axis = 'y', nbins = 6)
    
    else:
        plt.figure(figsize=(10, 10))
        ARmap=plt.pcolormesh(wlmesh,thetamesh,ARdata,cmap=col.EELSmap())
        plt.axis("off")
        ARmap.set_clim(lim1AR,lim2AR)
        
    return ARdata


def AR_spec_processed_norm(ARdata,wllist,thetalist,ARlimits):    

    #colorlimits
    lim1AR=0
    lim2AR=1

    wllist1=wllist*1e9
    xlabelstringAR='Wavelength (nm)'    
    ylabelstringAR='Polar angle (degrees)'
#    rat=(wlist1.max()-wlist1.min())/(thetalist.max()-thetalist.min())
#    thetalist1=np.linspace(-90,90,np.size(thetalist))
    wlmesh, thetamesh = np.meshgrid(wllist1,thetalist)
         
    plt.figure(figsize=(10, 10))
    ARmap=plt.pcolormesh(wlmesh,thetamesh,ARdata,cmap=col.inferno())
    colb=colbar.add_colorbar(ARmap,15.5,0.8)
    colb.set_label('Normalized CL intensity',size=20)
    colb.ax.tick_params(labelsize=18)
    tick_locator = ticker.MaxNLocator(nbins=5)
    colb.locator = tick_locator
    colb.update_ticks()
    ARmap.set_clim(lim1AR,lim2AR)
    plt.axis([wllist1.min(), wllist1.max(), thetalist.min(), thetalist.max()])
    plt.xlabel(xlabelstringAR,fontsize=20)
    plt.ylabel(ylabelstringAR,fontsize=20)
    plt.xticks(fontsize=20) 
    plt.yticks(fontsize=20)
    plt.locator_params(axis = 'x', nbins = 5)  
    plt.locator_params(axis = 'y', nbins = 10)
        
    return ARdata

