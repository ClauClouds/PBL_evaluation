#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 16:22:21 2019

@author : cacquist@meteo.uni-koeln.de
date    : 15 Januar 2019
goal    : read meteogram of german domain and process columns over HOPE campaign domain to:
    - derive cloud mask from different columns
    - derive statistics of CCL height and CB height 
    - derive statistics of wind below cloud base for every cloudy pixel
"""

# ------------------------------------------------------------------------


# ---- importing libraries
import numpy as np
import matplotlib as mpl
import scipy
import numpy.ma as ma
import pandas as pd
import netCDF4 as nc4
import glob
from netCDF4 import Dataset
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import datetime

from myFunctions import f_CCL
from myFunctions import lcl
from myFunctions import f_calcPblHeight
from myFunctions import f_cloudmask
from myFunctions import f_calcWvariance
from myFunctions import f_runningMeanSkewnessW
from myFunctions import f_PBLClass
from myFunctions import f_closest
from myFunctions import f_calcCloudBaseTop
from myFunctions import f_pdfsBelowCloudBase
from myFunctions import f_calcWindSpeed_Dir

# define flags for doing plot (flag=1 is doing the plot, flag=0 skips the plot)
flag_cloudMask          = 0
flag_CBCTtimeSeries     = 0
flag_IWVTimeSeries      = 0
flag_LWPtimeSeries      = 0
flag_histogramWindSpeed = 1
flag_PBLheight          = 0
flag_CCL_LCL            = 0


# setting starting time and ending time for plotting and data processing 
timeStart                          = datetime.datetime(2013,5,2,8,0,0)
timeEnd                            = datetime.datetime(2013,5,2,18,0,0)

# ----- user defined parameters for cloud fraction calculations
timeSpinOver          = 0.0                      # ending time of the spin up of the model (corresponds to the time at which we start to calculate cloud fraction
intTime               = 400.                     # time interval over which calculate cloud fraction and coupling parameters [seconds] corresponding to minutes with a resolution of model output of 9 sec (5 min = 50), (15 min = 150)
QcThreshold           = 10**(-7)                 # threshold in Qc to identify liquid cloud presence
QiThreshold           = 10**(-7)                 # threshold in Qc to identify ice cloud presence   
SigmaWThres           = 0.2                      # threshold in the variance of the vertical velocity, used as a threshold for identifying turbulent regions.
nTimeMean             = 200                      # number of time stamps to be collected to average over 30 min=1800 s
timeStep              = 33                       # time step for running mean
timewindow            = 200                      # time window for calculating running mean of variance corresponding to 30 min with 9 sec resolution (200*9=1800 sec = 30 min)
gradWindThr           = 0.01                     # Threshold for wind gradient to determine wind shear presence in the PBLclas
timeWindowSk          = 33                       # number of time stamps corresponding to 5 minutes in the ICON file
runningWindow         = 200*4                    # number of time stamps corresponding to 30 minutes running window for the average


# ---- defining input/output files and directories
date                  = '20130502'
station               = 'JOYCE'
domain                = 'DOM03'
path_ICON_LEM         = '/work/cacquist/HDCP2_S2/ICON_METEOGRAMS/new/'
path_ICON_INSCAPE     = '/data/inscape/icon/experiments/juelich/meteo-4nest/'
filename_ICON_LEM     = '1d_vars_DOM03_20130502T000000Z-20130503T000000Z.nc'
filename_ICON_INSCAPE = 'METEOGRAM_patch003_20130502_joyce.nc'
pathOut               = '/work/cacquist/HDCP2_S2/PBL_evaluation/'
pathFig               = '/work/cacquist/HDCP2_S2/PBL_evaluation/figs/'

# ---- selecting file to process
filename              = filename_ICON_LEM
path                  = path_ICON_LEM
print('processing file:'+path+filename)


# ---- reading datafile selected
data                  = Dataset(path+filename, mode='r')
dateString            = data.variables['date'][:].copy()
lat                   = data.variables['station_lat'][:].copy()
lon                   = data.variables['station_lon'][:].copy()
station_hsurf         = data.variables['station_hsurf'][:].copy()
station_name          = data.variables['station_name'][:].copy()
time                  = data.variables['time'][:].copy()
datetime_ICON         = nc4.num2date(data.variables['time'][:],data.variables['time'].units)        # --- converting time array in datetime format

print('variables read')


# ---- reading indeces corresponding to meteograms stations of HOPE area 
HOPE_stations               = ['c1','c21','c22','c23','c24','c25','c26','c27','c28','c29','c30']
indexed_stations            = {}


for i in range(station_name.shape[0]):
    stat                    = station_name.data[i,~station_name.mask[i,:]].tostring().decode('ascii')
    if stat in HOPE_stations:
        indexed_stations[stat] = i

print(indexed_stations)


# ---- defining array of dictionaries to be filled with data for the analysis
dictionary_array            = []


for indStat in range(len(HOPE_stations)):
    
    # ---- selecting a station index for testing the program
    stationSelected             = HOPE_stations[indStat]
    indStation                  = indexed_stations[stationSelected]
    print('now processing the station '+stationSelected+' ,'+str(indStation))
    print('-------------------------------------------')
    
    # ---- reading data corresponding to the selected index
    # exctracting variables of interest for PBL classification
    print('reading values/heights/sfcvalues from the selected station')
    heights                     = data.variables['heights'][:,:,indStation].copy()
    values                      = data.variables['values'][:,:,:,indStation].copy()
    sfcvalues                   = data.variables['sfcvalues'][:,:,indStation].copy()
    Hsurf                       = station_hsurf[indStation]
    
    print('reading single variables for values from the selected station')
    Qi                          = values[:,:,10]
    Qc                          = values[:,:,9]
    T                           = values[:,:,1]  # in Kelvin
    zonalWind                   = values[:,:,5]
    merWind                     = values[:,:,6]
    vertWind                    = values[:,:,7]
    LWP                         = sfcvalues[:,49]
    LWPdia                      = sfcvalues[:,54]
    IWV                         = sfcvalues[:,48]
    IWVdia                      = sfcvalues[:,53]
    cloudCover                  = values[:,:,26]
    thetaV                      = values[:,:,4]
    height                      = heights[:,9]
    P                           = values[:,:,0] # pascal units
    RH                          = values[:,:,13]
    q                           = values[:,:,8]
    print('data loaded for '+date + ' ' + station)
    
    # --- reading dimension of height and time arrays
    dimTime                     = len(datetime_ICON)
    dimHeight                   = len(height)

    print('variable extracted from the data')
    
    
    # ------------------------------------------------------------------                
    # derivation of water vapor mixing ratio
    # ------------------------------------------------------------------                
    r = np.zeros((dimTime, dimHeight))
    for itempo in range(dimTime):
        for ih in range(dimHeight):
            r[itempo,ih] = q[itempo,ih]/(1. - q[itempo,ih] )
    print('water vapor mixing ratio calculated')         

    # ------------------------------------------------------------------
    # ----- calculation of cloud mask, cloud base and cloud top for the current column in HOPE area
    # ------------------------------------------------------------------

    cloudMask              = f_cloudmask(time,height,Qc,Qi,QiThreshold,QcThreshold)
    result                 = f_calcCloudBaseTop(cloudMask, len(datetime_ICON), len(height), height)
    CBMatrix_ICON          = result[0]
    CTMatrix_ICON          = result[1]
    CT_array_ICON          = np.empty(len(datetime_ICON))
    CB_array_ICON          = np.empty(len(datetime_ICON))
    CT_array_ICON[:]       = np.nan
    CB_array_ICON[:]       = np.nan
    
    for indT in range(len(datetime_ICON)):#
        if (~np.isnan(CBMatrix_ICON[indT,0]) == True) and (~np.isnan(CTMatrix_ICON[indT,0])== True):
            
            indCB          = f_closest(height, CBMatrix_ICON[indT,0])
            indCT          = f_closest(height, CTMatrix_ICON[indT,0])
            
            if (indCB == 0) or (indCT == 0):
                CT_array_ICON[indT] = np.nan
                CB_array_ICON[indT] = np.nan
            else:
                CT_array_ICON[indT] = height[indCT]                                 # saving cloud top height
                CB_array_ICON[indT] = height[indCB]                                 # saving cloud base height
                
    print('cloud base and cloud top for ICON-LEM calculated ')
    
    # ------------------------------------------------------------------
    # ---- calculating CCL height time serie
    # ------------------------------------------------------------------
    result_ICON = f_CCL(T, P, RH, height, datetime_ICON, Hsurf)
    T_ccl_ICON = result_ICON['T_ccl'][:]
    z_ccl_ICON = result_ICON['z_ccl'][:]
    print('CCL height and temperature for ICON-LEM calculated ')
    
    
    # ------------------------------------------------------------------
    # --- calculation of the LCL 
    # ------------------------------------------------------------------
    # determining P, T and RH at the surface 
    iSurf =f_closest(height, Hsurf)
    
    Psurf = P[:,iSurf]
    Tsurf = T[:,iSurf]
    RHsurf = RH[:,iSurf]
    LCLarray = []
    for iTime in range(dimTime):
        LCLarray.append(lcl(Psurf[iTime],Tsurf[iTime],RHsurf[iTime]/100.))
    print('LCL calculated')   
                

    
    # --------------------------------------------------------------------
    # ----- calculation of virtual potential temperature profiles for ICON, ICON_INSCAPE, COSMO, radiosondes
    # --------------------------------------------------------------------
    Rd = 287.058  # gas constant for dry air [Kg-1 K-1 J]
    Cp = 1004.
    
    # calculating profiles of virtual potential temperature for model output
    Theta_v_ICON = np.zeros((len(datetime_ICON), len(height)))
    
    # calculating virtual potential temperature in ICON and COSMO regridded
    for indTime in range(len(datetime_ICON)):
        for indHeight in range(len(height)):
                k_ICON = Rd*(1-0.23*r[indTime, indHeight])/Cp
                Theta_v_ICON[indTime, indHeight]= ( (1 + 0.61 * r[indTime, indHeight]) * \
                                                   T[indTime, indHeight] * (100000./P[indTime, indHeight])**k_ICON)
    

    # ------------------------------------------------------------------
    # --- Calculating Boundary layer height using the richardson number derivation according to Seidel Et al, 2010
    # ------------------------------------------------------------------
    PBLHeightArr=f_calcPblHeight(Theta_v_ICON,zonalWind,merWind,height,datetime_ICON)
    print('height of the PBL calculated')       
    

    
    # ----------------------------------------------------------------------------------------
    # ----------- Calculation of mean cloud base in model for every hour of the day ----------------
    # ----------------------------------------------------------------------------------------
    
    # calculating mean cloud base height for icon
    CB_ICON_DF               = pd.DataFrame(CB_array_ICON, index=datetime_ICON)
    mean_CB_arr_ICON         = []
    datetimeHourArr          = []
    for indHour in range(0,23):
        HourSup              = datetime.datetime(2013, 5, 2, indHour+1, 0, 0)
        HourInf              = datetime.datetime(2013, 5, 2, indHour, 0, 0)
        CB_ICON_DF_sliced_t  = CB_ICON_DF.loc[(CB_ICON_DF.index < HourSup) * (CB_ICON_DF.index > HourInf)]
        datetimeHourArr.append(HourInf)
    
        if ~np.isfinite(np.nanmean(CB_ICON_DF_sliced_t)):
            mean_CB_arr_ICON.append(np.nan)
        else:
            mean_CB_arr_ICON.append(height[f_closest(height, np.nanmean(CB_ICON_DF_sliced_t))] )
    
    
    
    # ----------------------------------------------------------------------------------------
    # calculation of wind direction and intensity for model output
    # ----------------------------------------------------------------------------------------
    windData_ICON             = f_calcWindSpeed_Dir(datetime_ICON, height, merWind, zonalWind)
    print('wind speed and direction calculated for ICON-LEM ')
    
    
    # ----------------------------------------------------------------------------------------
    # selecting PBL cloud bases by threshold on the difference between cloud base heigth and PBL height 
    CB_PBL_ICON = CB_array_ICON
    diff_abs = np.abs(CB_array_ICON - PBLHeightArr)
    Thr_PBL_CB = 1000
    CB_PBL_ICON[diff_abs > Thr_PBL_CB] = np.nan
    plt.plot(datetime_ICON, CB_array_ICON, color='blue')
    plt.plot(datetime_ICON, CB_PBL_ICON, color = 'red')
    
    # ------------------------------------------------------------------
    # ---- deriving distributions of vertical wind values below clouds and in no cloudy regions    
    # ------------------------------------------------------------------
    
    Pdfs_wind_cloudNoCloud = f_pdfsBelowCloudBase(vertWind, windData_ICON[2], datetime_ICON, datetimeHourArr, height, PBLHeightArr, CB_PBL_ICON, timeStart, timeEnd)
    
    print('pdfs of wind below cloud base and at mean cloud base height where there is no cloud - done')
    

    # ------------------------------------------------------------------
    # ----  saving data for the station in a dictionary   
    # ------------------------------------------------------------------
    dictionary_array.append({'stationName':stationSelected,\
                             'LWP':LWP, \
                             'IWV':IWV, \
                             'datetime_ICON':datetime_ICON, \
                             'CB':CB_array_ICON,\
                             'CCLheight':z_ccl_ICON,\
                             'T_ccl':T_ccl_ICON, \
                             'Tsurf':T[:,149], \
                             'PDF_Wwind_noCloud':Pdfs_wind_cloudNoCloud[1], \
                             'PDF_Wwind_cloud':Pdfs_wind_cloudNoCloud[0],\
                             'LCLheight':LCLarray, \
                             'PBLheight':PBLHeightArr})
    print('data saved in the dictionary for the selected station, end of processing for this station')
    
    
    
    
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # PLOTTING SECTION 
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    print('start plotting quantities selected by flags')

    # ---- plot of LWP 
    if flag_LWPtimeSeries == 1:
        fig, ax                            = plt.subplots(figsize=(10,5))
        label_size                         = 16
        mpl.rcParams['xtick.labelsize']    = label_size 
        mpl.rcParams['ytick.labelsize']    = label_size
        mpl.rcParams['savefig.dpi'] = 100
        plt.gcf().subplots_adjust(bottom= 0.15)
        fig.tight_layout()
        ax                                 = plt.subplot(111)  
        ax.spines["top"].set_visible(False)  
        ax.spines["right"].set_visible(False)  
        ax.get_xaxis().tick_bottom()  
        ax.get_yaxis().tick_left()  
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        plt.title(' LWP time serie', fontsize=18)
        plt.ylabel(' LWP [g/m^2]', fontsize=16)
        #plt.ylim(282.5, 300.)
        plt.xlabel('time [hh:mm]', fontsize=16)
        plt.xlim(timeStart,timeEnd)
        plt.plot(datetime_ICON, LWP*1000., label='ICON-LEM', color='red')
        plt.legend(loc='upper left', fontsize=12)
        plt.tight_layout()
        plt.savefig(pathFig+'LWP_timeSerie_20130502_'+stationSelected+'.pdf', format='pdf')
    
    # ---- plot of IWV
    if flag_IWVTimeSeries == 1:
        fig, ax                            = plt.subplots(figsize=(10,5))
        label_size                         = 16
        mpl.rcParams['xtick.labelsize']    = label_size 
        mpl.rcParams['ytick.labelsize']    = label_size
        mpl.rcParams['savefig.dpi'] = 100
        plt.gcf().subplots_adjust(bottom= 0.15)
        fig.tight_layout()
        ax                                 = plt.subplot(111)  
        ax.spines["top"].set_visible(False)  
        ax.spines["right"].set_visible(False)  
        ax.get_xaxis().tick_bottom()  
        ax.get_yaxis().tick_left()  
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        plt.title(' IWV time serie', fontsize=18)
        plt.ylabel(' IWV [Kg/m^2]', fontsize=16)
        #plt.ylim(282.5, 300.)
        plt.xlabel('time [hh:mm]', fontsize=16)
        plt.xlim(timeStart,timeEnd)
        plt.plot(datetime_ICON, IWV, label='ICON-LEM', color='red')
        plt.legend(loc='upper left', fontsize=12)
        plt.tight_layout()
        plt.savefig(pathFig+'IWV_timeSerie_20130502_'+stationSelected+'.pdf', format='pdf')
        

    
    # ---- plot of cloud base and cloud top time series in the PBL
    if flag_CBCTtimeSeries == 1:
        fig, ax                            = plt.subplots(figsize=(10,5))
        label_size                         = 16
        mpl.rcParams['xtick.labelsize']    = label_size 
        mpl.rcParams['ytick.labelsize']    = label_size
        mpl.rcParams['savefig.dpi'] = 100
        plt.gcf().subplots_adjust(bottom= 0.15)
        fig.tight_layout()
        ax                                 = plt.subplot(111)  
        ax.spines["top"].set_visible(False)  
        ax.spines["right"].set_visible(False)  
        ax.get_xaxis().tick_bottom()  
        ax.get_yaxis().tick_left()  
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        plt.title(' cloud base/cloud top time series', fontsize=18)
        plt.ylabel(' height [m]', fontsize=16)
        plt.ylim(0., 5000.)
        plt.xlabel('time [hh:mm]', fontsize=16)
        plt.xlim(timeStart,timeEnd)
        plt.plot(datetime_ICON, CB_array_ICON, label='cloud base', marker='v', linestyle='none', color='red')
        plt.plot(datetime_ICON, CT_array_ICON, label='cloud top', marker='v', linestyle='none', color='blue')
        plt.plot(datetime_ICON, z_ccl_ICON, label='CCL height', marker='v', linestyle='--', color='black')
        #plt.plot(datetime_ICON, z_ccl_ICON, label='CCL height', marker='v', linestyle='none', color='black')
        plt.plot(datetimeHourArr, mean_CB_arr_ICON, label='mean hourly cloud base',  linestyle='-', color='black')
        plt.plot(datetime_ICON, LCLarray, label='LCL height', marker = 'v', linestyle = 'none', color = 'orange')
        plt.legend(loc='upper left', fontsize=12)
        plt.tight_layout()
        plt.savefig(pathFig+'CB_CT_timeSerie_20130502_'+stationSelected+'.pdf', format='pdf')
        
    # plot of CCL and LCL heights 
    if flag_CCL_LCL == 1:
        fig, ax                            = plt.subplots(figsize=(10,5))
        label_size                         = 16
        mpl.rcParams['xtick.labelsize']    = label_size 
        mpl.rcParams['ytick.labelsize']    = label_size
        mpl.rcParams['savefig.dpi'] = 100
        plt.gcf().subplots_adjust(bottom= 0.15)
        fig.tight_layout()
        ax                                 = plt.subplot(111)  
        ax.spines["top"].set_visible(False)  
        ax.spines["right"].set_visible(False)  
        ax.get_xaxis().tick_bottom()  
        ax.get_yaxis().tick_left()  
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.title(' cloud base/cloud top time series', fontsize=18)
        plt.ylabel(' height [m]', fontsize=16)
        plt.ylim(0., 5000.)
        plt.xlabel('time [hh:mm]', fontsize=16)
        plt.xlim(timeStart,timeEnd)
        plt.plot(datetime_ICON, CB_array_ICON, label='cloud base', marker='v', linestyle='none', color='black')        
        plt.plot(datetime_ICON, LCLarray, label='LCL height', linestyle = '-', color = 'blue')
        plt.plot(datetime_ICON, z_ccl_ICON, label='CCL height', linestyle='-', color='red')
        plt.legend(loc='upper left', fontsize=12)
        plt.tight_layout()
        plt.savefig(pathFig+'CCL_LCL_timeSerie_20130502_'+stationSelected+'.pdf', format='pdf') 
        
    if flag_PBLheight == 1:
        fig, ax                            = plt.subplots(figsize=(10,5))
        label_size                         = 16
        mpl.rcParams['xtick.labelsize']    = label_size 
        mpl.rcParams['ytick.labelsize']    = label_size
        mpl.rcParams['savefig.dpi'] = 100
        plt.gcf().subplots_adjust(bottom= 0.15)
        fig.tight_layout()
        ax                                 = plt.subplot(111)  
        ax.spines["top"].set_visible(False)  
        ax.spines["right"].set_visible(False)  
        ax.get_xaxis().tick_bottom()  
        ax.get_yaxis().tick_left()  
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.title(' cloud base/cloud top and PBL height time series', fontsize=18)
        plt.ylabel(' height [m]', fontsize=16)
        plt.ylim(0., 5000.)
        plt.xlabel('time [hh:mm]', fontsize=16)
        plt.xlim(timeStart,timeEnd)
        plt.plot(datetime_ICON, CB_array_ICON, label='cloud base', marker='v', linestyle='none', color='red')        
        plt.plot(datetime_ICON, PBLHeightArr, label='PBL height', linestyle = '-', color = 'blue')
        plt.legend(loc='upper left', fontsize=12)
        plt.tight_layout()
        plt.savefig(pathFig+'PBL_CB_timeSerie_20130502_'+stationSelected+'.pdf', format='pdf') 
                
        
    # ---- plot of cloudmask time height colormesh plot
    if flag_cloudMask == 1:
        fig, ax                             = plt.subplots(figsize=(10,4))
        mpl.rcParams['xtick.labelsize']    = label_size 
        mpl.rcParams['ytick.labelsize']    = label_size
        cax                                 = ax.pcolormesh(datetime_ICON, height, cloudMask.transpose(), vmin=0, vmax=3, cmap=plt.cm.get_cmap("RdPu", 4))
        ax.set_ylim(0,5000.)                                               # limits of the y-axes
        ax.set_xlim(timeStart,timeEnd)
        ax.set_title("cloud mask", fontsize=14)
        ax.set_xlabel("time ", fontsize=12)
        ax.set_ylabel("height [m]", fontsize=12)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        cbar                                = fig.colorbar(cax, ticks=[0, 1, 2, 3], orientation='vertical')
        cbar.ticks=([0,1,2,3])
        cbar.ax.set_yticklabels(['no cloud','liquid','ice', 'mixed phase'])
        cbar.set_label(label="cloud type",size=12)
        cbar.ax.tick_params(labelsize=12)
        cbar.aspect=80
        plt.savefig(pathFig+'cloudMask_20130502_'+stationSelected+'.pdf', format='pdf')
        
        
        
    # ---- plot of pdfs of vertical wind below cloud base and on clear air
    if flag_histogramWindSpeed == 1:
        nbins     = 30
        fig, ax   = plt.subplots(figsize=(14,6))
        ax        = plt.subplot(111)
        ax.spines["top"].set_visible(False)  
        ax.spines["right"].set_visible(False)  
        alphaval = 0.5
        ax.get_xaxis().tick_bottom()  
        ax.get_yaxis().tick_left() 
        ax.tick_params(labelsize=16)
        plt.hist(Pdfs_wind_cloudNoCloud[0], bins=nbins, range=[-3., 3], normed=True, color='red', cumulative=False, alpha=alphaval, label='ICON-LEM cloud')
        plt.hist(Pdfs_wind_cloudNoCloud[1], bins=nbins, range=[-3., 3], normed=True, color='blue', cumulative=False, alpha=alphaval, label='ICON-LEM no cloud')
        plt.legend(loc='upper right', fontsize=12)#range=[np.nanmin(LWP_obs_PBL),np.nanmax(LWP_obs_PBL)]
        plt.title('vertical wind below cloud base - PDFs', fontsize=16)
        plt.xlabel('vertical wind below cloud base [m/s]', fontsize=16)
        plt.ylabel('Occurrences', fontsize=16)
        plt.savefig(pathFig+'histogram_WwindSpeedbelowCloudBase_ICON_LEM_20130502_'+stationSelected+'.pdf', format='pdf')  
    
    
        
    
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # END OF PLOTTING SECTION 
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
    print('end of plotting for this station, go to next station')
    print('-----------------------------------------------------------------------------------------')
    
    
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ---- END OF LOOP ON HOPE METEOGRAMS : PROCESSING NOW FOR STATISTICS OVER HOPE REGION
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ---- calculating statistics for vertical wind below cloud base in meteograms HOPE
PDF_no_cloud_tot = []
PDF_cloud_tot    = []
CCL_height_tot   = []
PBL_height_tot   = []
LCL_height_tot   = []
CB_height_tot    = []

for indStat in range(len(HOPE_stations)):

   PDF_no_cloud_tot.append(dictionary_array[indStat]['PDF_Wwind_noCloud'])
   PDF_cloud_tot.append(dictionary_array[indStat]['PDF_Wwind_cloud'])
   CCL_height_tot.append(dictionary_array[indStat]['PDF_Wwind_cloud'])

median_no_cloud = np.nanmedian(np.concatenate(PDF_no_cloud_tot))
median_cloud = np.nanmedian(np.concatenate(PDF_cloud_tot))
mean_no_cloud = np.nanmean(np.concatenate(PDF_no_cloud_tot))
mean_cloud = np.nanmean(np.concatenate(PDF_cloud_tot))
std_no_cloud = np.nanstd(np.concatenate(PDF_no_cloud_tot))
std_cloud = np.nanstd(np.concatenate(PDF_cloud_tot))
percentiles_no_cloud = np.nanpercentile(np.concatenate(PDF_no_cloud_tot), [50, 75, 90])
percentiles_cloud = np.nanpercentile(np.concatenate(PDF_cloud_tot),[50, 75, 90])

# ----calculating moments of the distributions of wind values
Moments_w_noCloud = scipy.stats.moment(np.concatenate(PDF_no_cloud_tot), nan_policy='omit')
nbins     = 40
fig, ax   = plt.subplots(figsize=(14,6))
ax        = plt.subplot(111)
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
alphaval = 0.5
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left() 
ax.tick_params(labelsize=16)
plt.hist(np.concatenate(PDF_no_cloud_tot), bins=nbins, range=[-4., 4], normed=True, color='blue', cumulative=False, alpha=alphaval, label='ICON-LEM no cloud')
plt.hist(np.concatenate(PDF_cloud_tot), bins=nbins, range=[-4., 4], normed=True, color='red', cumulative=False, alpha=alphaval, label='ICON-LEM cloud')
plt.legend(loc='upper right', fontsize=12)#range=[np.nanmin(LWP_obs_PBL),np.nanmax(LWP_obs_PBL)]
plt.title('vertical wind below cloud base - PDFs', fontsize=16)
plt.xlabel('vertical wind below cloud base [m/s]', fontsize=16)
plt.ylabel('Occurrences', fontsize=16)
plt.savefig(pathFig+'histogram_WwindSpeedbelowCloudBase_Total_ICON_LEM_20130502.pdf', format='pdf')  

