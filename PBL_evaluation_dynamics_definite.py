
# coding: utf-8

#Created on Thu Apr 12 16:01:09 2018
#goal : this code is supposed to analyze ICON-LEM and observations data for the 2nd of may
#tasks:
#    1) we read all imput data and convert all different time formats to datetime 
#    2) we resize all observations on the time scale of the model output, which is 9 sec. This is done by:
#    upsampling or downsampling the original datasets

#@author: cacquist


# In[1]:




#!pip install plotly
# ----- importing libraries needed
import numpy as np
import matplotlib
import scipy
import pylab
import netCDF4 as nc4
import numpy.ma as ma
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import struct
import glob
import pandas as pd
import datetime as dt
import random
import datetime
import matplotlib.dates as mdates

#import plotly.plotly as py  # tools to communicate with Plotly's server
#import xarray as xr
from numpy import convolve
from cloudnetFunctions import CheckIceLiquidCloudnet
from cloudnetFunctions import f_calculateCloudMaskCloudnet

from myFunctions import f_closest
from myFunctions import getNearestIndex
from myFunctions import getIndexList
from myFunctions import getIndexListsuka
from myFunctions import getResampledDataPd
from myFunctions import hourDecimal_to_datetime
from myFunctions import f_calcTheta_MWR
from myFunctions import f_readingTowerData
from myFunctions import f_calcPblHeight
from myFunctions import f_resamplingMatrix
from myFunctions import f_pdfsBelowCloudBase
from myFunctions import f_plotTimeHeightColorMaps
from myFunctions import f_calcWindSpeed_Dir
from myFunctions import f_plotWindRose

#get_ipython().magic('matplotlib inline')


# In[2]:
# ----------------------------------------------------------------------------------------
# reading all input datasets from ICON-LEM, ICON-INSCAPE, OBS
# ----------------------------------------------------------------------------------------


#------ providing input directories
PathIn                = '/Users/cacquist/Lavoro/PBL_hdcp2_s2/PBL_hdcp2_s2_minion/data/'
#'/Volumes/CLAMINION/PBL_hdcp2_s2/'
#PathIn = '/media/cacquist/CLAMINION/PBL_hdcp2_s2/'

year                  = 2013
month                 = 5
day                   = 2
date                  = '20130502'
Hcut                  = 3500.
Hsurf                 = 107.
pathFig               = '/Users/cacquist/Lavoro/PBL_hdcp2_s2/PBL_hdcp2_s2_minion/figs/figs_selected/'

# ----- reading ICON, CLOUDNET and PBL data
ICON_data             = Dataset(PathIn+'PBL_properties_DOM03_20130502_JOYCE.nc', mode='r')
ICON_vera             = Dataset(PathIn+'PBL_properties_DOM03_20130502_JOYCE_INSCAPE.nc', mode='r')
OBS_PBL_data          = Dataset(PathIn+'20130502_bl_classification_juelich_t_3min.nc', mode='r')
CLOUDNET_data         = Dataset(PathIn+'20130502_juelich_categorize.nc', mode='r')
#MWR_data = Dataset(PathIn+'sups_joy_mwr00_l2_clwvi_v01_20130502000018.nc', mode='r')
#PRW_data = Dataset(PathIn+'sups_joy_mwr00_l2_prw_v01_20130502000018.nc', mode='r')
#HUA_data = Dataset(PathIn+'sups_joy_mwr00_l2_hua_v01_20130502000018.nc', mode='r')
#TA_data = Dataset(PathIn+'sups_joy_mwr00_l2_ta_v01_20130502000018.nc', mode='r')
# reading data from SUNHAT zenith pointing during HOPE
MWR_data              = Dataset(PathIn+'130502_hps_l2a.nc', mode='r')
PRW_data              = Dataset(PathIn+'130502_hps_l2a.nc', mode='r')
HUA_data              = Dataset(PathIn+'130502_hph_l2b.nc', mode='r')
TA_data               = Dataset(PathIn+'130502_hph_l2b.nc', mode='r')
RH_data               = Dataset(PathIn+'130502_hph_l2c.nc', mode='r')

# ----- reading PBL class data variables
time_PBL_class        = OBS_PBL_data.variables['time'][:].copy()
height_PBL_class      = OBS_PBL_data.variables['height'][:].copy()
PBLclass              = OBS_PBL_data.variables['bl_classification'][:].copy()
#PBLclass = PBLclass[np.where(time_PBL_class < 2.)[0], :]
datetime_PBL          = hourDecimal_to_datetime(year, month, day, time_PBL_class)
beta                  = OBS_PBL_data.variables['beta'][:].copy()
eps                   = OBS_PBL_data.variables['eps'][:].copy()
shear                 = OBS_PBL_data.variables['shear'][:].copy()
skew                  = OBS_PBL_data.variables['skew'][:].copy()
w                     = OBS_PBL_data.variables['velo'][:].copy()
Hwind                 = OBS_PBL_data.variables['speed'][:].copy()
wDir                  = OBS_PBL_data.variables['dir'][:].copy()


# ----- reading MWR radiometer data variables ( varnames for HDCP2 format)
#time_MWR = MWR_data.variables['time'][:].copy()
#LWP_MWR = MWR_data.variables['clwvi'][:].copy()    # liquid water path [kg m^-2]
# creating time array for LWP in date time format
#TA_MWR = TA_data.variables['ta'][:].copy()         # air temperature [K]
#HUA_MWR = HUA_data.variables['hua'][:].copy()      # absolute humidity [kg m^-3]
#IWV_MWR = PRW_data.variables['prw'][:].copy()      # integrated water vapor [kg m^-2]
#datetime_IWV = nc4.num2date(PRW_data.variables['time'][:],PRW_data.variables['time'].units)
#datetime_HUA = nc4.num2date(HUA_data.variables['time'][:],HUA_data.variables['time'].units)
#datetime_MWR = nc4.num2date(MWR_data.variables['time'][:],MWR_data.variables['time'].units) 
#datetime_TA = nc4.num2date(TA_data.variables['time'][:],TA_data.variables['time'].units) 
#height_HUA = HUA_data.variables['height'][:].copy()
#height_TA = TA_data.variables['height'][:].copy()

# ----- reading MWR radiometer data variables for la and lb formats
time_MWR              = MWR_data.variables['time'][:].copy()
LWP_MWR               = MWR_data.variables['atmosphere_liquid_water_content'][:].copy()    # liquid water path [kg m^-2]
# creating time array for LWP in date time format
TA_MWR                = TA_data.variables['tprof'][:].copy()         # air temperature [K]
HUA_MWR               = HUA_data.variables['qprof'][:].copy()      # absolute humidity [kg m^-3]
IWV_MWR               = PRW_data.variables['atmosphere_water_vapor_content'][:].copy()      # integrated water vapor [kg m^-2]
datetime_IWV          = nc4.num2date(PRW_data.variables['time'][:],PRW_data.variables['time'].units)
datetime_HUA          = nc4.num2date(HUA_data.variables['time'][:],HUA_data.variables['time'].units)
datetime_MWR          = nc4.num2date(MWR_data.variables['time'][:],MWR_data.variables['time'].units) 
datetime_TA           = nc4.num2date(TA_data.variables['time'][:],TA_data.variables['time'].units) 
height_HUA            = HUA_data.variables['z'][:].copy()
height_TA             = TA_data.variables['z'][:].copy()


# reading relative humidity values at the ground from the microwave radiometer
time_RH_MWR           = RH_data.variables['time'][:].copy()
height_RH_MWR         = RH_data.variables['z'][:].copy()
RH_MWR                = RH_data.variables['relative_humidity_profile'][:].copy()
datetime_RH_MWR       = nc4.num2date(RH_data.variables['time'][:],RH_data.variables['time'].units)


# ----- reading tower measurements
tower_dict            = f_readingTowerData(date, PathIn)
P_surf                = tower_dict['P'][:]
T_surf                = tower_dict['Tsurf'][:]
RH_surf               = tower_dict['RHsurf'][:]
datetime_tower        = tower_dict['time'][:]


# ----- reading ICON data variables
time_ICON             = ICON_data.groups['Temp_data'].variables['datetime_ICON'][:].copy()
cloudMask_ICON        = ICON_data.groups['Temp_data'].variables['cloudMask'][:]
datetime_ICON         = nc4.num2date(ICON_data.groups['Temp_data'].variables['datetime_ICON'][:],                             ICON_data.groups['Temp_data'].variables['datetime_ICON'].units) 
height_ICON           = ICON_data.groups['Temp_data'].variables['height'][:].copy()
skewness_ICON         = ICON_data.groups['Temp_data'].variables['skewnessW'][:].copy()
theta_ICON            = ICON_data.groups['Temp_data'].variables['theta'][:].copy()
thetal_ICON           = ICON_data.groups['Temp_data'].variables['theta_liquid'][:].copy()
varw_ICON             = ICON_data.groups['Temp_data'].variables['varianceW'][:].copy()
Hwind_ICON            = ICON_data.groups['Temp_data'].variables['windSpeed'][:].copy()
alphaq_ICON           = ICON_data.groups['Temp_data'].variables['alpha_q'][:].copy()
alphatheta_ICON       = ICON_data.groups['Temp_data'].variables['alpha_q'][:].copy()
cloudBase_ICON        = ICON_data.groups['Temp_data'].variables['cloudBaseHeightArr'][:].copy()
LTS_ICON              = ICON_data.groups['Temp_data'].variables['LTS'][:].copy()
LCL_ICON              = ICON_data.groups['Temp_data'].variables['LCLarray'][:].copy()
w_ICON                = ICON_data.groups['Temp_data'].variables['vertWind'][:].copy()
RH_ICON               = ICON_data.groups['Temp_data'].variables['RH'][:].copy()
mixingRatio_ICON      = ICON_data.groups['Temp_data'].variables['r'][:].copy()
specHum_ICON          = ICON_data.groups['Temp_data'].variables['q'][:].copy()
Hsurf                 = ICON_data.groups['Temp_data'].variables['Height_surface'][:].copy()
IWV_ICON              = ICON_data.groups['Temp_data'].variables['IWV'][:].copy()
LWP_ICON              = ICON_data.groups['Temp_data'].variables['LWP'][:].copy()
P_ICON                = ICON_data.groups['Temp_data'].variables['P'][:].copy()
PBL_class_ICON        = ICON_data.groups['Temp_data'].variables['PBLclass'][:].copy()
PBL_height_ICON       = ICON_data.groups['Temp_data'].variables['PBLHeightArr'][:].copy()
T_ICON                = ICON_data.groups['Temp_data'].variables['T'][:].copy()
merWind               = ICON_data.groups['Temp_data'].variables['merWind'][:].copy()
zonalWind             = ICON_data.groups['Temp_data'].variables['zonalWind'][:].copy()
P                     = ICON_data.groups['Temp_data'].variables['P'][:].copy()  # in Pa

# ----- reading ICON data variables from Vera's file
time_ICON_vera        = ICON_vera.groups['Temp_data'].variables['datetime_ICON'][:].copy()
cloudMask_ICON_vera   = ICON_vera.groups['Temp_data'].variables['cloudMask'][:]
datetime_ICON_vera    = nc4.num2date(ICON_vera.groups['Temp_data'].variables['datetime_ICON'][:],                                  ICON_vera.groups['Temp_data'].variables['datetime_ICON'].units)
height_ICON_vera      = ICON_vera.groups['Temp_data'].variables['height2'][:].copy()
skewness_ICON_vera    = ICON_vera.groups['Temp_data'].variables['skewnessW'][:].copy()
theta_ICON_vera       = ICON_vera.groups['Temp_data'].variables['theta'][:].copy()
thetal_ICON_vera      = ICON_vera.groups['Temp_data'].variables['theta_liquid'][:].copy()
varw_ICON_vera        = ICON_vera.groups['Temp_data'].variables['varianceW'][:].copy()
alphaq_ICON_vera      = ICON_vera.groups['Temp_data'].variables['alpha_q'][:].copy()
alphatheta_ICON_vera  = ICON_vera.groups['Temp_data'].variables['alpha_q'][:].copy()
cloudBase_ICON_vera   = ICON_vera.groups['Temp_data'].variables['cloudBaseHeightArr'][:].copy()
LTS_ICON_vera         = ICON_vera.groups['Temp_data'].variables['LTS'][:].copy()
LCL_ICON_vera         = ICON_vera.groups['Temp_data'].variables['LCLarray'][:].copy()
w_ICON_vera           = ICON_vera.groups['Temp_data'].variables['vertWind'][:].copy()
RH_ICON_vera          = ICON_vera.groups['Temp_data'].variables['RH'][:].copy()
mixingRatio_ICON_vera = ICON_vera.groups['Temp_data'].variables['r'][:].copy()
specHum_ICON_vera     = ICON_vera.groups['Temp_data'].variables['q'][:].copy()
Hsurf_vera            = ICON_vera.groups['Temp_data'].variables['Height_surface'][:].copy()
IWV_ICON_vera         = ICON_vera.groups['Temp_data'].variables['IWV'][:].copy()
LWP_ICON_vera         = ICON_vera.groups['Temp_data'].variables['LWP'][:].copy()
P_ICON_vera           = ICON_vera.groups['Temp_data'].variables['P'][:].copy()
PBL_height_ICON_vera  = ICON_vera.groups['Temp_data'].variables['PBLHeightArr'][:].copy()
T_ICON_vera           = ICON_vera.groups['Temp_data'].variables['T'][:].copy()
merWind_vera          = ICON_vera.groups['Temp_data'].variables['merWind'][:].copy()
zonalWind_vera        = ICON_vera.groups['Temp_data'].variables['zonalWind'][:].copy()
shear_ICON_vera       = ICON_vera.groups['Temp_data'].variables['shearHwind'][:].copy()


# ----- reading CLOUDNET data variables
time_CLOUDNET         = CLOUDNET_data.variables['time'][:].copy()
height_CLOUDNET       = CLOUDNET_data.variables['height'][:].copy()
datetime_CLOUDNET     = hourDecimal_to_datetime(year, month, day, time_CLOUDNET)
cloudnet              = CLOUDNET_data.variables['category_bits'][:].copy()
Ze_CLOUDNET           = CLOUDNET_data.variables['Z'][:].copy()
Vd_CLOUDNET           = CLOUDNET_data.variables['v'][:].copy()
Sw_CLOUDNET           = CLOUDNET_data.variables['width'][:].copy()
P_CLOUDNET            = CLOUDNET_data.variables['pressure'][:].copy()
T_CLOUDNET            = CLOUDNET_data.variables['temperature'][:].copy()
Q_CLOUDNET            = CLOUDNET_data.variables['specific_humidity'][:].copy()
model_Height_CLOUDNET = CLOUDNET_data.variables['model_height'][:].copy()

# ---- reading data from radiosondes
#z_ccl_radiosondes     = [948.1, 938.1, 968.4, 1400.8, 1455.7, 1406.9, 1464.3]
#T_ccl_radiosondes     = [289.47138, 289.23338, 289.8803, 291.78784, 292.29586, 292.47762, 292.15014]
#Td_surf_radiosondes   = [ 282.15,  281.75,  281.45,  281.95,  281.85,  280.45,  279.45]
#time_ccl_radiosondes  = ['2013-05-02 07:00:00', '2013-05-02 09:00:00', '2013-05-02 11:00:00',                        '2013-05-02 13:00:00', '2013-05-02 15:00:00', '2013-05-02 17:00:00', '2013-05-02 23:00:00']
#a                     = pd.to_datetime(time_ccl_radiosondes)
#time_ccl_rs           = np.array(a,dtype=np.datetime64)

# ---- reading data from radiosondes
RadiosondesData = Dataset(PathIn+'AllRadiosondesData.nc', mode='r')
z_ccl_radiosondes = RadiosondesData.variables['z_cclTemp'][:].copy()#[948.1, 938.1, 968.4, 1400.8, 1455.7, 1406.9, 1464.3]
T_ccl_radiosondes = RadiosondesData.variables['T_cclTemp'][:].copy()#[289.47138, 289.23338, 289.8803, 291.78784, 292.29586, 292.47762, 292.15014]
Td_surf_radiosondes = [ 282.15,  281.75,  281.45,  281.95,  281.85,  280.45,  279.45]
time_ccl_radiosondes = ['2013-05-02 07:00:00', '2013-05-02 09:00:00', '2013-05-02 11:00:00',\
                        '2013-05-02 13:00:00', '2013-05-02 15:00:00', '2013-05-02 17:00:00', '2013-05-02 23:00:00']
z_lcl_radiosondes = RadiosondesData.variables['z_lclTemp'][:].copy()
Td_radiosondes = RadiosondesData.variables['Td'][:].copy()
T_radiosondes = RadiosondesData.variables['T'][:].copy()
T_surf_radiosondes = T_radiosondes[0,:]
Theta_v_radiosondes = RadiosondesData.variables['theta_v'][:].copy()
EIS1_radiosondes = RadiosondesData.variables['EIS1'][:].copy()
EIS2_radiosondes = RadiosondesData.variables['EIS2'][:].copy()
LTS_radiosondes = RadiosondesData.variables['LTS'][:].copy()
EIS_atmos_radiosondes = RadiosondesData.variables['EIS_atmos'][:].copy()
height_radiosondes = RadiosondesData.variables['height'][:]
a=pd.to_datetime(time_ccl_radiosondes)
time_ccl_rs = np.array(a,dtype=np.datetime64)

# ---- reading data from radiosondes ESSEN 
#PRES   HGHT   TEMP   DWPT   RELH   MIXR   DRCT   SKNT   THTA   THTE   THTV
cols = ['P [hPa]','Hˆhe [m]','T [C]','dew [C]','RH [%]','r [%]','Wdir [∞]', 'Wind speed [m/s]','pot_T [K]','Eq_pot_T [K]', 'virtual_pot_T [K]' ]
DF_essen = pd.read_csv(PathIn+'ESSEN_radiosonde_2013050212.txt', delim_whitespace=True,  names=cols, header=None)
Theta_v_RadiosondeEssen = DF_essen.values[:,10]
height_essen = DF_essen.values[:,1]
height_RadiosondeEssen = np.zeros(len(height_essen))

height_essen[0] = 0.
for ind in range(len(height_essen)-2):
    height_RadiosondeEssen[ind+1] = float(height_essen[ind+1])

height_RadiosondeEssen = np.asarray(height_RadiosondeEssen)


print('read all imput data')


# In[4]:

# ----------------------------------------------------------------------------------------
# Resampling of dataset on ICON time resolution
# ----------------------------------------------------------------------------------------
# variables resampled: 
# - vertical velocity, 
# - horizontal wind speed,
# - wind direction, 
# - wind shear, 
# - tower obs
# - temperature mwr
# - abs humidity mwr
# - rel humidity mwr
# - T, P, spec humidity from COSMO
# - wind shear obs wind lidar
# - variance of vertical velocity (obs)
# - vertical wind speed 
# ----------------------------------------------------------------------------------------

# ---- defining ICON data as dataframe reference
ICON_DF           = pd.DataFrame(cloudMask_ICON, index=datetime_ICON, columns=height_ICON)

# ---- resampling PBL classification on ICON resolution
print('resampling PBL observations on ICON time resolution')
PBL_DF            = pd.DataFrame(PBLclass, index=datetime_PBL, columns=height_PBL_class)
SelectedIndex_PBL = getIndexList(PBL_DF, ICON_DF.index)

# ---- resampling vertical velocity on ICON resolution
print('resampling vertical velocity on ICON time resolution ')
values            = np.zeros((len(datetime_ICON), len(height_PBL_class)))
w_DF              = pd.DataFrame(w, index=datetime_PBL, columns=height_PBL_class)
w_resampled       = pd.DataFrame(values, index=datetime_ICON, columns=height_PBL_class)
w_resampled       = getResampledDataPd(w_resampled, w_DF, SelectedIndex_PBL)

# ---- resampling horizontal velocity on ICON resolution
print('resampling horizontal wind speed on ICON time resolution ')
values            = np.zeros((len(datetime_ICON), len(height_PBL_class)))
Hwind_DF          = pd.DataFrame(Hwind, index=datetime_PBL, columns=height_PBL_class)
Hwind_resampled   = pd.DataFrame(values, index=datetime_ICON, columns=height_PBL_class)
Hwind_resampled   = getResampledDataPd(Hwind_resampled, Hwind_DF, SelectedIndex_PBL)

# ---- resampling wind direction on ICON resolution
print('resampling wind direction on ICON time resolution ')
values            = np.zeros((len(datetime_ICON), len(height_PBL_class)))
wDir_DF           = pd.DataFrame(wDir, index=datetime_PBL, columns=height_PBL_class)
wDir_resampled    = pd.DataFrame(values, index=datetime_ICON, columns=height_PBL_class)
wDir_resampled    = getResampledDataPd(wDir_resampled, wDir_DF, SelectedIndex_PBL)

# ---- resampling shear on ICON time resolution
print('calculating shear for observations on ICON time grid')
values_shear      = np.zeros((len(datetime_ICON), len(height_PBL_class)))
shear_DF          = pd.DataFrame(shear, index=datetime_PBL, columns=height_PBL_class)
shear_resampled   = pd.DataFrame(values_shear, index=datetime_ICON, columns=height_PBL_class)
shear_resampled   = getResampledDataPd(shear_resampled, shear_DF, SelectedIndex_PBL)


# ---- resamplig tower observations ( used only for surface values, so no resampling needed in height)
print('resampling tower observations on ICON time resolution (surf meas, no need for height resampling')

Psurf_DF = pd.DataFrame(P_surf, index=datetime_tower)
SelectedIndex_tower = getIndexList(Psurf_DF, ICON_DF.index)
values = np.arange(0, len(ICON_DF.index))
Psurf_resampled = pd.DataFrame(values, index=datetime_ICON)
Psurf_resampled = getResampledDataPd(Psurf_resampled, Psurf_DF, SelectedIndex_tower)

Tsurf_DF = pd.DataFrame(T_surf, index=datetime_tower)
SelectedIndex_tower_T = getIndexList(Tsurf_DF, ICON_DF.index)
values = np.arange(0, len(ICON_DF.index))
Tsurf_resampled = pd.DataFrame(values, index=datetime_ICON)
Tsurf_resampled = getResampledDataPd(Tsurf_resampled, Tsurf_DF, SelectedIndex_tower_T)


RHsurf_DF = pd.DataFrame(RH_surf, index=datetime_tower)
SelectedIndex_tower_RH = getIndexList(RHsurf_DF, ICON_DF.index)
values = np.arange(0, len(ICON_DF.index))
RHsurf_resampled = pd.DataFrame(values, index=datetime_ICON)
RHsurf_resampled = getResampledDataPd(RHsurf_resampled, RHsurf_DF, SelectedIndex_tower_RH)

print('resampled tower observations: Psurf, Tsurf, RHsurf')


#------ resampling MWR observations on ICON time grid
#---------------------------------------------------------------------------------
print('resampling MWR observations on ICON time grid')


# ---- resampling T
TA_DF = pd.DataFrame(TA_MWR, index=datetime_TA)
# ---- removing double values
TA_DF = TA_DF.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')
values_TA = np.zeros((len(datetime_ICON),len(height_TA)))
SelectedIndex_TA = getIndexList(TA_DF, ICON_DF.index)
TA_resampled = pd.DataFrame(values_TA, index=datetime_ICON)
TA_resampled = getResampledDataPd(TA_resampled, TA_DF, SelectedIndex_TA)


# ---- resampling absolute humidity
QA_DF = pd.DataFrame(HUA_MWR, index=datetime_HUA)
# ---- removing double values
QA_DF = QA_DF.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')
values_HUA = np.zeros((len(datetime_ICON),len(height_HUA)))
SelectedIndex_QA = getIndexList(QA_DF, ICON_DF.index)
QA_resampled = pd.DataFrame(values_HUA, index=datetime_ICON)
QA_resampled = getResampledDataPd(QA_resampled, QA_DF, SelectedIndex_QA)


# ---- resampling relative humidity
RH_DF = pd.DataFrame(RH_MWR, index=datetime_RH_MWR)
# ---- removing double values
RH_DF = RH_DF.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')
values_RH_MWR = np.zeros((len(datetime_ICON),len(height_RH_MWR)))
SelectedIndex_RH_MWR = getIndexList(RH_DF, ICON_DF.index)
RH_MWR_resampled = pd.DataFrame(values_RH_MWR, index=datetime_ICON)
RH_MWR_resampled = getResampledDataPd(RH_MWR_resampled, RH_DF, SelectedIndex_RH_MWR)
print('MWR radiometer variables resampled: T, Q, RH')
      
#------ resampling MWR observations on ICON height grid (T, RH, QA)
print('resampling MWR observations on icon vertical height grid')

# ---- defining ICON data as dataframe reference
ICON_DF_T= pd.DataFrame(cloudMask_ICON.transpose(), index=height_ICON, columns=datetime_ICON)

# ---- resampling temperature
T_MWR_transpose_values = TA_resampled.values.transpose()
T_MWR_transpose_DF = pd.DataFrame(T_MWR_transpose_values, index=height_TA, columns=datetime_ICON)

Selectedcol_T_MWR = getIndexList(T_MWR_transpose_DF, ICON_DF_T.index)

values_T_MWR_transpose = np.zeros((len(height_ICON), len(datetime_ICON)))

T_MWR_transpose_less = pd.DataFrame(T_MWR_transpose_values, index=height_TA, columns=datetime_ICON)
T_MWR_ICONres = pd.DataFrame(values_T_MWR_transpose, index=height_ICON, columns=datetime_ICON)
T_MWR_ICONres = getResampledDataPd(T_MWR_ICONres, T_MWR_transpose_less, Selectedcol_T_MWR)

# --- resampling absolute humidity
QA_MWR_transpose_values = QA_resampled.values.transpose()
QA_MWR_transpose_DF = pd.DataFrame(QA_MWR_transpose_values, index=height_HUA, columns=datetime_ICON)

Selectedcol_QA_MWR = getIndexList(QA_MWR_transpose_DF, ICON_DF_T.index)

values_QA_MWR_transpose = np.zeros((len(height_ICON), len(datetime_ICON)))

QA_MWR_transpose_less = pd.DataFrame(QA_MWR_transpose_values, index=height_HUA, columns=datetime_ICON)
QA_MWR_ICONres = pd.DataFrame(values_QA_MWR_transpose, index=height_ICON, columns=datetime_ICON)
QA_MWR_ICONres = getResampledDataPd(QA_MWR_ICONres, QA_MWR_transpose_less, Selectedcol_QA_MWR)


# --- resampling relative humidity
RH_MWR_transpose_values = RH_MWR_resampled.values.transpose()
RH_MWR_transpose_DF = pd.DataFrame(RH_MWR_transpose_values, index=height_RH_MWR, columns=datetime_ICON)

Selectedcol_RH_MWR = getIndexList(RH_MWR_transpose_DF, ICON_DF_T.index)

values_RH_MWR_transpose = np.zeros((len(height_ICON), len(datetime_ICON)))

RH_MWR_transpose_less = pd.DataFrame(RH_MWR_transpose_values, index=height_RH_MWR, columns=datetime_ICON)
RH_MWR_ICONres = pd.DataFrame(values_RH_MWR_transpose, index=height_ICON, columns=datetime_ICON)
RH_MWR_ICONres = getResampledDataPd(RH_MWR_ICONres, RH_MWR_transpose_less, Selectedcol_RH_MWR)

print('resampling in time/height of T, RH and QA done, yeah!')

# In[4]:
#------ regridding Cloudnet COSMO variables on ICON time and height grid
#---------------------------------------------------------------------------------
# ---- defining ICON data as dataframe reference
ICON_DF= pd.DataFrame(cloudMask_ICON, index=datetime_ICON, columns=height_ICON)


# ---- resampling cloudnet classification on ICON resolution
print('resampling CLOUDNET observations on ICON time resolution')
# --- Pressure from COSMO_EU
P_CLOUDNET_DF = pd.DataFrame(P_CLOUDNET, index=datetime_CLOUDNET, columns=model_Height_CLOUDNET)
SelectedIndex_CN = getIndexList(P_CLOUDNET_DF, ICON_DF.index)
P_values_CN = np.zeros((len(datetime_ICON),len(model_Height_CLOUDNET)))
CLOUDNET_resampled_P = pd.DataFrame(P_values_CN, index=datetime_ICON, columns=model_Height_CLOUDNET)
P_CLOUDNET_resampled = getResampledDataPd(CLOUDNET_resampled_P, P_CLOUDNET_DF, SelectedIndex_CN)

# --- Temperature from COSMO-EU
T_CLOUDNET_DF = pd.DataFrame(T_CLOUDNET, index=datetime_CLOUDNET, columns=model_Height_CLOUDNET)
SelectedIndex_CN = getIndexList(T_CLOUDNET_DF, ICON_DF.index)
T_values_CN = np.zeros((len(datetime_ICON),len(model_Height_CLOUDNET)))
CLOUDNET_resampled_T = pd.DataFrame(T_values_CN, index=datetime_ICON, columns=model_Height_CLOUDNET)
T_CLOUDNET_resampled = getResampledDataPd(CLOUDNET_resampled_T, T_CLOUDNET_DF, SelectedIndex_CN)

# --- specific humidity from COSMO-EU
Q_CLOUDNET_DF = pd.DataFrame(Q_CLOUDNET, index=datetime_CLOUDNET, columns=model_Height_CLOUDNET)
SelectedIndex_CN = getIndexList(Q_CLOUDNET_DF, ICON_DF.index)
Q_values_CN = np.zeros((len(datetime_ICON),len(model_Height_CLOUDNET)))
CLOUDNET_resampled_Q = pd.DataFrame(Q_values_CN, index=datetime_ICON, columns=model_Height_CLOUDNET)
Q_CLOUDNET_resampled = getResampledDataPd(CLOUDNET_resampled_Q, Q_CLOUDNET_DF, SelectedIndex_CN)

print('COSMO-EU variables resampled on ICON time resolution')

#------ regridding Cloudnet COSMO variables on ICON height grid
print('resampling COSMO on icon vertical height grid')

# ---- defining ICON data as dataframe reference
ICON_DF_T= pd.DataFrame(cloudMask_ICON.transpose(), index=height_ICON, columns=datetime_ICON)

# ---- resampling pressure
P_transpose_values = P_CLOUDNET_resampled.values.transpose()
P_transpose_DF = pd.DataFrame(P_transpose_values, index=model_Height_CLOUDNET, columns=datetime_ICON)

Selectedcol_P_cloudnet = getIndexList(P_transpose_DF, ICON_DF_T.index)
values_P_transpose = np.zeros((len(height_ICON), len(datetime_ICON)))

P_transpose_less = pd.DataFrame(P_transpose_values, index=model_Height_CLOUDNET, columns=datetime_ICON)
P_Cloudnet_ICONres = pd.DataFrame(values_P_transpose, index=height_ICON, columns=datetime_ICON)
P_Cloudnet_ICONres = getResampledDataPd(P_Cloudnet_ICONres, P_transpose_less, Selectedcol_P_cloudnet)

# ---- resampling temperature
T_transpose_values = T_CLOUDNET_resampled.values.transpose()
T_transpose_DF = pd.DataFrame(T_transpose_values, index=model_Height_CLOUDNET, columns=datetime_ICON)

Selectedcol_T_cloudnet = getIndexList(T_transpose_DF, ICON_DF_T.index)
values_T_transpose = np.zeros((len(height_ICON), len(datetime_ICON)))

T_transpose_less = pd.DataFrame(T_transpose_values, index=model_Height_CLOUDNET, columns=datetime_ICON)
T_Cloudnet_ICONres = pd.DataFrame(values_T_transpose, index=height_ICON, columns=datetime_ICON)
T_Cloudnet_ICONres = getResampledDataPd(T_Cloudnet_ICONres, T_transpose_less, Selectedcol_T_cloudnet)

# ---- resampling specific humidity
Q_transpose_values = Q_CLOUDNET_resampled.values.transpose()
Q_transpose_DF = pd.DataFrame(Q_transpose_values, index=model_Height_CLOUDNET, columns=datetime_ICON)

Selectedcol_Q_cloudnet = getIndexList(Q_transpose_DF, ICON_DF_T.index)
values_Q_transpose = np.zeros((len(height_ICON), len(datetime_ICON)))

Q_transpose_less = pd.DataFrame(Q_transpose_values, index=model_Height_CLOUDNET, columns=datetime_ICON)
Q_Cloudnet_ICONres = pd.DataFrame(values_Q_transpose, index=height_ICON, columns=datetime_ICON)
Q_Cloudnet_ICONres = getResampledDataPd(Q_Cloudnet_ICONres, Q_transpose_less, Selectedcol_Q_cloudnet)


print('resampling on icon vertical height grid of pressure, temperature and specific humidity from COSMO-DE done, yuhu!')

        
# In[4]:


print('resampling obs wind shear on icon vertical height grid')
# ---- defining ICON data as dataframe reference
ICON_DF_T        = pd.DataFrame(cloudMask_ICON.transpose(), index=height_ICON, columns=datetime_ICON)
shearH_values    = shear_resampled.values.transpose()
shear_DF         = pd.DataFrame(shearH_values, index=height_PBL_class, columns=datetime_ICON)
Selectedcol_PBL  = getIndexList(shear_DF, ICON_DF_T.index)
# define empty values for dataframe finer resolved
values_shear_H   = np.zeros((len(height_ICON), len(datetime_ICON)))

# define dataframe coarse resolution
shearH_less      = pd.DataFrame(shearH_values, index=height_PBL_class, columns=datetime_ICON)
# define output dataframe with resolution as icon
shearH_resampled = pd.DataFrame(values_shear_H, index=height_ICON, columns=datetime_ICON)
shearH_resampled = getResampledDataPd(shearH_resampled, shearH_less, Selectedcol_PBL)
print('resampling on icon vertical height grid done, yuhu!')

# ---- calculate variance of vertical velocity for observations using a running mean over 30 min 
#      and resampling it to the standard icon time height grid
# ----------------------------------------------------------------------------------------

print('calculating variance of vertical velocity for observations')
timewindow        = 200#10 #200 for 9 seconds  time window corresponding to 30 min considering that PBL data have time resolution of 3 minutes
from myFunctions import f_calcWvariance
varianceW_obs     = f_calcWvariance(w_resampled.values,datetime_ICON,height_PBL_class,timewindow)



print('resampling obs variance on icon vertical height grid')
varianceW_values  = varianceW_obs.transpose()
varianceW_DF      = pd.DataFrame(varianceW_values, index=height_PBL_class, columns=datetime_ICON)
values_varianceW  = np.zeros((len(height_ICON), len(datetime_ICON)))
varianceW_less    = pd.DataFrame(varianceW_values, index=height_PBL_class, columns=datetime_ICON)
varW_resampled    = pd.DataFrame(values_varianceW, index=height_ICON, columns=datetime_ICON)
varW_resampled    = getResampledDataPd(varW_resampled, varianceW_less, Selectedcol_PBL)
print('resampling on icon vertical height for variance grid done, yuhu!')

print('resampling vertical velocity on icon vertical height grid')
w_values          = w_resampled.transpose()
wwind_DF          = pd.DataFrame(w_values, index=height_PBL_class, columns=datetime_ICON)
values_W          = np.zeros((len(height_ICON), len(datetime_ICON)))
Wwind_less        = pd.DataFrame(w_values, index=height_PBL_class, columns=datetime_ICON)
Wwind_resampledh   = pd.DataFrame(values_W, index=height_ICON, columns=datetime_ICON)
Wwind_resampledh   = getResampledDataPd(Wwind_resampledh, Wwind_less, Selectedcol_PBL)
print('resampling on icon vertical height for variance grid done, yuhu!')


print('resampling wind speed and direction on icon vertical height grid')
Hwind_values = Hwind_resampled.transpose()
Hwind_DF     = pd.DataFrame(Hwind_values, index=height_PBL_class, columns=datetime_ICON)
values_Hwind = np.zeros((len(height_ICON), len(datetime_ICON)))
Hwind_less   = pd.DataFrame(Hwind_values, index=height_PBL_class, columns=datetime_ICON)
Hwind_resampledh   = pd.DataFrame(values_Hwind, index=height_ICON, columns=datetime_ICON)
Hwind_resampledh   = getResampledDataPd(Hwind_resampledh, Hwind_less, Selectedcol_PBL)


wDir_values = wDir_resampled.transpose()
wDir_DF     = pd.DataFrame(wDir_values, index=height_PBL_class, columns=datetime_ICON)
values_wDir = np.zeros((len(height_ICON), len(datetime_ICON)))
wDir_less   = pd.DataFrame(wDir_values, index=height_PBL_class, columns=datetime_ICON)
wDir_resampledh   = pd.DataFrame(values_wDir, index=height_ICON, columns=datetime_ICON)
wDir_resampledh   = getResampledDataPd(wDir_resampledh, wDir_less, Selectedcol_PBL)

print('resampling on icon vertical height for wind speed and direction done, yuhu!')

# In[5]:

# After having resampled in time to bring observations on the same icon resolution and in 
# in height ( the coarser resolution is the one from ICON (height_ICON), 
# while the finer is the one from PBL (height_PBL_class)) we nw proceed to calculate quantities of interest
# 
        
# In[4]:

# ----------------------------------------------------------------------------------------
# calculating mixing ratio (MR) and RH for Cloudnet profiles (COSMO-EU)
# ----------------------------------------------------------------------------------------

Q_calc = Q_Cloudnet_ICONres.values.transpose()
P_calc = P_Cloudnet_ICONres.values.transpose()
T_calc = T_Cloudnet_ICONres.values.transpose()

MR_Cloudnet_ICONres = np.zeros((len(datetime_ICON), len(height_ICON)))
RH_Cloudnet_ICONres = np.zeros((len(datetime_ICON), len(height_ICON)))
T0 = 273.15
for iTime in range(len(datetime_ICON)):
    for iHeight in range(len(height_ICON)-1):
        MR_Cloudnet_ICONres[iTime,iHeight] = (Q_calc[iTime, iHeight])/(1-Q_calc[iTime, iHeight])
        RH_Cloudnet_ICONres[iTime,iHeight] = 0.263*P_calc[iTime, iHeight] * \
        Q_calc[iTime, iHeight] * (np.exp( 17.67 * (T_calc[iTime, iHeight]-T0) / (T_calc[iTime, iHeight] - 29.65)))**(-1)

        
# In[4]:


# ----------------------------------------------------------------------------------------
# calculating cloud mask, cloud base and cloud top for the observations, ICON-LEM, and ICON-INSCAPE
# ----------------------------------------------------------------------------------------

# --- observations 
Cat               = f_resamplingMatrix(datetime_CLOUDNET, height_CLOUDNET, cloudnet, datetime_ICON, height_ICON, P)
Cloudnet          = Cat.transpose()
CloudMask         = f_calculateCloudMaskCloudnet(time_ICON, height_ICON, Cloudnet.astype(int))
from myFunctions import f_calcCloudBaseTop
result            = f_calcCloudBaseTop(CloudMask, len(time_ICON), len(height_ICON), height_ICON)
np.shape(result)

CBMatrix          = result[0]
CTMatrix          = result[1]
CT_array_OBS      = np.empty(len(datetime_ICON))
CB_array_OBS      = np.empty(len(datetime_ICON))
CT_array_OBS[:]   = np.nan
CB_array_OBS[:]   = np.nan

for indT in range(len(datetime_ICON)):#
    if (~np.isnan(CBMatrix[indT,0]) == True) and (~np.isnan(CTMatrix[indT,0])== True):
        
        indCB = f_closest(height_ICON, CBMatrix[indT,0])
        indCT = f_closest(height_ICON, CTMatrix[indT,0])
        
        if (indCB == 0) or (indCT == 0):
            CT_array_OBS[indT] = np.nan
            CB_array_OBS[indT] = np.nan
        else:
            CT_array_OBS[indT] = height_ICON[indCT]                                 # saving cloud top height
            CB_array_OBS[indT] = height_ICON[indCB]                                 # saving cloud base height
            
print('cloud base and cloud top for observations calculated ')


# --- ICON-LEM

result                 = f_calcCloudBaseTop(cloudMask_ICON, len(time_ICON), len(height_ICON), height_ICON)
np.shape(result)

CBMatrix_ICON          = result[0]
CTMatrix_ICON          = result[1]
CT_array_ICON          = np.empty(len(datetime_ICON))
CB_array_ICON          = np.empty(len(datetime_ICON))
CT_array_ICON[:]       = np.nan
CB_array_ICON[:]       = np.nan

for indT in range(len(datetime_ICON)):#
    if (~np.isnan(CBMatrix_ICON[indT,0]) == True) and (~np.isnan(CTMatrix_ICON[indT,0])== True):
        
        indCB          = f_closest(height_ICON, CBMatrix_ICON[indT,0])
        indCT          = f_closest(height_ICON, CTMatrix_ICON[indT,0])
        
        if (indCB == 0) or (indCT == 0):
            CT_array_ICON[indT] = np.nan
            CB_array_ICON[indT] = np.nan
        else:
            CT_array_ICON[indT] = height_ICON[indCT]                                 # saving cloud top height
            CB_array_ICON[indT] = height_ICON[indCB]                                 # saving cloud base height
            
print('cloud base and cloud top for ICON-LEM calculated ')

# --- ICON-INSCAPE
result                 = f_calcCloudBaseTop(cloudMask_ICON_vera, len(time_ICON_vera), len(height_ICON_vera), height_ICON_vera)
np.shape(result)    

CBMatrix_ICON_INSCAPE          = result[0]
CTMatrix_ICON_INSCAPE          = result[1]
CT_array_ICON_INSCAPE          = np.empty(len(datetime_ICON_vera))
CB_array_ICON_INSCAPE          = np.empty(len(datetime_ICON_vera))
CT_array_ICON_INSCAPE[:]       = np.nan
CB_array_ICON_INSCAPE[:]       = np.nan

for indT in range(len(datetime_ICON_vera)):#
    if (~np.isnan(CBMatrix_ICON_INSCAPE[indT,0]) == True) and (~np.isnan(CTMatrix_ICON_INSCAPE[indT,0])== True):
        
        indCB          = f_closest(height_ICON_vera, CBMatrix_ICON_INSCAPE[indT,0])
        indCT          = f_closest(height_ICON_vera, CTMatrix_ICON_INSCAPE[indT,0])
        
        if (indCB == 0) or (indCT == 0):
            CT_array_ICON_INSCAPE[indT] = np.nan
            CB_array_ICON_INSCAPE[indT] = np.nan
        else:
            CT_array_ICON_INSCAPE[indT] = height_ICON_vera[indCT]                                 # saving cloud top height
            CB_array_ICON_INSCAPE[indT] = height_ICON_vera[indCB]                                 # saving cloud base height
            
print('cloud base and cloud top for ICON-INSCAPE calculated ')

        
# In[4]:

# ---------------------------------------------------------------------------------------
# --- deriving virtual temperature and Pressure for radiometer observations
# ----------------------------------------------------------------------------------------

print('deriving virtual temperature and Pressure for radiometer observations')
TV_MWR_ICONres = np.zeros((len(datetime_ICON), len(height_ICON)))
P_MWR_ICONres = np.zeros((len(datetime_ICON), len(height_ICON))) 
T_calc = T_MWR_ICONres.values.transpose()     # air temperature in K
Q_calc = QA_MWR_ICONres.values.transpose()    # absolute humidity in Kg/m^3


# calculation of virtual temperature
for indH in range(len(height_ICON)-1, 0, -1):
    TV_MWR_ICONres[:,indH] = T_calc[:,indH]*(1+0.608 * Q_calc[:,indH])


g = 9.81
Rl = 287.

P_MWR_ICONres[:,len(height_ICON)-1] = Psurf_resampled.values[:,0]      # pressure in Pa

for indH in range(len(height_ICON)-2, 0, -1):  
    P_MWR_ICONres[:,indH] = P_MWR_ICONres[:,indH+1] * np.exp( \
        -g*(height_ICON[indH-1]-height_ICON[indH])  /   (Rl *(TV_MWR_ICONres[:,indH]+TV_MWR_ICONres[:,indH-1])/2.))
    

        
# In[4]:
# ----------------------------------------------------------------------------------------
# --- Calculation of wind shear for ICON output as done for PBL  ( running mean over 30 min of sqrt(Delta U^2 + delta V^2))/delta H 
# where variations are calculated over 5 range gates 
# ----------------------------------------------------------------------------------------

print('Calculating wind shear for ICON run')

# --- calculating shear of horizontal wind 
u_rm                               = np.zeros((len(datetime_ICON), len(height_ICON)))
v_rm                               = np.zeros((len(datetime_ICON), len(height_ICON)))

# --- defining running mean values of zonal and meridional wind
for indH in range(0,len(height_ICON)):
    u_rm[:,indH]                   = pd.rolling_mean(zonalWind[:,indH], window=200) 
    v_rm[:,indH]                   = pd.rolling_mean(merWind[:,indH], window=200) 


shear_ICON                         = np.zeros((len(datetime_ICON), len(height_ICON)))
for indT in range(0,len(datetime_ICON)):
    for indH in range(0,len(height_ICON)):
        if (indH < 2.) or (indH > len(height_ICON)-3):
            shear_ICON[indT, indH] = 0.
        else:
            deltaV                 = (np.absolute(v_rm[indT, indH+2] - v_rm[indT, indH-2]))**2
            deltaU                 = (np.absolute(u_rm[indT, indH+2] - u_rm[indT, indH-2]))**2
            deltaH                 = np.absolute(height_ICON[indH+2] - height_ICON[indH-2])
            shear_ICON[indT, indH] = (np.sqrt(deltaU + deltaV))/deltaH


print('wind shear for ICON run done, oleoho')


# In[5]:

# In[14]:
# ----------------------------------------------------------------------------------------
# calculation of wind direction and intensity for model output
# ----------------------------------------------------------------------------------------

windData_ICON             = f_calcWindSpeed_Dir(datetime_ICON, height_ICON, merWind, zonalWind)
windData_ICON_INSCAPE     = f_calcWindSpeed_Dir(datetime_ICON_vera, height_ICON_vera, merWind_vera, zonalWind_vera)
print('wind speed and direction calculated for ICON-LEM AND ICON_INSCAPE')



# In[14]:
# ---------------------------------------------------------------------------
# ---- derivation of cloud condensation level and temperature for ICON, ICON_INSCAPE, COSMO_EU
# ---------------------------------------------------------------------------

# --- calculation of Tccl for ICON, ICON-INSCAPE, COSMO_EU
from myFunctions import f_CCL
result_ICON = f_CCL(T_ICON, P_ICON, RH_ICON, height_ICON, datetime_ICON, Hsurf)
result_ICON_INSCAPE = f_CCL(T_ICON_vera, P_ICON_vera, RH_ICON_vera, height_ICON_vera, datetime_ICON_vera, Hsurf)
result_COSMO = f_CCL(T_Cloudnet_ICONres.values.transpose(), P_Cloudnet_ICONres.values.transpose(), \
                     RH_Cloudnet_ICONres, height_ICON_vera, datetime_ICON_vera, Hsurf)
print(np.shape(T_ICON))
print(np.shape(P_ICON))
print(np.shape(RH_ICON))
print(np.shape(T_MWR_ICONres.values))
print(np.shape(P_MWR_ICONres))
print(np.shape(RH_MWR_ICONres.values))

result_MWR = f_CCL(T_MWR_ICONres.values.transpose(), P_MWR_ICONres, RH_MWR_ICONres.values.transpose(), height_ICON, datetime_ICON, Hsurf)


# --- reading variables to be plotted
Td_ICON = result_ICON['Td'][:]
T_ccl_ICON = result_ICON['T_ccl'][:]
z_ccl_ICON = result_ICON['z_ccl'][:]

Td_MWR = result_MWR['Td'][:]
T_ccl_MWR = result_MWR['T_ccl'][:]
z_ccl_MWR = result_MWR['z_ccl'][:]
T_surface_MWR = T_MWR_ICONres.values.transpose()[:,len(height_ICON)-1]

Td_ICON_INSCAPE = result_ICON_INSCAPE['Td'][:]
T_ccl_ICON_INSCAPE = result_ICON_INSCAPE['T_ccl'][:]
z_ccl_ICON_INSCAPE = result_ICON_INSCAPE['z_ccl'][:]


Td_COSMO = result_COSMO['Td'][:]
T_ccl_COSMO = result_COSMO['T_ccl'][:]
z_ccl_COSMO = result_COSMO['z_ccl'][:]

T_surf_tower = Tsurf_resampled.values


# In[14]:

# --------------------------------------------------------------------
# ----- plot figure with TCCL and T surface temperature time series
# --------------------------------------------------------------------
import matplotlib as mpl
fig, ax = plt.subplots(figsize=(10,5))

label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams['savefig.dpi'] = 100
plt.gcf().subplots_adjust(bottom=0.15)
fig.tight_layout()

ax = plt.subplot(111)  
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left()  
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

plt.title(' T_CCL - T_surface time series - 20130502', fontsize=18)
plt.ylabel('T_ccl - T_surf [K]', fontsize=16)
#plt.ylim(282.5, 300.)
plt.xlabel('time [hh:mm]', fontsize=16)
plt.xlim(datetime.datetime(2013,5,2,6,0,0),datetime.datetime(2013,5,2,20,0,0))
plt.plot(datetime_ICON, T_ccl_ICON-T_ICON[:,149], label='ICON-LEM', color='red')
plt.plot(datetime_ICON_vera, T_ccl_ICON_INSCAPE-T_ICON_vera[:,149], label='ICON_INSCAPE', color='blue')
plt.plot(datetime_ICON_vera, T_ccl_COSMO-T_Cloudnet_ICONres.values.transpose()[:-1,149], label='COSMO-EU', color='green')
#plt.plot(datetime_ICON, T_ccl_MWR, label='T ccl (obs-MWR radiometer)', color='black')
plt.plot(time_ccl_rs, T_ccl_radiosondes-T_surf_radiosondes, 'o', markersize=12, label='radiosondes', color='black')
T_surf_tower_interp = []
for ind in range(len(time_ccl_rs)):
    indTimeTower = f_closest(datetime_ICON, pd.Series(time_ccl_rs)[ind])
    T_surf_tower_interp.append(T_surf_tower[indTimeTower,0])
    
plt.plot(time_ccl_rs, T_ccl_radiosondes-T_surf_tower_interp, 'v', markersize=12, label='radiosondes/tower', color='black')

#plt.plot(datetime_ICON, T_ICON[:,149], label='T surface (ICON)', color='red', linestyle=':')
#plt.plot(datetime_ICON_vera, T_ICON_vera[:,149], label='T surface (ICON_INSCAPE)', color='blue', linestyle=':')
#plt.plot(datetime_ICON, T_Cloudnet_ICONres.values.transpose()[:,149], label='T surface (COSMO-EU)', color='green', linestyle=':')
#plt.plot(datetime_ICON, T_surface_MWR, label='T surface (obs - MWR)', color='black', linestyle=':')
#plt.plot(datetime_ICON, T_surf_tower, label='T surface (obs - tower)', color='black', linestyle='-')
plt.axhline(y=0., color='black', linestyle='--')
plt.plot()
plt.legend(loc='upper left', fontsize=12)
plt.tight_layout()
plt.savefig(pathFig+'TCCL_Tsurf_comparison_20130502.pdf', format='pdf')


# --------------------------------------------------------------------
# ----- plot figure with TCCL and T surface temperature time series
# --------------------------------------------------------------------
import matplotlib as mpl
fig, ax = plt.subplots(figsize=(12,6))

label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams['savefig.dpi'] = 100
plt.gcf().subplots_adjust(bottom=0.15)
fig.tight_layout()

ax = plt.subplot(111)  
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left()  
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

plt.title('Time series of surface and CCL temperatures', fontsize=18)
plt.ylabel('T [K]', fontsize=16)
plt.ylim(282.5, 300.)
plt.xlabel('time [hh:mm]', fontsize=16)
plt.xlim(datetime.datetime(2013,5,2,6,0,0),datetime.datetime(2013,5,2,20,0,0))
plt.plot(datetime_ICON, T_ccl_ICON, label='T ccl (ICON)', color='red')
plt.plot(datetime_ICON_vera, T_ccl_ICON_INSCAPE, label='T ccl (ICON_INSCAPE)', color='blue')
plt.plot(datetime_ICON_vera, T_ccl_COSMO, label='T ccl (COSMO-EU)', color='green')
#plt.plot(datetime_ICON, T_ccl_MWR, label='T ccl (obs-MWR radiometer)', color='black')

plt.plot(time_ccl_rs, T_ccl_radiosondes, 'o', markersize=12, label='T ccl (radiosondes)', color='black')
plt.plot(datetime_ICON, T_ICON[:,149], label='T surface (ICON)', color='red', linestyle=':')
plt.plot(datetime_ICON_vera, T_ICON_vera[:,149], label='T surface (ICON_INSCAPE)', color='blue', linestyle=':')
plt.plot(datetime_ICON, T_Cloudnet_ICONres.values.transpose()[:,149], label='T surface (COSMO-EU)', color='green', linestyle=':')
#plt.plot(datetime_ICON, T_surface_MWR, label='T surface (obs - MWR)', color='black', linestyle=':')
plt.plot(time_ccl_rs, T_surf_radiosondes, label='T surface (radiosondes)', color='black', linestyle='-')



plt.legend(loc='lower right', fontsize=14)
plt.tight_layout()
plt.savefig(pathFig+'TCCL_Tsurf_comparison_20130502.pdf', format='pdf')

strasuka

# In[14]:
# ----------------------------------------------------------------------------------------
# --------------- plot time height of vertical wind speed from ICON and ICON_INSCAPE and OBS ------------------------
# ----------------------------------------------------------------------------------------

# general settings
matplotlib.rc('xtick', labelsize=15)                        # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=15)                        # sets dimension of ticks in the plots
TimeStart = datetime.datetime(2013, 5, 2, 8, 0, 0)
TimeEnd   = datetime.datetime(2013, 5, 2, 16, 0, 0)



# ---------------   plot of vertical wind speed from ICON
Dict = {}
Dict = {'xvar':datetime_ICON, 'yvar':height_ICON, 'colorVar':w_ICON.transpose(), \
            'xmin':TimeStart, 'xmax':TimeEnd, 'ymin':107., 'ymax':5000.,\
            'colorVarMin':-2., 'colorVarMax':2.,'xlabel':"time [hh:mm]", \
            'ylabel':"height [m]",'ColorVarLabel':"wind speed [m/s]", \
            'title':'vertical wind speed (ICON-LEM)', 'xFont':16, 'yFont':16, \
            'ColorBarFont':16, 'titleFont':16, 'colorPalette':'PiYG', \
            'outFile':pathFig+'TimeHeightColorMaps_verticalWindSpeed_ICON.png'}

fig, ax = plt.subplots(figsize=(14,6))
# formatting x axis for datetime plotting
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
# calling pcolormesh to plot data
cax = ax.pcolormesh(Dict['xvar'], Dict['yvar'], Dict['colorVar'], vmin=Dict['colorVarMin'],\
                    vmax=Dict['colorVarMax'], cmap = matplotlib.cm.get_cmap(Dict['colorPalette']))
plt.plot(datetime_ICON, cloudBase_ICON, color='black', marker="v", linestyle='None', label='cloud base')
plt.plot(datetime_ICON, CT_array_ICON, color='black', marker="1", linestyle='None', label='cloud top')

#plt.plot(datetime_ICON_vera, cloudBase_ICON_vera, color='black', linestyle=':')
#plt.plot(datetime_ICON, CB_array_OBS, color='black', linestyle='-')

#plt.plot(datetime_ICON, CB_array_OBS, color='black', linestyle='-.')
ax.set_xlim(Dict['xmin'],Dict['xmax'])                                               # limits of the y-axes
ax.set_ylim(Dict['ymin'],Dict['ymax'])                                               # limits of the y-axes
ax.set_title(Dict['title'], fontsize=Dict['titleFont'])
ax.set_xlabel(Dict['xlabel'], fontsize=Dict['xFont'])
ax.set_ylabel(Dict['ylabel'], fontsize=Dict['yFont'])
plt.legend(loc='upper left', fontsize=12)
cbar = fig.colorbar(cax, orientation='vertical')
cbar.set_label(label=Dict['ColorVarLabel'],size=Dict['ColorBarFont'])
cbar.ax.tick_params(labelsize=16)
cbar.aspect=80
plt.savefig(Dict['outFile'], format='png')

# ----------------------------------------------------------------------------------------
# ---------------   plot of vertical wind speed from OBSERVATIONS
# ----------------------------------------------------------------------------------------

#Dict      = {  'ylabel':"height [m]", 'ColorVarLabel':"wind speed [m/s]", 'title':'vertical wind speed (OBS)', 'xFont':16, 'yFont':16, 'ColorBarFont':16, 'titleFont':16, 'colorPalette':'PiYG', 'outFile':pathFig+'TimeHeightColorMaps_verticalWindSpeed_OBS.png'}
Dict = {}
Dict = {'xvar':datetime_ICON, 'yvar':height_ICON, 'colorVar':Wwind_resampledh.values, \
            'xmin':TimeStart, 'xmax':TimeEnd, 'ymin':107., 'ymax':5000.,\
            'colorVarMin':-2., 'colorVarMax':2.,'xlabel':"time [hh:mm]", \
            'ylabel':"height [m]",'ColorVarLabel':"wind speed [m/s]", \
            'title':'vertical wind speed (OBS)', 'xFont':16, 'yFont':16, \
            'ColorBarFont':16, 'titleFont':16, 'colorPalette':'PiYG', \
            'outFile':pathFig+'TimeHeightColorMaps_verticalWindSpeed_OBS.png'}
fig, ax   = plt.subplots(figsize=(14,6))
# formatting x axis for datetime plotting
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
# calling pcolormesh to plot data
cax       = ax.pcolormesh(Dict['xvar'], Dict['yvar'], Dict['colorVar'], vmin=Dict['colorVarMin'], 
                          vmax=Dict['colorVarMax'], cmap = matplotlib.cm.get_cmap(Dict['colorPalette']))
#plt.plot(datetime_ICON, cloudBase_ICON, color='black', linestyle='-')
#plt.plot(datetime_ICON_vera, cloudBase_ICON_vera, color='black', linestyle=':')
plt.plot(datetime_ICON, CB_array_OBS, color='black', marker="v", linestyle='None', label='cloud base')
plt.plot(datetime_ICON, CT_array_OBS, color='black', marker="1", linestyle='None', label='cloud top')

#plt.plot(datetime_ICON, CB_array_OBS, color='black', linestyle='-.')
ax.set_xlim(Dict['xmin'],Dict['xmax'])                                               # limits of the y-axes
ax.set_ylim(Dict['ymin'],Dict['ymax'])                                               # limits of the y-axes
ax.set_title(Dict['title'], fontsize=Dict['titleFont'])
ax.set_xlabel(Dict['xlabel'], fontsize=Dict['xFont'])
ax.set_ylabel(Dict['ylabel'], fontsize=Dict['yFont'])
plt.legend(loc='upper left', fontsize=12)

cbar = fig.colorbar(cax, orientation='vertical')
cbar.set_label(label=Dict['ColorVarLabel'],size=Dict['ColorBarFont'])
cbar.ax.tick_params(labelsize=16)
cbar.aspect=80
plt.savefig(Dict['outFile'], format='png')

# ----------------------------------------------------------------------------------------
# ----------  plot of vertical wind speed for  ICON_INSCAPE
# ----------------------------------------------------------------------------------------

Dict = {}
Dict = {'xvar':datetime_ICON_vera, 'yvar':height_ICON, 'colorVar':w_ICON_vera.transpose(), \
            'xmin':TimeStart, 'xmax':TimeEnd, 'ymin':107., 'ymax':5000.,\
            'colorVarMin':-2., 'colorVarMax':2.,'xlabel':"time [hh:mm]", \
            'ylabel':"height [m]",'ColorVarLabel':"wind speed [m/s]", \
            'title':'vertical wind speed (ICON-ISCAPE)', 'xFont':16, 'yFont':16, \
            'ColorBarFont':16, 'titleFont':16, 'colorPalette':'PiYG', \
            'outFile':pathFig+'TimeHeightColorMaps_verticalWindSpeed_ICON_INSCAPE.png'}
#Dict = {'xvar':datetime_ICON_vera, 'yvar':height_ICON, 'colorVar':w_ICON_vera.transpose(), 
#            'xmin':TimeStart, 'xmax':TimeEnd,             'ymin':107., 'ymax':4000., 'colorVarMin':-2., 'colorVarMax':2.,            'xlabel':"time [hh:mm]", 'ylabel':"height [m]",'ColorVarLabel':"wind speed [m/s]",             'title':'vertical wind speed (ICON-INSCAPE)', 'xFont':16, 'yFont':16, 'ColorBarFont':16, 'titleFont':16,             'colorPalette':'PiYG', 'outFile':pathFig+'TimeHeightColorMaps_verticalWindSpeed_ICON_INSCAPE.png'}
matplotlib.rc('xtick', labelsize=15)                        # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=15)                        # sets dimension of ticks in the plots
fig, ax = plt.subplots(figsize=(14,6))
# formatting x axis for datetime plotting
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
# calling pcolormesh to plot data
cax = ax.pcolormesh(Dict['xvar'], Dict['yvar'], Dict['colorVar'], vmin=Dict['colorVarMin'], \
                    vmax=Dict['colorVarMax'], cmap = matplotlib.cm.get_cmap(Dict['colorPalette']))
#plt.plot(datetime_ICON, cloudBase_ICON, color='black', linestyle='-')
plt.plot(datetime_ICON_vera, cloudBase_ICON_vera, color='black', marker="v", linestyle='None', label='cloud base')
plt.plot(datetime_ICON_vera, CT_array_ICON_INSCAPE, color='black', marker="1", linestyle='None', label='cloud top')
ax.set_xlim(Dict['xmin'],Dict['xmax'])                                               # limits of the y-axes
ax.set_ylim(Dict['ymin'],Dict['ymax'])                                               # limits of the y-axes
ax.set_title(Dict['title'], fontsize=Dict['titleFont'])
ax.set_xlabel(Dict['xlabel'], fontsize=Dict['xFont'])
ax.set_ylabel(Dict['ylabel'], fontsize=Dict['yFont'])
plt.legend(loc='upper left', fontsize=12)

cbar = fig.colorbar(cax, orientation='vertical')
cbar.set_label(label=Dict['ColorVarLabel'],size=Dict['ColorBarFont'])
cbar.ax.tick_params(labelsize=16)
cbar.aspect=80
plt.savefig(Dict['outFile'], format='png')



# In[14]:

# ---------------------------------------------------------------------------
# ---- derivation of lifting condensation level for ICON, ICON_INSCAPE, COSMO_EU
# ---------------------------------------------------------------------------
from myFunctions import lcl

# calculation for observations using P and T from tower measurements, and RH at 
#the ground from microwave radiometer and for COSMO model

# requested variables for observations
LCL_obs = []
P_lcl_obs = Psurf_resampled.values[:,0]
T_lcl_obs = Tsurf_resampled.values[:,0]
RH_lcl_obs = RH_MWR_ICONres.values.transpose()[:,149]


# requested variables for COSMO model
LCL_COSMO = []
P_lcl_COSMO = P_Cloudnet_ICONres.values.transpose()[:,149]
T_lcl_COSMO = T_Cloudnet_ICONres.values.transpose()[:,149]
RH_lcl_COSMO = RH_Cloudnet_ICONres[:,149]


for iTime in range(len(datetime_ICON)):
    LCL_obs.append(lcl(P_lcl_obs[iTime],T_lcl_obs[iTime],RH_lcl_obs[iTime]/100.))
    LCL_COSMO.append(lcl(P_lcl_COSMO[iTime],T_lcl_COSMO[iTime],RH_lcl_COSMO[iTime]/100.))
    

# --------------------------------------------------------------------
# ----- plot figure with TCCL and T surface temperature time series
# --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14,6))

label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams['savefig.dpi'] = 100
plt.gcf().subplots_adjust(bottom=0.15)
fig.tight_layout()

ax = plt.subplot(111)  
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left()  
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xlim(datetime.datetime(2013,5,2,6,0,0),datetime.datetime(2013,5,2,20,0,0))

#plt.plot(datetime_ICON, LCL_obs, color='black', label='LCL observations tower + MWR ')
plt.plot(datetime_ICON, LCL_ICON, color='red', label='LCL ICON')
plt.plot(datetime_ICON_vera, LCL_ICON_vera, color='blue', label='LCL ICON-INSCAPE')
plt.plot(time_ccl_rs, z_lcl_radiosondes, 'o', markersize=12, label='LCL (radiosondes)', color='black')
plt.ylabel('LCL height [m]', fontsize=16)
plt.xlabel('time [hh:mm]', fontsize=16)
z_lcl_radiosondes
plt.title('Lifting condensation level (LCL): comparison ICON/obs ', fontsize=16)
plt.legend(loc='upper left', fontsize=16)
plt.tight_layout()
plt.savefig(pathFig+'LCL_timeserie_obs_ICON_20130502.pdf', format='pdf')


# In[14]:


# ----------------------------------------------------------------------------------------
# ----------- Calculation and plot of mean cloud base in model and observations for every hour of the day ----------------
# ----------------------------------------------------------------------------------------

# calculating mean cloud base height for icon
CB_ICON_DF               = pd.DataFrame(cloudBase_ICON, index=datetime_ICON)
CB_ICON_VERA_DF          = pd.DataFrame(cloudBase_ICON_vera, index=datetime_ICON_vera)
mean_CB_arr_ICON         = []
mean_CB_arr_ICON_VERA    = []
mean_CB_arr_OBS          = []
CB_array_OBS_DF          = pd.DataFrame(CB_array_OBS, index=datetime_ICON)
minGlobalCB_arr          = []
maxGlobalCB_arr          = []
datetimeHourArr          = []
for indHour in range(0,23):
    HourSup              = datetime.datetime(2013, 5, 2, indHour+1, 0, 0)
    HourInf              = datetime.datetime(2013, 5, 2, indHour, 0, 0)
    CB_ICON_DF_sliced_t  = CB_ICON_DF.loc[(CB_ICON_DF.index < HourSup) * (CB_ICON_DF.index > HourInf)]
    CB_ICON_VERA_DF_sliced_t  = CB_ICON_VERA_DF.loc[(CB_ICON_VERA_DF.index < HourSup) * (CB_ICON_VERA_DF.index > HourInf)]
    CB_array_OBS_DF_sliced_t = CB_array_OBS_DF.loc[(CB_array_OBS_DF.index < HourSup) * (CB_array_OBS_DF.index > HourInf)]
    datetimeHourArr.append(HourInf)

    if ~np.isfinite(np.nanmean(CB_ICON_DF_sliced_t)):
        mean_CB_arr_ICON.append(np.nan)
    else:
        mean_CB_arr_ICON.append(height_ICON[f_closest(height_ICON, np.nanmean(CB_ICON_DF_sliced_t))] )
        
        
    if ~np.isfinite(np.nanmean(CB_ICON_VERA_DF_sliced_t)):
        mean_CB_arr_ICON_VERA.append(np.nan)
    else:
        mean_CB_arr_ICON_VERA.append(height_ICON[f_closest(height_ICON, np.nanmean(CB_ICON_VERA_DF_sliced_t))])
  

    if ~np.isfinite(np.nanmean(CB_array_OBS_DF_sliced_t)):    
        mean_CB_arr_OBS.append(np.nan)
    else:
        mean_CB_arr_OBS.append(height_ICON[f_closest(height_ICON, np.nanmean(CB_array_OBS_DF_sliced_t))])

    # calculating min global CB 
    minGlobalCB_arr.append(np.nanmin([mean_CB_arr_OBS[indHour], mean_CB_arr_ICON_VERA[indHour],mean_CB_arr_ICON[indHour]]))
    maxGlobalCB_arr.append(np.nanmax([mean_CB_arr_OBS[indHour], mean_CB_arr_ICON_VERA[indHour],mean_CB_arr_ICON[indHour]]))


# ----------------------------------------------------------------------------------------
# ------------------- plot mean cloud base heights for model and obs
# ----------------------------------------------------------------------------------------

fig, ax   = plt.subplots(figsize=(14,6))
ax        = plt.subplot(111)
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left() 
ax.tick_params(labelsize=16)
plt.plot(mean_CB_arr_ICON, color='blue', label='ICON-LEM')
plt.plot(mean_CB_arr_ICON_VERA, color='red', label='ICON-INSCAPE')
plt.plot(mean_CB_arr_OBS, color='black', label='OBS')
plt.plot(minGlobalCB_arr, color='black', label='min cloud base', linestyle=':')
plt.plot(maxGlobalCB_arr, color='black', label='max cloud base', linestyle='--')
plt.ylim(0, 5000.)
plt.legend(loc='upper left', fontsize=14)#range=[np.nanmin(LWP_obs_PBL),np.nanmax(LWP_obs_PBL)]
plt.title('hourly mean cloud base', fontsize=16)
plt.xlabel('hours ', fontsize=16)
plt.ylabel('height', fontsize=16)
plt.savefig(pathFig+'meanCloudBaseHeight_20130502.png', format='png')   



# In[14]:

# ----------------------------------------------------------------------------------------
# ----------- Histograms of vertical wind speed and wind direction below cloud base for observations, ICON-LEM and ICON_INSCAPE ----------------------------------------------
# ----------------------------------------------------------------------------------------

# ---- observations -----
WwindObs = Wwind_resampledh.values
HwindObs = Hwind_resampledh.values
array_output_OBS  = f_pdfsBelowCloudBase(WwindObs.transpose(), HwindObs.transpose(), datetime_ICON, height_ICON, mean_CB_arr_OBS, CB_array_OBS)

nbins     = 20
fig, ax   = plt.subplots(figsize=(14,6))
ax        = plt.subplot(111)
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left() 
ax.tick_params(labelsize=16)
plt.hist(array_output_OBS[0], bins=nbins, range=[-3.,3.], normed=True, color='red', cumulative=False, alpha=0.5, label='obs cloud')
plt.hist(array_output_OBS[1], bins=nbins, range=[-3.,3.], normed=True, color='blue', cumulative=False, alpha=0.5, label='obs no cloud')
plt.legend(loc='upper right', fontsize=12)#range=[np.nanmin(LWP_obs_PBL),np.nanmax(LWP_obs_PBL)]
plt.title('vertical wind below cloud base - PDFs observations', fontsize=16)
plt.xlabel('vertical wind below cloud base [m/s]', fontsize=16)
plt.ylabel('Occurrences', fontsize=16)
plt.savefig(pathFig+'histogram_WwindSpeedbelowCloudBase_OBS_20130502.png', format='png')  


nbins     = 20
fig, ax   = plt.subplots(figsize=(14,6))
ax        = plt.subplot(111)
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left() 
ax.tick_params(labelsize=16)
plt.hist(array_output_OBS[2], bins=nbins, range=[0.,10.], normed=True, color='red', cumulative=False, alpha=0.5, label='obs cloud')
plt.hist(array_output_OBS[3], bins=nbins, range=[0.,10.], normed=True, color='blue', cumulative=False, alpha=0.5, label='obs no cloud')
plt.legend(loc='upper right', fontsize=12)#range=[np.nanmin(LWP_obs_PBL),np.nanmax(LWP_obs_PBL)]
plt.title('horizontal wind below cloud base - PDFs observations', fontsize=16)
plt.xlabel('horizontal wind below cloud base [m/s]', fontsize=16)
plt.ylabel('Occurrences', fontsize=16)
plt.savefig(pathFig+'histogram_HwindSpeedbelowCloudBase_OBS_20130502.png', format='png')  


# ---- ICON-LEM -----
array_output_ICON = f_pdfsBelowCloudBase(w_ICON, windData_ICON[2], datetime_ICON, height_ICON, mean_CB_arr_ICON, cloudBase_ICON)

nbins     = 20
fig, ax   = plt.subplots(figsize=(14,6))
ax        = plt.subplot(111)
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
alphaval = 0.5
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left() 
ax.tick_params(labelsize=16)
plt.hist(array_output_ICON[0], bins=nbins, range=[-3., 3], normed=True, color='red', cumulative=False, alpha=alphaval, label='ICON-LEM cloud')
plt.hist(array_output_ICON[1], bins=nbins, range=[-3., 3], normed=True, color='blue', cumulative=False, alpha=alphaval, label='ICON-LEM no cloud')
plt.legend(loc='upper right', fontsize=12)#range=[np.nanmin(LWP_obs_PBL),np.nanmax(LWP_obs_PBL)]
plt.title('vertical wind below cloud base - PDFs', fontsize=16)
plt.xlabel('vertical wind below cloud base [m/s]', fontsize=16)
plt.ylabel('Occurrences', fontsize=16)
plt.savefig(pathFig+'histogram_WwindSpeedbelowCloudBase_ICON_20130502.png', format='png')  


nbins     = 20
fig, ax   = plt.subplots(figsize=(14,6))
ax        = plt.subplot(111)
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left() 
ax.tick_params(labelsize=16)
plt.hist(array_output_ICON[2], bins=nbins, range=[0.,10.], normed=True, color='red', cumulative=False, alpha=alphaval, label='ICON-LEM cloud')
plt.hist(array_output_ICON[3], bins=nbins, range=[0.,10.], normed=True, color='blue', cumulative=False, alpha=alphaval, label='ICON-LEM no cloud')
plt.legend(loc='upper right', fontsize=12)#range=[np.nanmin(LWP_obs_PBL),np.nanmax(LWP_obs_PBL)]
plt.title('horizontal wind below cloud base - PDFs ICON-LEM', fontsize=16)
plt.xlabel('horizontal wind below cloud base [m/s]', fontsize=16)
plt.ylabel('Occurrences', fontsize=16)
plt.savefig(pathFig+'histogram_HwindSpeedbelowCloudBase_ICON_20130502.png', format='png')  


# ---- ICON-INSCAPE -----
# interpolating Wwind on the 150 points grid
flippato = np.flip(w_ICON_vera[:],1) 
wWind_ICON_INSCAPE = 0.5*(flippato[:,:-1]+flippato[:,1:])

array_output_ICON_INSCAPE = f_pdfsBelowCloudBase(wWind_ICON_INSCAPE, windData_ICON_INSCAPE[2], datetime_ICON_vera, height_ICON_vera, mean_CB_arr_ICON_VERA, cloudBase_ICON_vera)
#w_ICON, Hwind, datetime_ICON, height_ICON, mean_CB_arr_OBS, CB_array_OBS
nbins     = 20
fig, ax   = plt.subplots(figsize=(14,6))
ax        = plt.subplot(111)
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
alphaval = 0.5
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left() 
ax.tick_params(labelsize=16)
plt.hist(array_output_ICON_INSCAPE[0], bins=nbins, range=[-3., 3], normed=True, color='red', cumulative=False, alpha=alphaval, label='ICON-INSCAPE cloud')
plt.hist(array_output_ICON_INSCAPE[1], bins=nbins, range=[-3., 3], normed=True, color='blue', cumulative=False, alpha=alphaval, label='ICON-INSCAPE no cloud')
plt.legend(loc='upper right', fontsize=12)#range=[np.nanmin(LWP_obs_PBL),np.nanmax(LWP_obs_PBL)]
plt.title('vertical wind below cloud base - PDFs', fontsize=16)
plt.xlabel('vertical wind below cloud base [m/s]', fontsize=16)
plt.ylabel('Occurrences', fontsize=16)
plt.savefig(pathFig+'histogram_WwindSpeedbelowCloudBase_ICON_INSCAPE_20130502.png', format='png')  


nbins     = 20
fig, ax   = plt.subplots(figsize=(14,6))
ax        = plt.subplot(111)
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left() 
ax.tick_params(labelsize=16)
plt.hist(array_output_ICON_INSCAPE[2], bins=nbins, range=[0.,10.], normed=True, color='red', cumulative=False, alpha=alphaval, label='ICON-INSCAPE cloud')
plt.hist(array_output_ICON_INSCAPE[3], bins=nbins, range=[0.,10.], normed=True, color='blue', cumulative=False, alpha=alphaval, label='ICON-INSCAPE no cloud')
plt.legend(loc='upper right', fontsize=12)#range=[np.nanmin(LWP_obs_PBL),np.nanmax(LWP_obs_PBL)]
plt.title('horizontal wind below cloud base - PDFs ICON-INSCAPE', fontsize=16)
plt.xlabel('horizontal wind below cloud base [m/s]', fontsize=16)
plt.ylabel('Occurrences', fontsize=16)
plt.savefig(pathFig+'histogram_HwindSpeedbelowCloudBase_ICON_INSCAPE_20130502.png', format='png')  

# In[19]:

# ----------------------------------------------------------------------------------------
# ------- Analysis of the mean hourly profiles of variance of vertical velocity for obs. ICON-LEM, ICON-INSCAPE
# ----------------------------------------------------------------------------------------


#---- calculating mean variance and standard deviation profiles for each hour of the day for obs and model
print('calculating mean variance and standard deviation profiles for each hour of the day for obs and models')

Variance_w_resH = varW_resampled.values

# defining dataframes for calculating mean
varW_obs_DF                 = pd.DataFrame(Variance_w_resH.transpose(), index=datetime_ICON, columns=height_ICON)
varW_ICON_DF                = pd.DataFrame(varw_ICON, index=datetime_ICON, columns=height_ICON)
varW_ICON_vera_DF           = pd.DataFrame(varw_ICON_vera, index=datetime_ICON_vera, columns=height_ICON_vera)

Profiles_var_DF_obs         = pd.DataFrame(np.zeros((len(height_ICON),24)), columns=np.arange(0,24), index=height_ICON)
Profiles_var_DF_icon        = pd.DataFrame(np.zeros((len(height_ICON),24)), columns=np.arange(0,24), index=height_ICON)
Profiles_var_DF_icon_vera   = pd.DataFrame(np.zeros((len(height_ICON_vera),24)), columns=np.arange(0,24), index=height_ICON_vera)

Std_var_DF_obs              = pd.DataFrame(np.zeros((len(height_ICON),24)), columns=np.arange(0,24), index=height_ICON)
Std_var_DF_icon             = pd.DataFrame(np.zeros((len(height_ICON),24)), columns=np.arange(0,24), index=height_ICON)
Std_var_DF_icon_vera        = pd.DataFrame(np.zeros((len(height_ICON_vera),24)), columns=np.arange(0,24), index=height_ICON_vera)

# order of arguments in datetime : year, month, day, hour, minute, second,
for indHour in range(0,23):
    HourSup                 = datetime.datetime(2013, 5, 2, indHour+1, 0, 0)
    HourInf                 = datetime.datetime(2013, 5, 2, indHour, 0, 0)

    varW_obs_sliced_t       = varW_obs_DF.loc[(varW_obs_DF.index < HourSup) * (varW_obs_DF.index > HourInf),:]
    varW_obs_mean           = varW_obs_sliced_t.mean(axis=0)
    varW_obs_std            = varW_obs_sliced_t.std(axis=0)
    Profiles_var_DF_obs.loc[:,indHour]       = varW_obs_mean
    Std_var_DF_obs.loc[:,indHour]            = varW_obs_std

    varW_icon_sliced_t      = varW_ICON_DF.loc[(varW_ICON_DF.index < HourSup) * (varW_ICON_DF.index > HourInf),:]
    varW_icon_mean          = varW_icon_sliced_t.mean(axis=0)
    varW_icon_std           = varW_icon_sliced_t.std(axis=0)
    Profiles_var_DF_icon.loc[:,indHour]      = varW_icon_mean   
    Std_var_DF_icon.loc[:,indHour]           = varW_icon_std
    
    varW_icon_vera_sliced_t = varW_ICON_vera_DF.loc[(varW_ICON_vera_DF.index < HourSup) * (varW_ICON_vera_DF.index > HourInf),:]
    varW_icon_vera_mean     = varW_icon_vera_sliced_t.mean(axis=0)
    varW_icon_vera_std      = varW_icon_vera_sliced_t.std(axis=0)
    Profiles_var_DF_icon_vera.loc[:,indHour] = varW_icon_vera_mean   
    Std_var_DF_icon_vera.loc[:,indHour]      = varW_icon_vera_std
    
np.shape(Std_var_DF_obs)


print('Plotting hourly profiles of variance of vertical velocity during the day for obs, ICON, ICON_INSCAPE')
# ---- plotting hourly profiles of variance of vertical velocity during the day
fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(14,10))
matplotlib.rcParams['savefig.dpi'] = 300
plt.gcf().subplots_adjust(bottom=0.15)
fig.tight_layout()
ymax=5000.
ymin=107.
xmax=2.
fontSizeTitle=16
fontSizeX=15
fontSizeY=15

ax = plt.subplot(251)  
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left() 
ax.text(1.8, ymax-200., 'a)', fontsize=15)


matplotlib.rc('xtick', labelsize=15)                        # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=15)                        # sets dimension of ticks in the plots
plt.plot(Profiles_var_DF_obs.loc[:,8], height_ICON, label='obs',  color='black')
plt.errorbar(Profiles_var_DF_obs.loc[:,8], height_ICON, xerr=Std_var_DF_obs.loc[:,8], color='black')
plt.plot(Profiles_var_DF_icon.loc[:,8], height_ICON, label='ICON',  color='red')
plt.errorbar(Profiles_var_DF_icon.loc[:,8], height_ICON, xerr=Std_var_DF_icon.loc[:,8], color='red')
plt.plot(Profiles_var_DF_icon_vera.loc[:,8], height_ICON_vera, label='ICON-INSCAPE',  color='blue')
plt.errorbar(Profiles_var_DF_icon_vera.loc[:,8], height_ICON_vera, xerr=Std_var_DF_icon_vera.loc[:,8], color='blue')
plt.legend(loc='upper right', fontsize=14)
plt.ylim(ymin,ymax)
plt.xlim(0.,xmax)
plt.title('8:00 UTC', fontsize=fontSizeTitle)
plt.xlabel('var(w) [m/s]', fontsize=fontSizeX)
plt.ylabel('height [m]', fontsize=fontSizeY)
plt.tight_layout()

ax = plt.subplot(252)  
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left() 
ax.text(1.8, ymax-200., 'b)', fontsize=15)

matplotlib.rc('xtick', labelsize=15)                        # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=15)                        # sets dimension of ticks in the plots
plt.plot(Profiles_var_DF_obs.loc[:,9], height_ICON, label='obs',  color='black')
plt.errorbar(Profiles_var_DF_obs.loc[:,9], height_ICON, xerr=Std_var_DF_obs.loc[:,9], color='black')
plt.plot(Profiles_var_DF_icon.loc[:,9], height_ICON, label='ICON',  color='red')
plt.errorbar(Profiles_var_DF_icon.loc[:,9], height_ICON, xerr=Std_var_DF_icon.loc[:,9], color='red')
plt.plot(Profiles_var_DF_icon_vera.loc[:,9], height_ICON_vera, label='ICON-INSCAPE',  color='blue')
plt.errorbar(Profiles_var_DF_icon_vera.loc[:,9], height_ICON_vera, xerr=Std_var_DF_icon_vera.loc[:,9], color='blue')
plt.ylim(ymin,ymax)
plt.xlim(0.,xmax)
plt.title('9:00 UTC', fontsize=fontSizeTitle)
plt.xlabel('var(w) [m/s]', fontsize=fontSizeX)
plt.tight_layout()

ax = plt.subplot(253)  
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left() 
ax.text(1.8, ymax-200., 'c)', fontsize=15)


matplotlib.rc('xtick', labelsize=15)                        # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=15)                        # sets dimension of ticks in the plots
plt.plot(Profiles_var_DF_obs.loc[:,10], height_ICON, label='obs',  color='black')
plt.errorbar(Profiles_var_DF_obs.loc[:,10], height_ICON, xerr=Std_var_DF_obs.loc[:,10], color='black')
plt.plot(Profiles_var_DF_icon.loc[:,10], height_ICON, label='ICON',  color='red')
plt.errorbar(Profiles_var_DF_icon.loc[:,10], height_ICON, xerr=Std_var_DF_icon.loc[:,10], color='red')
plt.plot(Profiles_var_DF_icon_vera.loc[:,10], height_ICON_vera, label='ICON-INSCAPE',  color='blue')
plt.errorbar(Profiles_var_DF_icon_vera.loc[:,10], height_ICON_vera, xerr=Std_var_DF_icon_vera.loc[:,10], color='blue')
plt.ylim(ymin,ymax)
plt.xlim(0.,xmax)
plt.title('10:00 UTC', fontsize=fontSizeTitle)
plt.xlabel('var(w) [m/s]', fontsize=fontSizeX)
plt.tight_layout()

ax = plt.subplot(254)  
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left() 
ax.text(1.8, ymax-200., 'd)', fontsize=15)

matplotlib.rc('xtick', labelsize=15)                        # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=15)                        # sets dimension of ticks in the plots
plt.plot(Profiles_var_DF_obs.loc[:,11], height_ICON, label='obs',  color='black')
plt.errorbar(Profiles_var_DF_obs.loc[:,11], height_ICON, xerr=Std_var_DF_obs.loc[:,11], color='black')
plt.plot(Profiles_var_DF_icon.loc[:,11], height_ICON, label='ICON',  color='red')
plt.errorbar(Profiles_var_DF_icon.loc[:,11], height_ICON, xerr=Std_var_DF_icon.loc[:,11], color='red')
plt.plot(Profiles_var_DF_icon_vera.loc[:,11], height_ICON_vera, label='ICON-INSCAPE',  color='blue')
plt.errorbar(Profiles_var_DF_icon_vera.loc[:,11], height_ICON_vera, xerr=Std_var_DF_icon_vera.loc[:,11], color='blue')
plt.ylim(ymin,ymax)
plt.xlim(0.,xmax)
plt.title('11:00 UTC', fontsize=fontSizeTitle)
plt.xlabel('var(w) [m/s]', fontsize=fontSizeX)
plt.tight_layout()

ax = plt.subplot(255)  
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left() 
ax.text(1.8, ymax-200., 'e)', fontsize=15)


matplotlib.rc('xtick', labelsize=15)                        # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=15)                        # sets dimension of ticks in the plots
plt.plot(Profiles_var_DF_obs.loc[:,12], height_ICON, label='obs',  color='black')
plt.errorbar(Profiles_var_DF_obs.loc[:,12], height_ICON, xerr=Std_var_DF_obs.loc[:,12], color='black')
plt.plot(Profiles_var_DF_icon.loc[:,12], height_ICON, label='ICON',  color='red')
plt.errorbar(Profiles_var_DF_icon.loc[:,12], height_ICON, xerr=Std_var_DF_icon.loc[:,12], color='red')
plt.plot(Profiles_var_DF_icon_vera.loc[:,12], height_ICON_vera, label='ICON-INSCAPE',  color='blue')
plt.errorbar(Profiles_var_DF_icon_vera.loc[:,12], height_ICON_vera, xerr=Std_var_DF_icon_vera.loc[:,12], color='blue')
plt.ylim(ymin,ymax)
plt.xlim(0.,xmax)
plt.title('12:00 UTC', fontsize=fontSizeTitle)
plt.xlabel('var(w) [m/s]', fontsize=fontSizeX)
plt.tight_layout()

ax = plt.subplot(256)  
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left() 
ax.text(1.8, ymax-200., 'f)', fontsize=15)

matplotlib.rc('xtick', labelsize=15)                        # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=15)                        # sets dimension of ticks in the plots
plt.plot(Profiles_var_DF_obs.loc[:,13], height_ICON, label='obs',  color='black')
plt.errorbar(Profiles_var_DF_obs.loc[:,13], height_ICON, xerr=Std_var_DF_obs.loc[:,13], color='black')
plt.plot(Profiles_var_DF_icon.loc[:,13], height_ICON, label='ICON',  color='red')
plt.errorbar(Profiles_var_DF_icon.loc[:,13], height_ICON, xerr=Std_var_DF_icon.loc[:,13], color='red')
plt.plot(Profiles_var_DF_icon_vera.loc[:,13], height_ICON_vera, label='ICON-INSCAPE',  color='blue')
plt.errorbar(Profiles_var_DF_icon_vera.loc[:,13], height_ICON_vera, xerr=Std_var_DF_icon_vera.loc[:,13], color='blue')
plt.ylim(ymin,ymax)
plt.xlim(0.,xmax)
plt.title('13:00 UTC', fontsize=fontSizeTitle)
plt.xlabel('var(w) [m/s]', fontsize=fontSizeX)
plt.tight_layout()

ax = plt.subplot(257)  
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left() 
ax.text(1.8, ymax-200., 'g)', fontsize=15)


matplotlib.rc('xtick', labelsize=15)                        # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=15)                        # sets dimension of ticks in the plots
plt.plot(Profiles_var_DF_obs.loc[:,14], height_ICON, label='obs',  color='black')
plt.errorbar(Profiles_var_DF_obs.loc[:,14], height_ICON, xerr=Std_var_DF_obs.loc[:,14], color='black')
plt.plot(Profiles_var_DF_icon.loc[:,14], height_ICON, label='ICON',  color='red')
plt.errorbar(Profiles_var_DF_icon.loc[:,14], height_ICON, xerr=Std_var_DF_icon.loc[:,14], color='red')
plt.plot(Profiles_var_DF_icon_vera.loc[:,14], height_ICON_vera, label='ICON-INSCAPE',  color='blue')
plt.errorbar(Profiles_var_DF_icon_vera.loc[:,14], height_ICON_vera, xerr=Std_var_DF_icon_vera.loc[:,14], color='blue')
plt.ylim(ymin,ymax)
plt.xlim(0.,xmax)
plt.title('14:00 UTC', fontsize=fontSizeTitle)
plt.xlabel('var(w) [m/s]', fontsize=fontSizeX)
plt.tight_layout()

ax = plt.subplot(258)  
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left() 
ax.text(1.8, ymax-200. , 'h)', fontsize=15)


matplotlib.rc('xtick', labelsize=15)                        # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=15)                        # sets dimension of ticks in the plots
plt.plot(Profiles_var_DF_obs.loc[:,15], height_ICON, label='obs',  color='black')
plt.errorbar(Profiles_var_DF_obs.loc[:,15], height_ICON, xerr=Std_var_DF_obs.loc[:,15], color='black')
plt.plot(Profiles_var_DF_icon.loc[:,15], height_ICON, label='ICON',  color='red')
plt.errorbar(Profiles_var_DF_icon.loc[:,15], height_ICON, xerr=Std_var_DF_icon.loc[:,15], color='red')
plt.plot(Profiles_var_DF_icon_vera.loc[:,15], height_ICON_vera, label='ICON-INSCAPE',  color='blue')
plt.errorbar(Profiles_var_DF_icon_vera.loc[:,15], height_ICON_vera, xerr=Std_var_DF_icon_vera.loc[:,15], color='blue')
plt.ylim(ymin,ymax)
plt.xlim(0.,xmax)
plt.title('15:00 UTC', fontsize=fontSizeTitle)
plt.xlabel('var(w) [m/s]', fontsize=fontSizeX)
plt.tight_layout()

ax = plt.subplot(259)  
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left() 
ax.text(1.8, ymax-200., 'i)', fontsize=15)

matplotlib.rc('xtick', labelsize=15)                        # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=15)                        # sets dimension of ticks in the plots
plt.plot(Profiles_var_DF_obs.loc[:,16], height_ICON, label='obs',  color='black')
plt.errorbar(Profiles_var_DF_obs.loc[:,16], height_ICON, xerr=Std_var_DF_obs.loc[:,16], color='black')
plt.plot(Profiles_var_DF_icon.loc[:,16], height_ICON, label='ICON',  color='red')
plt.errorbar(Profiles_var_DF_icon.loc[:,16], height_ICON, xerr=Std_var_DF_icon.loc[:,16], color='red')
plt.plot(Profiles_var_DF_icon_vera.loc[:,16], height_ICON_vera, label='ICON-INSCAPE',  color='blue')
plt.errorbar(Profiles_var_DF_icon_vera.loc[:,16], height_ICON_vera, xerr=Std_var_DF_icon_vera.loc[:,16], color='blue')
plt.ylim(ymin,ymax)
plt.xlim(0.,xmax)
plt.title('16:00 UTC', fontsize=fontSizeTitle)
plt.xlabel('var(w) [m/s]', fontsize=fontSizeX)
plt.tight_layout()

ax = plt.subplot(2,5,10)  
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left() 
ax.text(1.8, ymax-200. , 'l)', fontsize=15)

matplotlib.rc('xtick', labelsize=15)                        # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=15)                        # sets dimension of ticks in the plots
plt.plot(Profiles_var_DF_obs.loc[:,17], height_ICON, label='obs',  color='black')
plt.errorbar(Profiles_var_DF_obs.loc[:,17], height_ICON, xerr=Std_var_DF_obs.loc[:,17], color='black')
plt.plot(Profiles_var_DF_icon.loc[:,17], height_ICON, label='ICON',  color='red')
plt.errorbar(Profiles_var_DF_icon.loc[:,17], height_ICON, xerr=Std_var_DF_icon.loc[:,17], color='red')
plt.plot(Profiles_var_DF_icon_vera.loc[:,17], height_ICON_vera, label='ICON-INSCAPE',  color='blue')
plt.errorbar(Profiles_var_DF_icon_vera.loc[:,17], height_ICON_vera, xerr=Std_var_DF_icon_vera.loc[:,17], color='blue')
plt.ylim(ymin,ymax)
plt.xlim(0.,xmax)
plt.title('17:00 UTC', fontsize=fontSizeTitle)
plt.xlabel('var(w) [m/s]', fontsize=fontSizeX)

plt.savefig(pathFig+'varW_Profiles_diurnal_cycle_20130502.pdf', format='pdf')
# In[12]:


# --------------------------------------------------------------------
# ----- calculation of virtual potential temperature profiles for ICON, ICON_INSCAPE, COSMO, radiosondes
# --------------------------------------------------------------------
Rd = 287.058  # gas constant for dry air [Kg-1 K-1 J]
Cp = 1004.

# calculating profiles of virtual potential temperature for model output
Theta_v_ICON = np.zeros((len(datetime_ICON), len(height_ICON)))
Theta_v_COSMO = np.zeros((len(datetime_ICON), len(height_ICON)))
Theta_v_ICON_INSCAPE = np.zeros((len(datetime_ICON_vera), len(height_ICON_vera)))

T_CLOUDNET=T_Cloudnet_ICONres.values.transpose()
P_CLOUDNET=P_Cloudnet_ICONres.values.transpose()
print(np.shape(T_CLOUDNET))

print(np.shape(P_CLOUDNET))

# calculating virtual potential temperature in ICON and COSMO regridded
for indTime in range(len(datetime_ICON)):
    for indHeight in range(len(height_ICON)):
            k_ICON = Rd*(1-0.23*mixingRatio_ICON[indTime, indHeight])/Cp
            Theta_v_ICON[indTime, indHeight]= ( (1 + 0.61 * mixingRatio_ICON[indTime, indHeight]) * \
                                               T_ICON[indTime, indHeight] * (100000./P_ICON[indTime, indHeight])**k_ICON)
            k_COSMO = Rd*(1-0.23*MR_Cloudnet_ICONres[indTime, indHeight])/Cp
            Theta_v_COSMO[indTime, indHeight]= ( (1 + 0.61 * MR_Cloudnet_ICONres[indTime, indHeight]) * T_CLOUDNET[indTime, indHeight] * (100000./P_CLOUDNET[indTime, indHeight])**k_COSMO)

# calculating virtual potential temperature in ICON-INSCAPE
for indTime in range(len(datetime_ICON_vera)):
    for indHeight in range(len(height_ICON_vera)):
        k_ICON_vera = Rd*(1-0.23*mixingRatio_ICON_vera[indTime, indHeight])/Cp
        Theta_v_ICON_INSCAPE[indTime, indHeight]= ( (1 + 0.61 * mixingRatio_ICON_vera[indTime, indHeight]) * \
                                               T_ICON_vera[indTime, indHeight] * (100000./P_ICON_vera[indTime, indHeight])**k_ICON_vera)
        
        


# --------------------------------------------------------------------
# ----- plotting mean over two hours profiles from ICON, ICON-VERA and COSMO and radiosondes together
# --------------------------------------------------------------------

#reducing the thetaV matrix to a matrix only for PBL
H_threshold = 5000.

DF_ThetaV_ICON = pd.DataFrame(Theta_v_ICON, index=datetime_ICON, columns=height_ICON)  
DF_ThetaV_ICON_sliced_t = DF_ThetaV_ICON.loc[:, (DF_ThetaV_ICON.columns < H_threshold) * (DF_ThetaV_ICON.columns > 0.) ]
mask_h_ICON = (DF_ThetaV_ICON.columns < H_threshold) * (DF_ThetaV_ICON.columns > 0.) 

DF_ThetaV_ICON_INSCAPE = pd.DataFrame(Theta_v_ICON_INSCAPE, index=datetime_ICON_vera, columns=height_ICON_vera)  
DF_ThetaV_ICON_INSCAPE_sliced_t = DF_ThetaV_ICON_INSCAPE.loc[:, (DF_ThetaV_ICON_INSCAPE.columns < H_threshold) * (DF_ThetaV_ICON_INSCAPE.columns > 0.) ]
mask_h_ICON_INSCAPE = (DF_ThetaV_ICON_INSCAPE.columns < H_threshold) * (DF_ThetaV_ICON_INSCAPE.columns > 0.) 

DF_ThetaV_COSMO = pd.DataFrame(Theta_v_COSMO, index=datetime_ICON, columns=height_ICON)  
DF_ThetaV_COSMO_sliced_t = DF_ThetaV_COSMO.loc[:, (DF_ThetaV_COSMO.columns < H_threshold) * (DF_ThetaV_COSMO.columns > 0.) ]
mask_h_COSMO = (DF_ThetaV_COSMO.columns < H_threshold) * (DF_ThetaV_COSMO.columns > 0.) 

# plotting two hour interval of thetaV profiles together with the corresponding radiosounding thetaV
TMax = 310.
fig, ax = plt.subplots(figsize=(15,12))
ax = plt.subplot(231)  
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left()  
plt.ylabel('Height [m]', fontsize=16)
plt.xlabel('Virtual potential Temperature [K]', fontsize=16)
plt.title('6:00-8:00 UTC', fontsize=16)
hourstart = 6
hourEnd = 8
mask_t1_ICON = (DF_ThetaV_ICON_sliced_t.index < datetime.datetime(2013,5,2,hourEnd,0,0)) * \
(DF_ThetaV_ICON_sliced_t.index > datetime.datetime(2013,5,2,hourstart,0,0))

mask_t1_INSCAPE = (DF_ThetaV_ICON_INSCAPE_sliced_t.index < datetime.datetime(2013,5,2,hourEnd,0,0)) * \
(DF_ThetaV_ICON_INSCAPE_sliced_t.index > datetime.datetime(2013,5,2,hourstart,0,0))

mask_t1_COSMO = (DF_ThetaV_COSMO_sliced_t.index < datetime.datetime(2013,5,2,hourEnd,0,0)) * \
(DF_ThetaV_COSMO_sliced_t.index > datetime.datetime(2013,5,2,hourstart,0,0))


# calculating mean profiles of sliced matrices for ICON / ICON-INSCAPE
DF_plot_ICON = DF_ThetaV_ICON_sliced_t.loc[mask_t1_ICON,:]
matrix_plot_ICON = DF_plot_ICON.values

DF_plot_ICON_INSCAPE = DF_ThetaV_ICON_INSCAPE_sliced_t.loc[mask_t1_INSCAPE,:]
matrix_plot_ICON_INSCAPE = DF_plot_ICON_INSCAPE.values

DF_plot_COSMO = DF_ThetaV_COSMO_sliced_t.loc[mask_t1_COSMO,:]
matrix_plot_COSMO = DF_plot_COSMO.values


mean_ICON = np.mean(matrix_plot_ICON, axis = 0)
std_ICON = np.std(matrix_plot_ICON, axis = 0)
mean_ICON_INSCAPE = np.mean(matrix_plot_ICON_INSCAPE, axis = 0)
std_ICON_INSCAPE = np.std(matrix_plot_ICON_INSCAPE, axis = 0)
mean_COSMO = np.mean(matrix_plot_COSMO, axis = 0)
std_COSMO = np.std(matrix_plot_COSMO, axis = 0)

#for itime in range(799):
#    plt.plot(matrix_plot[itime,:],height_ICON[mask_h_ICON])
plt.plot(mean_ICON, height_ICON[mask_h_ICON], color='red', label='ICON')
plt.errorbar(mean_ICON, height_ICON[mask_h_ICON], xerr=std_ICON, color='red')

plt.plot(mean_COSMO, height_ICON[mask_h_COSMO], color='green', label='COSMO-EU')
plt.errorbar(mean_COSMO, height_ICON[mask_h_COSMO], xerr=std_COSMO, color='green')

plt.plot(mean_ICON_INSCAPE, height_ICON_vera[mask_h_ICON_INSCAPE], color='blue', label='ICON-INSCAPE')
plt.errorbar(mean_ICON_INSCAPE, height_ICON_vera[mask_h_ICON_INSCAPE], xerr=std_ICON_INSCAPE, color='blue')

plt.plot(Theta_v_radiosondes[:,0],height_radiosondes, color='black', label='radiosonde', linewidth=2.0)
plt.ylim(107.,H_threshold)
plt.xlim(285., TMax)
plt.tight_layout()
#------------------
ax = plt.subplot(232)  
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left()  
plt.ylabel('Height [m]', fontsize=16)
plt.xlabel('Virtual potential Temperature [K]', fontsize=16)
plt.title('8:00-10:00 UTC', fontsize=16)

hourstart = 8
hourEnd = 10
mask_t1_ICON = (DF_ThetaV_ICON_sliced_t.index < datetime.datetime(2013,5,2,hourEnd,0,0)) * \
(DF_ThetaV_ICON_sliced_t.index > datetime.datetime(2013,5,2,hourstart,0,0))

mask_t1_INSCAPE = (DF_ThetaV_ICON_INSCAPE_sliced_t.index < datetime.datetime(2013,5,2,hourEnd,0,0)) * \
(DF_ThetaV_ICON_INSCAPE_sliced_t.index > datetime.datetime(2013,5,2,hourstart,0,0))

mask_t1_COSMO = (DF_ThetaV_COSMO_sliced_t.index < datetime.datetime(2013,5,2,hourEnd,0,0)) * \
(DF_ThetaV_COSMO_sliced_t.index > datetime.datetime(2013,5,2,hourstart,0,0))


# calculating mean profiles of sliced matrices for ICON / ICON-INSCAPE
DF_plot_ICON = DF_ThetaV_ICON_sliced_t.loc[mask_t1_ICON,:]
matrix_plot_ICON = DF_plot_ICON.values

DF_plot_ICON_INSCAPE = DF_ThetaV_ICON_INSCAPE_sliced_t.loc[mask_t1_INSCAPE,:]
matrix_plot_ICON_INSCAPE = DF_plot_ICON_INSCAPE.values

DF_plot_COSMO = DF_ThetaV_COSMO_sliced_t.loc[mask_t1_COSMO,:]
matrix_plot_COSMO = DF_plot_COSMO.values


mean_ICON = np.mean(matrix_plot_ICON, axis = 0)
std_ICON = np.std(matrix_plot_ICON, axis = 0)
mean_ICON_INSCAPE = np.mean(matrix_plot_ICON_INSCAPE, axis = 0)
std_ICON_INSCAPE = np.std(matrix_plot_ICON_INSCAPE, axis = 0)
mean_COSMO = np.mean(matrix_plot_COSMO, axis = 0)
std_COSMO = np.std(matrix_plot_COSMO, axis = 0)

#for itime in range(799):
#    plt.plot(matrix_plot[itime,:],height_ICON[mask_h_ICON])
plt.plot(mean_ICON, height_ICON[mask_h_ICON], color='red', label='ICON')
plt.errorbar(mean_ICON, height_ICON[mask_h_ICON], xerr=std_ICON, color='red')

plt.plot(mean_COSMO, height_ICON[mask_h_COSMO], color='green', label='COSMO-EU')
plt.errorbar(mean_COSMO, height_ICON[mask_h_COSMO], xerr=std_COSMO, color='green')

plt.plot(mean_ICON_INSCAPE, height_ICON_vera[mask_h_ICON_INSCAPE], color='blue', label='ICON-INSCAPE')
plt.errorbar(mean_ICON_INSCAPE, height_ICON_vera[mask_h_ICON_INSCAPE], xerr=std_ICON_INSCAPE, color='blue')

plt.plot(Theta_v_radiosondes[:,1],height_radiosondes, color='black', label='radiosonde', linewidth=2.0)
plt.ylim(107.,H_threshold)
plt.xlim(285., TMax)
plt.tight_layout()


#------------------
ax = plt.subplot(233)  
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left()  
plt.ylabel('Height [m]', fontsize=16)
plt.xlabel('Virtual potential Temperature [K]', fontsize=16)
plt.title('10:00-12:00 UTC', fontsize=16)   
hourstart = 10
hourEnd = 12
mask_t1_ICON = (DF_ThetaV_ICON_sliced_t.index < datetime.datetime(2013,5,2,hourEnd,0,0)) * \
(DF_ThetaV_ICON_sliced_t.index > datetime.datetime(2013,5,2,hourstart,0,0))

mask_t1_INSCAPE = (DF_ThetaV_ICON_INSCAPE_sliced_t.index < datetime.datetime(2013,5,2,hourEnd,0,0)) * \
(DF_ThetaV_ICON_INSCAPE_sliced_t.index > datetime.datetime(2013,5,2,hourstart,0,0))

mask_t1_COSMO = (DF_ThetaV_COSMO_sliced_t.index < datetime.datetime(2013,5,2,hourEnd,0,0)) * \
(DF_ThetaV_COSMO_sliced_t.index > datetime.datetime(2013,5,2,hourstart,0,0))


# calculating mean profiles of sliced matrices for ICON / ICON-INSCAPE
DF_plot_ICON = DF_ThetaV_ICON_sliced_t.loc[mask_t1_ICON,:]
matrix_plot_ICON = DF_plot_ICON.values

DF_plot_ICON_INSCAPE = DF_ThetaV_ICON_INSCAPE_sliced_t.loc[mask_t1_INSCAPE,:]
matrix_plot_ICON_INSCAPE = DF_plot_ICON_INSCAPE.values

DF_plot_COSMO = DF_ThetaV_COSMO_sliced_t.loc[mask_t1_COSMO,:]
matrix_plot_COSMO = DF_plot_COSMO.values


mean_ICON = np.mean(matrix_plot_ICON, axis = 0)
std_ICON = np.std(matrix_plot_ICON, axis = 0)
mean_ICON_INSCAPE = np.mean(matrix_plot_ICON_INSCAPE, axis = 0)
std_ICON_INSCAPE = np.std(matrix_plot_ICON_INSCAPE, axis = 0)
mean_COSMO = np.mean(matrix_plot_COSMO, axis = 0)
std_COSMO = np.std(matrix_plot_COSMO, axis = 0)

#for itime in range(799):
#    plt.plot(matrix_plot[itime,:],height_ICON[mask_h_ICON])
plt.plot(mean_ICON, height_ICON[mask_h_ICON], color='red', label='ICON')
plt.errorbar(mean_ICON, height_ICON[mask_h_ICON], xerr=std_ICON, color='red')

plt.plot(mean_COSMO, height_ICON[mask_h_COSMO], color='green', label='COSMO-EU')
plt.errorbar(mean_COSMO, height_ICON[mask_h_COSMO], xerr=std_COSMO, color='green')

plt.plot(mean_ICON_INSCAPE, height_ICON_vera[mask_h_ICON_INSCAPE], color='blue', label='ICON-INSCAPE')
plt.errorbar(mean_ICON_INSCAPE, height_ICON_vera[mask_h_ICON_INSCAPE], xerr=std_ICON_INSCAPE, color='blue')

plt.plot(Theta_v_radiosondes[:,2],height_radiosondes, color='black', label='radiosonde', linewidth=2.0)
plt.ylim(107.,H_threshold)
plt.xlim(285., TMax)
plt.tight_layout()




#------------------
ax = plt.subplot(234)  
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left()  
plt.ylabel('Height [m]', fontsize=16)
plt.xlabel('Virtual potential Temperature [K]', fontsize=16)
plt.title('12:00-14:00 UTC ', fontsize=16)

hourstart = 12
hourEnd = 14
mask_t1_ICON = (DF_ThetaV_ICON_sliced_t.index < datetime.datetime(2013,5,2,hourEnd,0,0)) * \
(DF_ThetaV_ICON_sliced_t.index > datetime.datetime(2013,5,2,hourstart,0,0))

mask_t1_INSCAPE = (DF_ThetaV_ICON_INSCAPE_sliced_t.index < datetime.datetime(2013,5,2,hourEnd,0,0)) * \
(DF_ThetaV_ICON_INSCAPE_sliced_t.index > datetime.datetime(2013,5,2,hourstart,0,0))

mask_t1_COSMO = (DF_ThetaV_COSMO_sliced_t.index < datetime.datetime(2013,5,2,hourEnd,0,0)) * \
(DF_ThetaV_COSMO_sliced_t.index > datetime.datetime(2013,5,2,hourstart,0,0))


# calculating mean profiles of sliced matrices for ICON / ICON-INSCAPE
DF_plot_ICON = DF_ThetaV_ICON_sliced_t.loc[mask_t1_ICON,:]
matrix_plot_ICON = DF_plot_ICON.values

DF_plot_ICON_INSCAPE = DF_ThetaV_ICON_INSCAPE_sliced_t.loc[mask_t1_INSCAPE,:]
matrix_plot_ICON_INSCAPE = DF_plot_ICON_INSCAPE.values

DF_plot_COSMO = DF_ThetaV_COSMO_sliced_t.loc[mask_t1_COSMO,:]
matrix_plot_COSMO = DF_plot_COSMO.values


mean_ICON = np.mean(matrix_plot_ICON, axis = 0)
std_ICON = np.std(matrix_plot_ICON, axis = 0)
mean_ICON_INSCAPE = np.mean(matrix_plot_ICON_INSCAPE, axis = 0)
std_ICON_INSCAPE = np.std(matrix_plot_ICON_INSCAPE, axis = 0)
mean_COSMO = np.mean(matrix_plot_COSMO, axis = 0)
std_COSMO = np.std(matrix_plot_COSMO, axis = 0)

#for itime in range(799):
#    plt.plot(matrix_plot[itime,:],height_ICON[mask_h_ICON])
plt.plot(mean_ICON, height_ICON[mask_h_ICON], color='red', label='ICON')
plt.errorbar(mean_ICON, height_ICON[mask_h_ICON], xerr=std_ICON, color='red')

plt.plot(mean_COSMO, height_ICON[mask_h_COSMO], color='green', label='COSMO-EU')
plt.errorbar(mean_COSMO, height_ICON[mask_h_COSMO], xerr=std_COSMO, color='green')

plt.plot(mean_ICON_INSCAPE, height_ICON_vera[mask_h_ICON_INSCAPE], color='blue', label='ICON-INSCAPE')
plt.errorbar(mean_ICON_INSCAPE, height_ICON_vera[mask_h_ICON_INSCAPE], xerr=std_ICON_INSCAPE, color='blue')

plt.plot(Theta_v_radiosondes[:,3],height_radiosondes, color='black', label='radiosonde', linewidth=2.0)

plt.plot(Theta_v_RadiosondeEssen, height_RadiosondeEssen, color = 'black', \
         linewidth=2.0, linestyle=':',label='radiosonde (Essen)')
plt.ylim(107.,H_threshold)
plt.xlim(285., TMax)
plt.tight_layout()



#------------------
ax = plt.subplot(235)  
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left()  
plt.ylabel('Height [m]', fontsize=16)
plt.xlabel('Virtual potential Temperature [K]', fontsize=16)
plt.title('14:00-16:00', fontsize=16)

hourstart = 14
hourEnd = 16

mask_t1_ICON = (DF_ThetaV_ICON_sliced_t.index < datetime.datetime(2013,5,2,hourEnd,0,0)) * \
(DF_ThetaV_ICON_sliced_t.index > datetime.datetime(2013,5,2,hourstart,0,0))

mask_t1_INSCAPE = (DF_ThetaV_ICON_INSCAPE_sliced_t.index < datetime.datetime(2013,5,2,hourEnd,0,0)) * \
(DF_ThetaV_ICON_INSCAPE_sliced_t.index > datetime.datetime(2013,5,2,hourstart,0,0))

mask_t1_COSMO = (DF_ThetaV_COSMO_sliced_t.index < datetime.datetime(2013,5,2,hourEnd,0,0)) * \
(DF_ThetaV_COSMO_sliced_t.index > datetime.datetime(2013,5,2,hourstart,0,0))


# calculating mean profiles of sliced matrices for ICON / ICON-INSCAPE
DF_plot_ICON = DF_ThetaV_ICON_sliced_t.loc[mask_t1_ICON,:]
matrix_plot_ICON = DF_plot_ICON.values

DF_plot_ICON_INSCAPE = DF_ThetaV_ICON_INSCAPE_sliced_t.loc[mask_t1_INSCAPE,:]
matrix_plot_ICON_INSCAPE = DF_plot_ICON_INSCAPE.values

DF_plot_COSMO = DF_ThetaV_COSMO_sliced_t.loc[mask_t1_COSMO,:]
matrix_plot_COSMO = DF_plot_COSMO.values


mean_ICON = np.mean(matrix_plot_ICON, axis = 0)
std_ICON = np.std(matrix_plot_ICON, axis = 0)
mean_ICON_INSCAPE = np.mean(matrix_plot_ICON_INSCAPE, axis = 0)
std_ICON_INSCAPE = np.std(matrix_plot_ICON_INSCAPE, axis = 0)
mean_COSMO = np.mean(matrix_plot_COSMO, axis = 0)
std_COSMO = np.std(matrix_plot_COSMO, axis = 0)

#for itime in range(799):
#    plt.plot(matrix_plot[itime,:],height_ICON[mask_h_ICON])
plt.plot(mean_ICON, height_ICON[mask_h_ICON], color='red', label='ICON')
plt.errorbar(mean_ICON, height_ICON[mask_h_ICON], xerr=std_ICON, color='red')

plt.plot(mean_COSMO, height_ICON[mask_h_COSMO], color='green', label='COSMO-EU')
plt.errorbar(mean_COSMO, height_ICON[mask_h_COSMO], xerr=std_COSMO, color='green')

plt.plot(mean_ICON_INSCAPE, height_ICON_vera[mask_h_ICON_INSCAPE], color='blue', label='ICON-INSCAPE')
plt.errorbar(mean_ICON_INSCAPE, height_ICON_vera[mask_h_ICON_INSCAPE], xerr=std_ICON_INSCAPE, color='blue')

plt.plot(Theta_v_radiosondes[:,4],height_radiosondes, color='black', label='radiosonde', linewidth=2.0)

plt.ylim(107.,H_threshold)
plt.xlim(285., TMax)



#------------------
ax = plt.subplot(236)  
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left()  
plt.ylabel('Height [m]', fontsize=16)
plt.xlabel('Virtual potential Temperature [K]', fontsize=16)
plt.title('16:00-18:00', fontsize=16)

hourstart = 16
hourEnd = 18

mask_t1_ICON = (DF_ThetaV_ICON_sliced_t.index < datetime.datetime(2013,5,2,hourEnd,0,0)) * \
(DF_ThetaV_ICON_sliced_t.index > datetime.datetime(2013,5,2,hourstart,0,0))

mask_t1_INSCAPE = (DF_ThetaV_ICON_INSCAPE_sliced_t.index < datetime.datetime(2013,5,2,hourEnd,0,0)) * \
(DF_ThetaV_ICON_INSCAPE_sliced_t.index > datetime.datetime(2013,5,2,hourstart,0,0))

mask_t1_COSMO = (DF_ThetaV_COSMO_sliced_t.index < datetime.datetime(2013,5,2,hourEnd,0,0)) * \
(DF_ThetaV_COSMO_sliced_t.index > datetime.datetime(2013,5,2,hourstart,0,0))


# calculating mean profiles of sliced matrices for ICON / ICON-INSCAPE
DF_plot_ICON = DF_ThetaV_ICON_sliced_t.loc[mask_t1_ICON,:]
matrix_plot_ICON = DF_plot_ICON.values

DF_plot_ICON_INSCAPE = DF_ThetaV_ICON_INSCAPE_sliced_t.loc[mask_t1_INSCAPE,:]
matrix_plot_ICON_INSCAPE = DF_plot_ICON_INSCAPE.values

DF_plot_COSMO = DF_ThetaV_COSMO_sliced_t.loc[mask_t1_COSMO,:]
matrix_plot_COSMO = DF_plot_COSMO.values


mean_ICON = np.mean(matrix_plot_ICON, axis = 0)
std_ICON = np.std(matrix_plot_ICON, axis = 0)
mean_ICON_INSCAPE = np.mean(matrix_plot_ICON_INSCAPE, axis = 0)
std_ICON_INSCAPE = np.std(matrix_plot_ICON_INSCAPE, axis = 0)
mean_COSMO = np.mean(matrix_plot_COSMO, axis = 0)
std_COSMO = np.std(matrix_plot_COSMO, axis = 0)

#for itime in range(799):
#    plt.plot(matrix_plot[itime,:],height_ICON[mask_h_ICON])
plt.plot(mean_ICON, height_ICON[mask_h_ICON], color='red', label='ICON')
plt.errorbar(mean_ICON, height_ICON[mask_h_ICON], xerr=std_ICON, color='red')

plt.plot(mean_COSMO, height_ICON[mask_h_COSMO], color='green', label='COSMO-EU')
plt.errorbar(mean_COSMO, height_ICON[mask_h_COSMO], xerr=std_COSMO, color='green')

plt.plot(mean_ICON_INSCAPE, height_ICON_vera[mask_h_ICON_INSCAPE], color='blue', label='ICON-INSCAPE')
plt.errorbar(mean_ICON_INSCAPE, height_ICON_vera[mask_h_ICON_INSCAPE], xerr=std_ICON_INSCAPE, color='blue')

plt.plot(Theta_v_radiosondes[:,5],height_radiosondes, color='black', label='radiosonde', linewidth=2.0)


plt.legend(loc='lower right', fontsize=16)
plt.ylim(107.,H_threshold)
plt.xlim(285., TMax)
plt.tight_layout()
plt.savefig(pathFig+'hourlyProfiles_thetaV_20130502.pdf', format='pdf')