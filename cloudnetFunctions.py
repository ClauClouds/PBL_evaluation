import numpy as np
import matplotlib
import scipy
import netCDF4
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset
#from cf_colors import RGB_colors_red2blue



def CheckIceLiquidCloudnet(array):
    
    
    dimArr=len(array)           # reading dimension for array in input
    stringFlag=[]               # generating output string list
    #print dimArr
    
    for ind in range(dimArr):
        
        # reading the element of the array
        elem=bin(array[ind])[2:]        # reading the element in binary base 
        #least significative on the right end of the number and cutting first two characters '0b' 
        #print array[ind]
        #print bin(array[ind])[2:]
        
        length=len(elem)           # counting number of digits that represent the number
        
        # adapting lenght of digits to the maximum lenght possible that can be found in Cloudnet (6 bits) if the length of the string is smaller
        if length < 6:
            Nbins2add= 6 - length               # number of zeros to add to the left 
            elem = '0' * Nbins2add + elem
            # print 'resized elem'
            # print elem
            
        # flags for bits of cloudnet that are on
        flagBin0 = int((elem)[-1])      # bit 0: small liquid droplets on
        flagBin1 = int((elem)[-2])      # bit 1: falling hydrometeors
        flagBin2 = int((elem)[-3])      # bit 2: wet bulb < 0, if bit 1 on, then phase
        flagBin3 = int((elem)[-4])      # bit 3: melting ice particles
        
        #print flagBin0, flagBin1, flagBin2, flagBin3
        # condition for only liquid clouds
        if ((flagBin0 == 1) and (flagBin2 == 0) and (flagBin3 == 0)):
            stringFlag.append('liquid') # cloud droplets and drizzle, cloud droplets only
            #print 'sono qui'
        if ((flagBin1 == 1) and (flagBin2 == 1)): 
            stringFlag.append('ice')
        if (flagBin3 == 1):
            stringFlag.append('ice')
        if ((flagBin0 == 0) and (flagBin1 == 0) and (flagBin2 == 0) and (flagBin3 == 0)):
            stringFlag.append('none')

    return stringFlag


def f_calculateCloudMaskCloudnet(time, height, cloudnet):
    #---------------------------------------------------------------------------------------------------------
    # date : 13 April 2018 
    # author: Claudia Acquistapace
    # abstract : routine to calculate cloud mask matrix from cloudnet target classification
    # input :
    #   - time
    #   - height
    #   - cloudnet categorization in bits
    # output:
    #   - cloud mask
    #---------------------------------------------------------------------------------------------------------      
    dimTime = len(time)
    dimHeight = len(height)
    cloudMask = np.zeros(shape=(dimTime,dimHeight))		# matrix for cloud mask
    
    # loop on hour array for calculating cloud fraction
    for itime in range(dimTime-1):
        for iheight in range(dimHeight):
            
            
            # reading the element of the array
            elem=bin(cloudnet[itime, iheight])[2:]        # reading the element in binary base 
            #least significative on the right end of the number and cutting first two characters '0b' 
            #print array[ind]
            #print bin(array[ind])[2:]
        
            length=len(elem)           # counting number of digits that represent the number
        
            # adapting lenght of digits to the maximum lenght possible that can be found in Cloudnet (6 bits) if the length of the string is smaller
            if length < 6:
                Nbins2add= 6 - length               # number of zeros to add to the left 
                elem = '0' * Nbins2add + elem
                # print 'resized elem'
                # print elem
            
            # flags for bits of cloudnet that are on
            flagBin0 = int((elem)[-1])      # bit 0: small liquid droplets on
            flagBin1 = int((elem)[-2])      # bit 1: falling hydrometeors
            flagBin2 = int((elem)[-3])      # bit 2: wet bulb < 0, if bit 1 on, then phase
            flagBin3 = int((elem)[-4])      # bit 3: melting ice particles
            
                    # condition for only liquid clouds
            if ((flagBin0 == 1) and (flagBin2 == 0) and (flagBin3 == 0)):
                cloudMask[itime,iheight] = 1  # liquid
                #stringFlag.append('liquid') # cloud droplets and drizzle, cloud droplets only
                #print 'sono qui'
            if ((flagBin1 == 1) and (flagBin2 == 1)): 
                cloudMask[itime,iheight] = 2  # ice                
                #stringFlag.append('ice')
            if (flagBin3 == 1):
                cloudMask[itime,iheight] = 2  # ice                
                #stringFlag.append('ice')
            if ((flagBin0 == 0) and (flagBin1 == 0) and (flagBin2 == 0) and (flagBin3 == 0)):
                cloudMask[itime,iheight] = 0  # no cloud               
                #stringFlag.append('none')

    return (cloudMask)

def f_calculateCloudFractionCloudnet(date,site,pathIn, pathOutPlot):
    #---------------------------------------------------------------------------------------------------------
    # date : 13 November 2017 (modified version from the previous code in september)
    # author: Claudia Acquistapace
    # abstract : routine to calculate mean cloud fraction every hour for and mean profile for every six hours of the day for total cloud fraction, ice cloud fraction and liquid cloud fraction. This is done for a specific site and for the whole day. The routine also provides an output dictionary containing the cloud fraction profiles and their standard deviations, for the hourly and six hourly mean and for each phase.
    # input :
    #   - date to process
    #   - site of the measurements
    #   - path to the data
    # output:
    #   - dictionary containing data
    #   - plots of the daily cloud fractions in the 
    #---------------------------------------------------------------------------------------------------------


    
    # -----------read cloudnet classification file, time and height arrays--------------------
    print('load cloudnet target class ncdf file')

    # opening file
    data = Dataset(pathIn+date+'_'+site+'_categorize.nc', mode='r')

    # reading time and height variables
    time = data.variables['time'][:].copy()  # reading time array
    height = data.variables['height'][:].copy()  # reading height array
    cloudnet = data.variables['category_bits'][:].copy()  # reading cloudnet classification

    # closing file
    data.close()




    # building hourly array for cloud fraction calculations
    hourArr = np.array(range(24))

    # generating output matrices
    CFTCloudnet=np.zeros(shape=(24,len(height)))		# matrix for total cloud fraction (ice+liquid)
    CFICloudnet=np.zeros(shape=(24,len(height)))		# matrix for ice cloud fraction 
    CFLCloudnet=np.zeros(shape=(24,len(height)))		# matrix for liquid cloud fraction 

    # building a dataframe for the cloudnet target classification
    CloudnetDF = pd.DataFrame(cloudnet,index=time, columns=height)

    # loop on hour array for calculating cloud fraction
    for itime in range(len(hourArr)-1):
        for iheight in range(len(height)):
            
            # selecting lines corresponding to measurements within the hour
            CloudnetDFArr = CloudnetDF.loc[ (CloudnetDF.index < hourArr[itime+1]) * (CloudnetDF.index >= hourArr[itime]), height[iheight]]
        
            # applying the function to check for ice/liquid cloud presence:returns an array of flags for the hour, where you find liquid/ice clouds at that heigth level
            stringArr=CheckIceLiquidCloudnet(CloudnetDFArr.values)
            
            # calculating cloud fraction in probabilistic way Nsel/Ntot
            Ntot=len(CloudnetDFArr.values)                     # total number of bins in the column
            Nliquid = stringArr.count('liquid')     # number of liquid clouds
            Nice = stringArr.count('ice')           # number of ice clouds
            
            # calculating cloud fractions for the time interval selected at the height j
            if Ntot == 0: 
                CFTCloudnet[itime,iheight]=0.
                CFICloudnet[itime,iheight]=0.
                CFLCloudnet[itime,iheight]=0.
            else:
                CFTCloudnet[itime,iheight]=float(Nice+Nliquid)/float(Ntot)
                # total cloud fraction (liquid+ic/mixed phase clouds)
                CFICloudnet[itime,iheight]=float(Nice)/float(Ntot)          # ice cloud fraction
                CFLCloudnet[itime,iheight]=float(Nliquid)/float(Ntot)            # liquid cloud fraction
                
                
                
    # plotting hourly profiles
    plt.figure('CF_hourly_iceliquid_'+date, figsize=[10,12])
    matplotlib.rc('xtick', labelsize=15)                        # sets dimension of ticks in the plots
    matplotlib.rc('ytick', labelsize=15)                        # sets dimension of ticks in the plots
    nameHourArr=['00','01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
    colorArr=['darkblue', 'blue', 'lightblue','aqua','cyan','green','lime', 'lavender', 'fuchsia', 'magenta', 'plum','red','orange','crimson','chocolate', 'beige','khaki', 'goldenrod', 'yellow', 'pink', 'orchid','purple','grey', 'black']

    plt.title('Cloud fraction per hour on the '+date, loc='center')
    # first subplot for liquid clouds
    subfig1 = plt.subplot(121) 
    subfig1.spines["top"].set_visible(False)  
    subfig1.spines["right"].set_visible(False) 
    subfig1.set_title(site+' - '+date+' liquid', fontsize=20)
    #plt.axes()
    plt.xlabel('Cloud fraction [%]', fontsize=14)
    plt.ylabel('Height [m]', fontsize=14)
    subfig1.set_ylim(0., 12000.)
    for i in range(0,24):
        plt.plot(CFLCloudnet[i,:], height, label=nameHourArr[i]+' UTC', color=colorArr[i])
            #plt.errorbar(CFIMeanCloudnet[i,:], height, color=colors_6hours[i], linewidth=2, elinewidth=0.3, label=nameHourArr[i], xerr=CFIStdCloudnet[i,:])
        #plt.legend(loc='center right', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)   

    # second subplot for ice clouds
    subfig2 = plt.subplot(122)
    subfig2.set_title(site+' - '+date+' ice', fontsize=14)
    #plt.axes()
    plt.xlabel(' Cloud fraction [%]', fontsize=14)
    #plt.ylabel('Height [m]', fontsize=20)
    subfig2.set_ylim(0., 12000.)
    for i in range(0,24):
        plt.plot(CFICloudnet[i,:], height, label=nameHourArr[i]+' UTC', color=colorArr[i])
        #plt.errorbar(CFLMeanCloudnet[i,:], height, color=colors_6hours[i], linewidth=2, elinewidth=0.3, label=nameHourArr[i], xerr=CFLStdCloudnet[i,:])
        
    figname=pathOutPlot+'CFtot_HourlyProfiles_'+site+'_'+date+'.pdf'
    plt.savefig(figname) 


    # plot figure of cloud fractions for every hour
    from cf_colors import RGB_colors_red2blue

    # reading colors for color bar plotting
    new_colors = RGB_colors_red2blue(N_colors = len(hourArr))[1] 
    
    
    #---------------calculating mean and standard deviation profiles every six hours
    # defining matrices to store the data
    CFTMeanCloudnet=np.zeros(shape=(4,len(height)))
    CFIMeanCloudnet=np.zeros(shape=(4,len(height)))
    CFLMeanCloudnet=np.zeros(shape=(4,len(height)))
    CFTStdCloudnet=np.zeros(shape=(4,len(height)))
    CFIStdCloudnet=np.zeros(shape=(4,len(height)))
    CFLStdCloudnet=np.zeros(shape=(4,len(height)))


    # calculating mean profiles and standard deviation for each interval of six hours for total cloud fraction
    CFTMeanCloudnet[0,:]=np.mean(CFTCloudnet[0:5,:], axis=0)        # axis=0 sets the average to be done on the selected columns
    CFTMeanCloudnet[1,:]=np.mean(CFTCloudnet[6:11,:], axis=0)       # axis=0 sets the average to be done on the selected columns
    CFTMeanCloudnet[2,:]=np.mean(CFTCloudnet[12:17,:], axis=0)      # axis=0 sets the average to be done on the selected columns
    CFTMeanCloudnet[3,:]=np.mean(CFTCloudnet[18:23,:], axis=0)      # axis=0 sets the average to be done on the selected columns

    CFTStdCloudnet[0,:]=np.std(CFTCloudnet[0:5,:], axis=0)        # axis=0 sets the average to be done on the selected columns
    CFTStdCloudnet[1,:]=np.std(CFTCloudnet[6:11,:], axis=0)       # axis=0 sets the average to be done on the selected columns
    CFTStdCloudnet[2,:]=np.std(CFTCloudnet[12:17,:], axis=0)      # axis=0 sets the average to be done on the selected columns
    CFTStdCloudnet[3,:]=np.std(CFTCloudnet[18:23,:], axis=0)      # axis=0 sets the average to be done on the selected columns

    # calculating mean profiles for each interval of six hours for ice cloud fraction
    CFIMeanCloudnet[0,:]=np.mean(CFICloudnet[0:5,:], axis=0)        # axis=0 sets the average to be done on the selected columns
    CFIMeanCloudnet[1,:]=np.mean(CFICloudnet[6:11,:], axis=0)       # axis=0 sets the average to be done on the selected columns
    CFIMeanCloudnet[2,:]=np.mean(CFICloudnet[12:17,:], axis=0)      # axis=0 sets the average to be done on the selected columns
    CFIMeanCloudnet[3,:]=np.mean(CFICloudnet[18:23,:], axis=0)      # axis=0 sets the average to be done on the selected columns

    CFIStdCloudnet[0,:]=np.std(CFICloudnet[0:5,:], axis=0)        # axis=0 sets the average to be done on the selected columns
    CFIStdCloudnet[1,:]=np.std(CFICloudnet[6:11,:], axis=0)       # axis=0 sets the average to be done on the selected columns
    CFIStdCloudnet[2,:]=np.std(CFICloudnet[12:17,:], axis=0)      # axis=0 sets the average to be done on the selected columns
    CFIStdCloudnet[3,:]=np.std(CFICloudnet[18:23,:], axis=0)      # axis=0 sets the average to be done on the selected columns


    # calculating mean profiles for each interval of six hours for ice cloud fraction
    CFLMeanCloudnet[0,:]=np.mean(CFLCloudnet[0:5,:], axis=0)        # axis=0 sets the average to be done on the selected columns
    CFLMeanCloudnet[1,:]=np.mean(CFLCloudnet[6:11,:], axis=0)       # axis=0 sets the average to be done on the selected columns
    CFLMeanCloudnet[2,:]=np.mean(CFLCloudnet[12:17,:], axis=0)      # axis=0 sets the average to be done on the selected columns
    CFLMeanCloudnet[3,:]=np.mean(CFLCloudnet[18:23,:], axis=0)      # axis=0 sets the average to be done on the selected columns

    CFLStdCloudnet[0,:]=np.std(CFLCloudnet[0:5,:], axis=0)        # axis=0 sets the average to be done on the selected columns
    CFLStdCloudnet[1,:]=np.std(CFLCloudnet[6:11,:], axis=0)       # axis=0 sets the average to be done on the selected columns
    CFLStdCloudnet[2,:]=np.std(CFLCloudnet[12:17,:], axis=0)      # axis=0 sets the average to be done on the selected columns
    CFLStdCloudnet[3,:]=np.std(CFLCloudnet[18:23,:], axis=0)      # axis=0 sets the average to be done on the selected columns



    #-------plot cloud fraction mean over six hours
    # reading colors for color bar plotting
    nameHourArr=[]
    nameHourArr.append('early morning: 0-6 UTC')
    nameHourArr.append('morning 6-12 UTC')
    nameHourArr.append('afternoon 12-18 UTC')
    nameHourArr.append('night 18-24 UTC')
    Nprofiles=len(nameHourArr)

    # defining color array 
    #colors_6hours = RGB_colors_red2blue(N_colors = 4)[1]
    colorArr=['darkblue', 'blue', 'lightblue','aqua','cyan','green','lime', 'lavender', 'fuchsia', 'magenta', 'plum','red','orange','crimson','chocolate', 'beige','khaki', 'goldenrod', 'yellow', 'pink', 'orchid','purple','grey', 'black']

    # opening figure for plot over 6 hours for total cloud fraction (with errorbars)
    fig1 = plt.figure('CF_mean6Hours', figsize=[10,10])
    matplotlib.rc('xtick', labelsize=14) 
    matplotlib.rc('ytick', labelsize=14) 
    plt.suptitle('Cloud fraction profiles over 6 hours', fontsize=14)
    plt.axes()
    plt.xlabel('Mean Cloud fraction over 6 hours [%]', fontsize=14)
    plt.ylabel('Height [m]', fontsize=14)
    plt.ylim(0., 12000.)
    for i in range(0,Nprofiles):
        plt.plot(CFTMeanCloudnet[i,:], height, color=colors_6hours[i], label=nameHourArr[i],linewidth=1.5)
        #plt.errorbar(CFTMeanCloudnet[i,:], height, color=colors_6hours[i], linewidth=2, elinewidth=0.3, label=nameHourArr[i], xerr=CFTStdCloudnet[i,:])
    plt.legend(loc='lower right', fontsize=14)
    figname=pathOutPlot+'CFtot_6Hours_'+site+'_'+date+'.pdf'
    fig1.savefig(figname,fontsize=24)  
    plt.clf()
    plt.cla()
    plt.close()

    #------plot figure for cloud fraction every 6 hours for liquid and ice, with subplots
    fig2 = plt.figure('CF_mean6Hours_ice_liquid', figsize=[10,10])
    matplotlib.rc('xtick', labelsize=14)                        # sets dimension of ticks in the plots
    matplotlib.rc('ytick', labelsize=14)                        # sets dimension of ticks in the plots

    # first subplot for ice clouds
    subfig1 = plt.subplot(121) 
    subfig1.set_title('ice clouds', fontsize=14)
    #plt.axes()
    #plt.xlabel('Mean Cloud fraction over 6 hours [%]', fontsize=20)
    #plt.ylabel('Height [m]', fontsize=20)
    subfig1.set_ylim(0., 12000.)
    for i in range(0,Nprofiles):
        plt.plot(CFIMeanCloudnet[i,:], height, color=colors_6hours[i], label=nameHourArr[i],linewidth=1.5)
        #plt.errorbar(CFIMeanCloudnet[i,:], height, color=colors_6hours[i], linewidth=2, elinewidth=0.3, label=nameHourArr[i], xerr=CFIStdCloudnet[i,:])
    #plt.legend(loc='center right', fontsize=15)


    # second subplot for liquid clouds
    subfig2 = plt.subplot(122)
    subfig2.set_title('liquid clouds', fontsize=20)
    #plt.axes()
    #plt.xlabel('Mean Cloud fraction over 6 hours [%]', fontsize=20)
    #plt.ylabel('Height [m]', fontsize=20)
    subfig2.set_ylim(0., 12000.)
    for i in range(0,Nprofiles):
        plt.plot(CFLMeanCloudnet[i,:], height, color=colors_6hours[i], label=nameHourArr[i],linewidth=1.5)
        #plt.errorbar(CFLMeanCloudnet[i,:], height, color=colors_6hours[i], linewidth=2, elinewidth=0.3, label=nameHourArr[i], xerr=CFLStdCloudnet[i,:])
    
    plt.legend(loc='upper right', fontsize=15)    
        
    figname=pathOutPlot+'CFice_liquid_6Hours_'+site+'_'+date+'.pdf'
    fig2.savefig(figname,fontsize=24)  
    plt.clf()
    plt.cla()
    plt.close()
    
    # defining dictionary containing data to have as output
    dict06={}
    dict612={}
    dict1218={}
    dict1824={}
    
    
    dict06={
        date+'CF_tot_0_6':CFTMeanCloudnet[0,:],
        date+'CF_tot_std_0_6':CFTStdCloudnet[0,:],
        date+'CFI_tot_0_6':CFIMeanCloudnet[0,:],
        date+'CFI_tot_std_0_6':CFIStdCloudnet[0,:],
        date+'CFL_tot_0_6':CFLMeanCloudnet[0,:],
        date+'CFL_tot_std_0_6':CFLStdCloudnet[0,:],
        date+'height':height
        }
    dict612={
        date+'CF_tot_6_12':CFTMeanCloudnet[1,:],
        date+'CF_tot_std_6_12':CFTStdCloudnet[1,:],
        date+'CFI_tot_6_12':CFIMeanCloudnet[1,:],
        date+'CFI_tot_std_6_12':CFIStdCloudnet[1,:],        
        date+'CFL_tot_6_12':CFLMeanCloudnet[1,:],
        date+'CFL_tot_std_6_12':CFLStdCloudnet[1,:],
        date+'height':height        
        }
    dict1218={    
        date+'CF_tot_12_18':CFTMeanCloudnet[2,:],
        date+'CF_tot_std_12_18':CFTStdCloudnet[2,:],
        date+'CFI_tot_12_18':CFIMeanCloudnet[2,:],
        date+'CFI_tot_std_12_18':CFIStdCloudnet[2,:],        
        date+'CFL_tot_12_18':CFLMeanCloudnet[2,:],
        date+'CFL_tot_std_12_18':CFLStdCloudnet[2,:],
        date+'height':height
        }
    dict1824={         
        date+'CF_tot_18_24':CFTMeanCloudnet[3,:],
        date+'CF_tot_std_18_24':CFTStdCloudnet[3,:], 
        date+'CFI_tot_18_24':CFIMeanCloudnet[3,:],
        date+'CFI_tot_std_18_24':CFIStdCloudnet[3,:],
        date+'CFL_tot_18_24':CFLMeanCloudnet[3,:],
        date+'CFL_tot_std_18_24':CFLStdCloudnet[3,:],
        date+'height':height    
        }
        
    return (dict06, dict612, dict1218, dict1824)









def f_calculateCloudFractionCloudnet_v2(date,site,pathIn, pathOutPlot):
    #---------------------------------------------------------------------------------------------------------
    # date : 11 April 2018 (modified version from the previous code (see above))
    # author: Claudia Acquistapace
    # abstract : routine to calculate mean cloud fraction every hour for and mean profile for every six hours of the day for total cloud fraction, ice cloud fraction and liquid cloud fraction. This is done for a specific site and for the whole day. The routine also provides an output dictionary containing the cloud fraction profiles and their standard deviations, for the hourly and six hourly mean and for each phase.
    # input :
    #   - date to process
    #   - site of the measurements
    #   - path to the data
    # output:
    #   - dictionary containing data
    #   - plots of the daily cloud fractions in the 
    #---------------------------------------------------------------------------------------------------------


    
    # -----------read cloudnet classification file, time and height arrays--------------------
    print('load cloudnet target class ncdf file')

    # opening file
    data = Dataset(pathIn+date+'_'+site+'_categorize.nc', mode='r')

    # reading time and height variables
    time = data.variables['time'][:].copy()  # reading time array
    height = data.variables['height'][:].copy()  # reading height array
    cloudnet = data.variables['category_bits'][:].copy()  # reading cloudnet classification

    # closing file
    data.close()




    # building hourly array for cloud fraction calculations
    hourArr = np.array(range(24))

    # generating output matrices
    CFTCloudnet=np.zeros(shape=(24,len(height)))		# matrix for total cloud fraction (ice+liquid)
    CFICloudnet=np.zeros(shape=(24,len(height)))		# matrix for ice cloud fraction 
    CFLCloudnet=np.zeros(shape=(24,len(height)))		# matrix for liquid cloud fraction 

    # building a dataframe for the cloudnet target classification
    CloudnetDF = pd.DataFrame(cloudnet,index=time, columns=height)

    # loop on hour array for calculating cloud fraction
    for itime in range(len(hourArr)-1):
        for iheight in range(len(height)):
            
            # selecting lines corresponding to measurements within the hour
            CloudnetDFArr = CloudnetDF.loc[ (CloudnetDF.index < hourArr[itime+1]) * (CloudnetDF.index >= hourArr[itime]), height[iheight]]
        
            # applying the function to check for ice/liquid cloud presence:returns an array of flags for the hour, where you find liquid/ice clouds at that heigth level
            stringArr=CheckIceLiquidCloudnet(CloudnetDFArr.values)
            
            # calculating cloud fraction in probabilistic way Nsel/Ntot
            Ntot=len(CloudnetDFArr.values)                     # total number of bins in the column
            Nliquid = stringArr.count('liquid')     # number of liquid clouds
            Nice = stringArr.count('ice')           # number of ice clouds
            
            # calculating cloud fractions for the time interval selected at the height j
            if Ntot == 0: 
                CFTCloudnet[itime,iheight]=0.
                CFICloudnet[itime,iheight]=0.
                CFLCloudnet[itime,iheight]=0.
            else:
                CFTCloudnet[itime,iheight]=float(Nice+Nliquid)/float(Ntot)
                # total cloud fraction (liquid+ic/mixed phase clouds)
                CFICloudnet[itime,iheight]=float(Nice)/float(Ntot)          # ice cloud fraction
                CFLCloudnet[itime,iheight]=float(Nliquid)/float(Ntot)            # liquid cloud fraction
                
                
                
    # plotting hourly profiles
    plt.figure('CF_hourly_iceliquid_'+date, figsize=[10,12])
    matplotlib.rc('xtick', labelsize=15)                        # sets dimension of ticks in the plots
    matplotlib.rc('ytick', labelsize=15)                        # sets dimension of ticks in the plots
    nameHourArr=['00','01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
    # defining color array 
    #colorArr = RGB_colors_red2blue(N_colors = 24)[1]
    colorArr=['darkblue', 'blue', 'lightblue','aqua','cyan','green','lime', 'lavender', 'fuchsia', 'magenta', 'plum','red','orange','crimson','chocolate', 'beige','khaki', 'goldenrod', 'yellow', 'pink', 'orchid','purple','grey', 'black']

    
    plt.title('Cloud fraction per hour on the '+date, loc='center')
    # first subplot for liquid clouds
    subfig1 = plt.subplot(121) 
    subfig1.spines["top"].set_visible(False)  
    subfig1.spines["right"].set_visible(False) 
    subfig1.set_title(site+' - '+date+' liquid', fontsize=14)
    #plt.axes()
    plt.xlabel('Cloud fraction [%]', fontsize=14)
    plt.ylabel('Height [m]', fontsize=14)
    subfig1.set_ylim(0., 12000.)
    for i in range(0,24):
        plt.plot(CFLCloudnet[i,:], height, label=nameHourArr[i]+' UTC', color=colorArr[i])
            #plt.errorbar(CFIMeanCloudnet[i,:], height, color=colors_6hours[i], linewidth=2, elinewidth=0.3, label=nameHourArr[i], xerr=CFIStdCloudnet[i,:])
        #plt.legend(loc='center right', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)   

    # second subplot for ice clouds
    subfig2 = plt.subplot(122)
    subfig2.spines["top"].set_visible(False)  
    subfig2.spines["right"].set_visible(False)    
    subfig2.set_title(site+' - '+date+' ice', fontsize=14)
    #plt.axes()
    plt.xlabel(' Cloud fraction [%]', fontsize=14)
    #plt.ylabel('Height [m]', fontsize=20)
    subfig2.set_ylim(0., 12000.)
    for i in range(0,24):
        plt.plot(CFICloudnet[i,:], height, label=nameHourArr[i]+' UTC', color=colorArr[i])
        #plt.errorbar(CFLMeanCloudnet[i,:], height, color=colors_6hours[i], linewidth=2, elinewidth=0.3, label=nameHourArr[i], xerr=CFLStdCloudnet[i,:])
        
    figname=pathOutPlot+'CF_CLOUDNET_HourlyProfiles_'+site+'_'+date+'.pdf'
    plt.savefig(figname) 


    # plot figure of cloud fractions for every hour
    from cf_colors import RGB_colors_red2blue

    # reading colors for color bar plotting
    new_colors = RGB_colors_red2blue(N_colors = len(hourArr))[1] 
    
    
    #---------------calculating mean and standard deviation profiles for hours between 6 and 24
    # defining matrices to store the data
    CFTMeanCloudnet=np.zeros(shape=(len(height)))
    CFIMeanCloudnet=np.zeros(shape=(len(height)))
    CFLMeanCloudnet=np.zeros(shape=(len(height)))
    CFTStdCloudnet=np.zeros(shape=(len(height)))
    CFIStdCloudnet=np.zeros(shape=(len(height)))
    CFLStdCloudnet=np.zeros(shape=(len(height)))


    # calculating mean profiles and standard deviation for each interval of six hours for total cloud fraction
    CFTMeanCloudnet[:]=np.mean(CFTCloudnet[6:24,:], axis=0)        # axis=0 sets the average to be done on the selected columns
    CFTStdCloudnet[:]=np.std(CFTCloudnet[6:24,:], axis=0)        # axis=0 sets the average to be done on the selected columns
 
    # calculating mean profiles for each interval of six hours for ice cloud fraction
    CFIMeanCloudnet[:]=np.mean(CFICloudnet[6:24,:], axis=0)        # axis=0 sets the average to be done on the selected columns
    CFIStdCloudnet[:]=np.std(CFICloudnet[6:24,:], axis=0)        # axis=0 sets the average to be done on the selected columns
 
    # calculating mean profiles for each interval of six hours for ice cloud fraction
    CFLMeanCloudnet[:]=np.mean(CFLCloudnet[6:24,:], axis=0)        # axis=0 sets the average to be done on the selected columns
    CFLStdCloudnet[:]=np.std(CFLCloudnet[6:24,:], axis=0)        # axis=0 sets the average to be done on the selected columns
 

    #-------plot cloud fraction mean over six hours
    # defining color array 
    #colors_24hours = RGB_colors_red2blue(N_colors = 3)[1]

    # opening figure for plot over 6 hours for total cloud fraction (with errorbars)
    fig1 = plt.figure('CF_Mean', figsize=[10,10])
    matplotlib.rc('xtick', labelsize=14) 
    matplotlib.rc('ytick', labelsize=14) 
    plt.suptitle('Cloud fraction profiles ', fontsize=14)
    plt.axes()
    plt.xlabel('Mean Cloud fraction [%]', fontsize=14)
    plt.ylabel('Height [m]', fontsize=14)
    plt.ylim(0., 12000.)
    plt.plot(CFTMeanCloudnet[:], height, color='black', label='ice+liquid',linewidth=1.5)
    plt.plot(CFIMeanCloudnet[:], height, color='red', label='ice',linewidth=1.5)
    plt.plot(CFLMeanCloudnet[:], height, color='blue', label='liquid',linewidth=1.5)
        #plt.errorbar(CFTMeanCloudnet[i,:], height, color=colors_6hours[i], linewidth=2, elinewidth=0.3, label=nameHourArr[i], xerr=CFTStdCloudnet[i,:])
    plt.legend(loc='lower right', fontsize=14)
    figname=pathOutPlot+'CF_Cloudnet_mean6_24_'+site+'_'+date+'.pdf'
    fig1.savefig(figname,fontsize=24)  
    plt.clf()
    plt.cla()
    plt.close()

    
    # defining dictionary containing data to have as output
    dict06_24={}

    # filling dictionaries with data 
    dict06_24={
            date+'CF_tot_06_24':CFTMeanCloudnet[:],
            date+'CF_liquid_06_24':CFLMeanCloudnet[:],
            date+'CF_ice_06_24':CFIMeanCloudnet[:],
            date+'height':height,
    }
    
    
    return (dict06_24)      
    



