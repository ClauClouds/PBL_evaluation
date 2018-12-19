# function to plot wind rose from an input dictionary of data
#---------------------------------------------------------------------------------
# date :  18.12.2018
# author: Claudia Acquistapace (cacquist@meteo.uni-koeln.de), Davide Ori
# goal: plot wind rose from an input dictionary of data
# input: Dictionary of the form:
#dict_windRose = {'windDir':wd_ICON_INSCAPE, 'windSpeed':ws_ICON_INSCAPE, 'maxSpeed':15, \
#                'minSpeed':0., 'Speed_res':3, 'NstepsSpeed':5,  '%min':0, '%max':30, '%step':5, \
#                'title':'wind rose - ICON-INSCAPE', 'modelName':'ICON-INSCAPE', 'titleFont':16, \
#                'legendPosition':'lower left', 'legendFont':14, 'date':'20130502', 'outFile':pathFig+'WindRose_ICON_INSCAPE_20130502.png'}
# meaning of the items of the dictionary
# - 'windDir': wind direction array or list
# - 'windSpeed': wind speed array or list
# - 'maxSpeed': maximum value of wind speed to be visualized in the legend
# - 'minSpeed': minimum value of wind speed to be visualized in the legend
# - 'Speed_res': resolution for the wind speed colors of the legend 
# - 'NstepsSpeed': number of colors to be used. It has to be an integer resulting from the division of 'maxSpeed'/'Speed_res',
# so for example, 15/3 = 5
# - '%min': minimum percentage of occurrence of wind values in the rose
# - '%max': maximum percentage of occurrence of wind values in the rose
# - '%step': resolution step for percentage in concentric circles
# - 'title': title of the plot
# - 'model/Obs Name': name of the dataset ( model or obs)
# - 'titleFont' : font size of the title
# - 'legendPosition': position of the legend
# - 'legendFont': font size of the legend
# - 'date': date of the dataset
# - 'outFile': output filename (including path)
# output:
# - png plot in the folder indicated by outFile
#--------------------------------------------------------------------------------
def f_plotWindRose(Dict):
    
    bins_range = np.arange(Dict['minSpeed'],Dict['maxSpeed'],Dict['Speed_res']) # this sets the legendscale
    ax = WindroseAxes.from_ax()
    bars = ax.bar(Dict['windDir'], Dict['windSpeed'], normed=True, bins=bins_range)
    ax.set_yticks(np.arange(Dict['%min'], Dict['%max'], step=Dict['%step']))
    ax.set_yticklabels(np.arange(Dict['%min'], Dict['%max'], step=Dict['%step']))
    ax.set_title(Dict['title'], fontsize=Dict['titleFont'])
    L=ax.legend(loc=Dict['legendPosition'], fontsize=Dict['legendFont'])
    for ind in range(Dict['NstepsSpeed']):
        L.get_texts()[ind].set_text(L.get_texts()[ind].get_text()+' m/s')
            #L.get_texts()[1].set_text(L.get_texts()[1].get_text()+' m/s')
            #L.get_texts()[2].set_text(L.get_texts()[2].get_text()+' m/s')
            #L.get_texts()[3].set_text(L.get_texts()[3].get_text()+' m/s')
            #L.get_texts()[4].set_text(L.get_texts()[4].get_text()+' m/s')
        #plt.show()
       #fig.plt()
    plt.savefig(Dict['outFile'])