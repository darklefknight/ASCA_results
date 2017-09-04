"""Creates for a given time a series of collocated radar and all-sky
   images which can be converted to a movie or animation.

   The data is autmatically gathered and decompressed from /ACPC/.
   The decompression takes place in the RAM or in an provided
   folder.

   USAGE:
   python create_BCO_animation.py -t yyyymmdd

   Options:
     -h, --help            show this help message and exist
     -t yyyymmdd,
                           time you would like to create the
                            animation for

   Further additional data is available and included:
      - Lifting Condensation Level
      - Ceilometer

   Remark: this script should basically work with python3, but
   within the ZMAW network the PIL package is currently only
   installed for python2

   Currently the ceilometer data is not used

   Marcus Klingbiel and Hauke Schulz, March 2016
"""
import os, sys, time, select
import os.path ,glob, shutil
import ASCA_BCO_Test as ASCA
import tarfile, bz2
from scipy.io.netcdf import netcdf_file as Dataset 
from netCDF4 import Dataset as Dataset2
import tempfile
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from matplotlib.ticker import AutoMinorLocator,MultipleLocator
from matplotlib import gridspec
import matplotlib.image as mpimg
from matplotlib import rcParams

####################     Parser       ##################################
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-t", "--time", dest="yyyymmdd", type=str,
     help="time of interest in the format yyyymmdd")

options = parser.parse_args()
########################################################################

def uncompressTGZ(tar_url, extract_path='.'):
    tar = tarfile.open(tar_url, 'r')
    names = []
    for item in tar:
        tar.extract(item, extract_path)
        if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
            extract(item.name, "./" + item.name[:item.name.rfind('/')])


# CONFIGURATION
MBRData         = '/data/mpi/mpiaes/obs/ACPC/MBR2/mom/'               #daily folders with yyyymmdd_0000.mmclx.bz2 files
CEILOData       = '/data/mpi/mpiaes/obs/ACPC/Ceilometer/CHM140102/'   #year/month folders with yyyymmdd_BCO_CHM140102.nc files
GroundData      = '/data/mpi/mpiaes/obs/ACPC/Weather/'                #yearmonth folders with Meteorology__Deebles_Point__2m_10s__yyyymmdd.nc.bz2
AllSkyData      = '/data/share/mpiaes/obs/ACPC/allsky/'
RadData         = '/data/share/mpiaes/obs/ACPC/Radiation/'

outputFiles     = './'

filename_website = True         # True if the fileoutput name shoud be a continus number
                                # False if the filename should contain time information
min_resolution  = 2             # Image frequency in minutes. Every x minute a picture should be created.

yyyymmdd        = options.yyyymmdd
yyyy            = yyyymmdd[0:4]
mm              = yyyymmdd[4:6]
dd              = yyyymmdd[6:8]

Filter          = -62
StartTime       = 10             #Alle Angaben in Stunden
EndTime         = 22

StartAlt        = 0              #Angabe in Meter
EndAlt          = 2000

LiftingCondensationLevel = False

# Create filenames
#filename       = BASE FOLDER  + SUBFOLDER1          +  /  + SUBFOLDER2 +  /  + SUBFOLDER3 + / + FILENAME
nc_file         = MBRData      + yyyymmdd            + '/' +                                     yyyymmdd+'_0000.mmclx.bz2'
Save_Images     = outputFiles        
Ceil_nc_file    = CEILOData    + yyyy                + '/' + mm         + '/' +                  yyyymmdd+'_BCO_CHM140102.nc'
if LiftingCondensationLevel:
   Weather_nc_file = GroundData   + yyyy+mm             + '/' +                                     'Meteorology__Deebles_Point__2m_10s__'+yyyymmdd+'.nc.bz2'
AllSky_file     = AllSkyData   + 'cc'+yyyy[2:]+mm    + '/' + dd         + '/' +                  yyyy[2:]+mm+dd+'.tgz'
Rad_file        = RadData      + yyyy+mm             + '/' +                                     'Radiation__Deebles_Point__DownwellingRadiation__*s__'+yyyymmdd+'.nc.bz2'

#---------------------------------------------

StartTime_Raw = StartTime
EndTime_Raw   = EndTime

ymin = StartAlt
ymax = EndAlt
	
#--------Read NETCDF variables------------
print('Load NetCDF Data')

if os.path.isfile(nc_file):
   uncomp_MBR2Data = bz2.BZ2File(nc_file)

   with Dataset(uncomp_MBR2Data, 'r') as f:
      Z     = f.variables['Z'][:]            # in mm6 m-3
      Time  = f.variables['time'][:]
      Range = f.variables['range'][:]
      Velo  = f.variables['VELg'][:]
      secs  = mdate.epoch2num(Time)

# if os.path.isfile(Ceil_nc_file):
#    with Dataset(Ceil_nc_file, 'r') as f:
#       Ceil_cbh  = f.variables['cbh'][:].copy()      # in mm6 m-3
#       Ceil_Time = f.variables['time'][:].copy()
#       for i in range(0,len(Ceil_Time)):
#          Ceil_Time[i]=Ceil_Time[i]-2082844800 

#    Ceil_secs    = mdate.epoch2num(Ceil_Time)
if LiftingCondensationLevel: 
   if os.path.isfile(Weather_nc_file):
      uncomp_GroundData = bz2.BZ2File(Weather_nc_file)
      with Dataset(uncomp_GroundData, 'r') as f:
         Weather_rh   = f.variables['RH'][:]       # in mm6 m-3
         Weather_temp = f.variables['T'][:]        # in mm6 m-3
         Weather_Time = f.variables['time'][:]
         Weather_secs = mdate.epoch2num(Weather_Time)

if os.path.isfile(AllSky_file):
   #Create temp. folder
   tmp_AllSky_folder = tempfile.mkdtemp()
   uncompressTGZ(AllSky_file,tmp_AllSky_folder)
try:
   Rad_file = glob.glob(Rad_file)[0] #evaluate wildcards in filename
   if os.path.isfile(Rad_file):
      # Create temporary file
      tmp_rad_file = tempfile.NamedTemporaryFile(delete=False)
      # Decompress and save to temp. file
      with open(Rad_file,'rb') as fbz2:
         tmp_rad_file.write(bz2.decompress(fbz2.read()))
      tmp_rad_file.close()
      #Read data
      with Dataset2(tmp_rad_file.name,'r') as f:
         SWdown_global = f.variables['SWdown_global'][:]
         Rad_Time      = f.variables['time'][:]
         Rad_secs     = mdate.epoch2num(Rad_Time)
      #Delete temporary file
      tmp_rad_file.delete = True
      tmp_rad_file.unlink(tmp_rad_file.name)
except IndexError:
   pass

#--------Umrechnung von Z in mm6 m-3 nach dBZ-------------
print('Calculate dBZ')
try:
   Z_dbz = np.empty(np.shape(Z))  # 2D-Array in der Groesse von Z erstellen
   Z_dbz = 10*np.log10(Z)
   Z_dbz = np.where(ma.greater_equal(Z_dbz,30), 30, Z_dbz)
   Z_dbz = np.where(ma.less_equal(Z_dbz,Filter),np.nan,Z_dbz)
except:
   #Remove temp. folders
   if os.path.isfile(AllSky_file):
      try:
         shutil.rmtree(tmp_AllSky_folder)
      except:
         print("Temporary folder {0} could not found or deleted".format(tmp_AllSky_folder))
   print('No animation created. Radar data missing')
   sys.exit()


#----------Calculate Lifting Condensation Level-----------------------------
try:
   LCL = np.empty(np.shape(Weather_temp))
   for i in range(0,len(Weather_Time)):
      LCL[i]=(20+(Weather_temp[i]/5))*(100 - Weather_rh[i])     #LCL in meter
except:
   LCL = None

#--------Plot Figure------------
print('Plot Figure')
minu='00'
hour="{0:0>2}".format(StartTime_Raw)
i=1

date_fmt = '%H:%M'
date_formatter = mdate.DateFormatter(date_fmt)

StartTime =(len(Time)/24*StartTime)
EndTime   =(len(Time)/24*EndTime)
StartAlt  =(float(len(Range)))/15000*StartAlt
EndAlt    =(float(len(Range)))/15000*EndAlt


#------Get ASCA values-----------

Cloudiness = ASCA.cloudiness("asdfas")


while i <= np.int(np.floor((EndTime_Raw-StartTime_Raw)*60/min_resolution)):      #841

   
   ##########################################################################################
   ##                        CREATE STATIC PLOT                                            ##
   ## Things that do not need to be recalculated again, like radar image, should be static ##
   ##########################################################################################
   if i == 1: #First plot (here everything needs to be created)
      #### LAYOUT
      fig = plt.figure(figsize=[20,9],dpi=60)
      gs1 = gridspec.GridSpec(8, 7)
      gs1.update(left=0.05, right=0.99, wspace=0.1)
      rcParams['legend.loc'] = 'best'
   
      ax1 = fig.add_subplot(gs1[0:4,0:4])
      ax2 = fig.add_subplot(gs1[0:6,4:7])
      ax3 = fig.add_subplot(gs1[4:7,0:4],sharex=ax1)
      ax4 = fig.add_subplot(gs1[7:8,0:4],sharex=ax1)
      
      cbaxes = fig.add_axes([0.592,0.06,0.4,0.47], frameon=None) #[0.07,0.75,0.4,0.47]
      
      plt.xticks(rotation=70)
      
      ax1.grid()
      
      ax1.xaxis_date()
      ax1.xaxis.set_major_formatter(date_formatter)
      ax1.xaxis.set_minor_locator(AutoMinorLocator(15))
      ax1.yaxis.set_ticks(np.arange(0,17000,200))
      ax1.tick_params('both', length=3, width=1, which='minor')
      ax1.tick_params('both', length=4, width=2, which='major')
      ax1.get_yaxis().set_tick_params(which='both', direction='out')
      ax1.get_xaxis().set_tick_params(which='both', direction='out')
      ax1.get_xaxis().set_visible(False)
      ax2.get_xaxis().set_visible(False)
      ax2.get_yaxis().set_visible(False)
      ax3.get_xaxis().set_visible(False)      
      ax4.yaxis.set_ticks(np.arange(0,101,50))
      ax4.get_xaxis().set_visible(False) 
      ax4.set_ylim(0,100)
      
      ax1.set_ylabel('Altitude in m',fontsize=16)
      ax3.set_ylabel(r'W/m$^2$',fontsize=16)
      ax4.set_xlabel('UTC Time',fontsize=16)
      ax4.set_ylabel('Cloud cover',fontsize=16)
      
      cbaxes.get_xaxis().set_visible(False)
      cbaxes.get_yaxis().set_visible(False)
         
      # ax1.set_xticks(np.arange(min(secs), max(secs),0.0208335/2)) #Abstand xtiks = 15min
      ax1.set_xticks(np.arange(min(secs), max(secs),0.083333)) #Abstand xticks = 2h, 1 Minute = 0.000694
      #ax1.set_xticks(np.arange(min(secs), max(secs),0.041667))   #Abstand xticks = 1h
      ax1.set_xlim(min(secs)+StartTime_Raw*60*0.000694, min(secs)+EndTime_Raw*60*0.000694)
      ax1.set_ylim([ymin,ymax])
      contour_levels = np.arange(-75, 30, 1) 

      plt.suptitle('Deebless Point, Barbados, MBR2, '+dd+'.'+mm+'.'+yyyy+ ', Filter='+str(Filter)+'dBZ', fontsize=20)

      try:
         print('Read image')
         img=mpimg.imread(glob.glob(tmp_AllSky_folder+'/cc'+yyyy[2:]+mm+dd+str(hour)+str(minu)+'*.jpg')[0])
         im2   = ax2.imshow(img,extent=[0,300,0,300], aspect='auto', interpolation='none')
      except:
         print("No image to read")
         pass
      print('Do some plotting')
      ### ACTUAL PLOTTING
      im1   = ax1.contourf(secs[StartTime:EndTime], Range[StartAlt:EndAlt],\
                        Z_dbz[StartTime:EndTime,StartAlt:EndAlt].transpose(), contour_levels)
      cb    = plt.colorbar(im1, ax = cbaxes, ticks=[-70,-60,-50,-40,-30,-20,-10,0,10,20,30], shrink=1, orientation='horizontal')
      

      #im3  = ax1.bar(Ceil_secs[:],Ceil_cbh[:,0], width=0.000694/4, color='gray',alpha=0.4, label='Ceilometer Cloud base height')
      if LCL is not None:
         im4   = ax1.plot(Weather_secs[:],LCL[:], color='red', label='Lifting Condensation Level')
         ax1.legend()   #Legend erzeugen bbox_to_anchor=(0.6, 0.95), loc=2, borderaxespad=0.
      try:
         im5   = ax3.plot(Rad_secs[:],SWdown_global[:],color='red',label='Downwelling SW radiation (global)')
      except:
         print("ax3 had a problem") 
         pass
      
      try:
         im6 = ax4.plot(Cloudiness[:], color='red',label='Cloud cover in %')
      except:
         print("ax4 had a problem") 
         pass
   
      SecondsVertLine=min(secs)+(int(hour)*60+int(minu))*0.000694
      ax1.axvline(SecondsVertLine, color='gray', linewidth=1.5)
      ax3.axvline(SecondsVertLine, color='gray', linewidth=1.5)
      ax4.axvline(SecondsVertLine, color='gray', linewidth=1.5)    
      
      cb.set_clim(-70, 30)
      cb.set_label('Radar Reflectivity Factor dBZ',fontsize=16)

   else:
   ##########################################################################################
   ##                        UPDATING STATIC PLOT                                          ##
   ##      Only replot things that changed from one timestep to the other                  ##
   ##########################################################################################
      try:
         print('Read image')
         img=mpimg.imread(glob.glob(tmp_AllSky_folder+'/cc'+yyyy[2:]+mm+dd+str(hour)+str(minu)+'*.jpg')[0])
         im2.set_data(img)
      except:
         print("No image to read")
         pass
      SecondsVertLine=min(secs)+(int(hour)*60+int(minu))*0.000694
      if LCL is not None:
         del ax1.lines[1] #0 is LCL line
      else:
         del ax1.lines[0]
      ax1.axvline(SecondsVertLine, color='gray', linewidth=1.5)
      try:
         del ax3.lines[1] #0 is the actual radiation curve
         ax3.axvline(SecondsVertLine, color='gray', linewidth=1.5)
      except:
         pass
      try:
         del ax4.lines[1]
         ax4.axvline(SecondsVertLine, color='gray', linewidth=1.5)
      except:
         pass

      print("Draw canvas")
      fig.canvas.draw()

   print("Save image")
   if filename_website:
      plt.savefig(Save_Images+'Mv_MBR2_'+"{0:0>4}".format(i)+'.png',dpi=70)
   else:
      plt.savefig(Save_Images+'Mv_MBR2_'+hour+minu+'.png',dpi=70)
   print("Saved")


   minu_n=int(minu)
   hour_n=int(hour)

   minu_n=minu_n+min_resolution
   if minu_n==60:
      minu_n=0
      hour_n=hour_n+1

   if minu_n<10:
      minu='0'+str(minu_n)
   else:
      minu=str(minu_n)

   if hour_n<10:
      hour='0'+str(hour_n)
   else:
      hour=str(hour_n)

   print str(hour)+':'+str(minu)
   i+=1
plt.close()
#Remove temp. folders
if os.path.isfile(AllSky_file):
   try:
      shutil.rmtree(tmp_AllSky_folder)
   except:
      print("Temporary folder {0} could not found or deleted".format(tmp_AllSky_folder))

print('---------------------------------')
print('Done :-)')
print('---------------------------------')
