"""
Script for calculating the cloudcoverage in percent and saving it to netCDF.

USAGE:
    ASCA_nc.py -t YYYYMMDD
   
What to modify:    
    - in the "Settings" section of the function "cloudiness" diverse modifications can be made
    - in the main-programm (at the very bottom of this script) is a "Set Parameter" section
    
Author: Tobias Machnitzki (tobias.machnitzki@mpimet.mpg.de)

"""
import argparse
import sys
import tempfile
import shutil
import os
from joblib import Parallel, delayed
from PIL import Image, ImageDraw, ImageOps, ImageFont
import math
from pvlib.location import Location
import matplotlib.dates as mdate
import pvlib
import pandas as pd
import datetime
import glob
import numpy as np
from time import clock
import copy
from Converter import convert
import tarfile
from netCDF4 import Dataset
import time
import datetime
import numpy as np


def cloudiness(InputFile,pos):
    # Information:
    #
    # Code written by Marcus Klingebiel, Max-Planck-Institute for Meteorology
    # E-Mail: marcus.klingebiel@mpimet.mpg.de
    #
    # PLEASE ASK BEFORE SHARING THIS CODE!
    #
    #
    # Preliminary version of the All-Sky Cloud Algorithms (ASCA)
    # The code is based on the analysis of every single pixel on a jpeg-Image.
    # The used Ski-Index and Brightness-Index base on Letu et al. (2014), Applied Optics, Vol. 53, No. 31.
    #
    #
    #
    # Marcus Klingebiel, March 2016

    # Code eddited by Tobias Machnitzki
    # Email: tobias-machnitzki@web.de


    # --------------------Settings------------------------------------------------------------------------------------------

    debugger = False  # if true, the program will print a message after each step

    TXTFile = False  # if True, the program will generate a csv file with several information. Delimiter = ','

    #    imagefont_size = 20 #Sets the font size of everything written into the picture

    Radius_synop = False  # If True: not the whole sky will be used, but just the 60 degrees from the middle on (like the DWD does with cloud covering)

    Save_image = False  # If True: an image will be printed at output-location, where recognized clouds are collored.

    #    font = ImageFont.truetype("/home/tobias/anaconda3/lib/python3.5/site-packages/matplotlib/mpl-data/fonts/ttf/Vera.ttf",imagefont_size)    # Font

    set_scale_factor = 100  # this factor sets the acuracy of the program. By scaling down the image size the program gets faster but also its acuracy dercreases.
    # It needs to be between 1 and 100. If set 100, then the original size of the image will be used.
    # If set to 50 the image will be scaled down to half its size
    #


    # ---------------------Calcutlate the SI-parameter--------------------------------------------------------------------------------------
    # The Parameter gets calculated before the loop over all filse start, to save computing time.
    # To see how the function for the parameter was generated, see the documentation.

    size = 100
    sza = float(pos.zenith[0])
    parameter = np.zeros(size)
    for j in range(size):
        parameter[j] = (0 + j * 0.4424283716980435 - pow(j, 2) * 0.06676211439554262 + pow(j,
                                                                                           3) * 0.0026358061791573453 - pow(
            j, 4) * 0.000029417130873311177 + pow(j, 5) * 1.0292852149593944e-7) * 0.001

        # ----------------------Read files------------------------------------------------------------------------------------------------------

    OutputPath = "/media/MPI/ASCA/images/s160521/out/"

    cloudiness_value = []
    ASCAtime = []
    cloudmasks = []
    #    print(InputFilePath)



    # ---------------------------------------------------------------------------------------------------------
    # --------Get day and time------------
    if debugger == True:
        print("Getting day and time")

    date_str = InputFile[len(InputFile) - 19:len(InputFile) - 19 + 12]
    if debugger == True:
        print("Date_Str: " + date_str)
    Year_str = date_str[0:2]
    Month_str = date_str[2:4]
    Day_str = date_str[4:6]
    Hour_str = date_str[6:8]
    Minute_str = date_str[8:10]
    Second_str = date_str[10:12]

    Year = int(date_str[0:2])
    Month = int(date_str[2:4])
    Day = int(date_str[4:6])
    Hour = int(date_str[6:8])
    Minute = int(date_str[8:10])
    Second = int(date_str[10:12])

    print("sza=" + str(sza))

    time1 = clock()
    azimuth = float(pos.azimuth[0])
    sza_orig = sza
    azi_orig = azimuth
    azimuth = azimuth + 190  # 197 good
    #            print(( str(sza) + '   '+Hour_str+':'+Minute_str))
    if azimuth > 360:
        azimuth = azimuth - 360

        # ------------Open csv-File-------------------------------------------------------------------------------------------------------------
    if debugger == True:
        print("Open csv-File")

    if TXTFile == True:
        f = open(
            OutputPath + Year_str + Month_str + Day_str + '_' + Hour_str + Minute_str + Second_str + '_ASCA.csv',
            'w')
        f.write(
            'Seconds_since_1970, UTC_Time, SZA_in_degree, Azimuth_in_degree, Cloudiness_in_percent, Cloudiness_in_oktas' + '\n')
        TXTFile = False

        # ---Read image and set some parameters-------------------------------------------------------------------------------------------------
    if debugger == True:
        print("Reading image and setting parameters")

    # ------------rescale picture-------------------------------------------
    image = Image.open(InputFile)

    x_size_raw = image.size[0]
    y_size_raw = image.size[1]
    scale_factor = (set_scale_factor / 100.)
    NEW_SIZE = (x_size_raw * scale_factor, y_size_raw * scale_factor)
    image.thumbnail(NEW_SIZE, Image.ANTIALIAS)

    image = ImageOps.mirror(image)  # Mirror picture

    x_size = image.size[0]
    y_size = image.size[1]
    x_mittel = x_size / 2  # Detect center of the true image
    y_mittel = y_size / 2
    Radius = 900  # pixel    #  Set area for the true allsky image

    scale = x_size / 2592.

    # -------------convert image to an array and remove unnecessary part araund true allsky image-----------------------------------------------------------------
    if debugger == True:
        print("Drawing circle around image and removing the rest")

    r = Radius * scale
    y, x = np.ogrid[-y_mittel:y_size - y_mittel, -x_mittel:x_size - x_mittel]
    x = x + (15 * scale)  # move centerpoint manually
    y = y - (40 * scale)
    mask = x ** 2 + y ** 2 <= r ** 2  # make a circular boolean array which is false in the area outside the true allsky image

    image_array = np.asarray(image,
                             order='F')  # converting the image to an array with array[x,y,color]; color: 0=red, 1,green, 2=blue
    image_array.setflags(write=True)  # making it able to work with that array and change it
    image_array[:, :, :][~mask] = [0, 0, 0]  # using the mask created before on that new made array

    if Radius_synop == True:
        mask = x ** 2 + y ** 2 <= (765 * scale) ** 2
        image_array[:, :, :][~mask] = [0, 0, 0]

    del x, y
    #
    # ------------Calculate position of sun on picture---------------------------------------------------------------------------------------
    if debugger == True:
        print("Calculating position of the sun on picture")

    sza = sza - 90
    if sza < 0:
        sza = sza * (-1)

    AzimutWinkel = ((2 * math.pi) / 360) * (azimuth - 90)
    sza = ((2 * math.pi) / 360) * sza
    x_sol_cen = x_mittel - (15 * scale)
    y_sol_cen = y_mittel + (40 * scale)
    RadiusBild = r
    sza_dist = RadiusBild * math.cos(sza)

    x = x_sol_cen - sza_dist * math.cos(AzimutWinkel)
    y = y_sol_cen - sza_dist * math.sin(AzimutWinkel)

    ###-----------Draw circle around position of sun-------------------------------------------------------------------------------------------
    if debugger == True:
        print("Drawing circle around position of sun")

    x_sol_cen = int(x)
    y_sol_cen = int(y)
    Radius_sol = 300 * scale
    Radius_sol_center = 250 * scale

    y, x = np.ogrid[-y_sol_cen:y_size - y_sol_cen, -x_sol_cen:x_size - x_sol_cen]
    sol_mask = x ** 2 + y ** 2 <= Radius_sol ** 2
    sol_mask_cen = x ** 2 + y ** 2 <= Radius_sol_center ** 2
    sol_mask_cen1 = sol_mask_cen
    image_array[:, :, :][sol_mask_cen] = [0, 0, 0]
    #        image_array[:,:,:][]

    ##-------Calculate Sky Index SI and Brightness Index BI------------Based on Letu et al. (2014)-------------------------------------------------
    if debugger == True:
        print("Calculating Sky Index SI and Brightness Index BI")

    image_array_f = image_array.astype(float)

    SI = ((image_array_f[:, :, 2]) - (image_array_f[:, :, 0])) / (
        ((image_array_f[:, :, 2]) + (image_array_f[:, :, 0])))
    where_are_NaNs = np.isnan(SI)
    SI[where_are_NaNs] = 1

    mask_sol1 = SI < 0.1
    Radius = 990 * scale
    sol_mask_double = x ** 2 + y ** 2 <= Radius ** 2
    mask_sol1 = np.logical_and(mask_sol1, ~sol_mask_double)
    image_array[:, :, :][mask_sol1] = [255, 0, 0]

    ###-------------Include area around the sun----------------------------------------------------------------------------------------------------
    if debugger == True:
        print("Including area around the sun")

    y, x = np.ogrid[-y_sol_cen:y_size - y_sol_cen, -x_sol_cen:x_size - x_sol_cen]
    sol_mask = x ** 2 + y ** 2 <= Radius_sol ** 2
    sol_mask_cen = x ** 2 + y ** 2 <= Radius_sol_center ** 2
    sol_mask_cen = np.logical_and(sol_mask_cen, sol_mask)

    Radius_sol = size * 100 * 2
    sol_mask = x ** 2 + y ** 2 <= Radius_sol ** 2
    mask2 = np.logical_and(~sol_mask_cen, sol_mask)

    image_array_c = copy.deepcopy(
        image_array)  # duplicating array: one for counting one for printing a colored image

    time3 = clock()

    for j in range(size):
        Radius_sol = j * 10 * scale
        sol_mask = (x * x) + (y * y) <= Radius_sol * Radius_sol
        mask2 = np.logical_and(~sol_mask_cen, sol_mask)
        sol_mask_cen = np.logical_or(sol_mask, sol_mask_cen)

        mask3 = SI < parameter[j]
        mask3 = np.logical_and(mask2, mask3)
        image_array_c[mask3] = [255, 0, 0]
        image_array[mask3] = [255, 300 - 3 * j, 0]

    time4 = clock()
    #        print 'Schleifenzeit:', time4-time3
    ##---------Count red pixel(clouds) and blue-green pixel(sky)-------------------------------------------------------------------------------------------
    if debugger == True:
        print("Counting red pixel for sky and blue for clouds")

    c_mask = np.logical_and(~sol_mask_cen1, mask)
    c_array = (
        image_array_c[:, :, 0] + image_array_c[:, :, 1] + image_array_c[:, :, 2])  # array just for the counting
    Count1 = np.shape(np.where(c_array == 255))[1]
    Count2 = np.shape(np.where(c_mask == True))[1]

    CloudinessPercent = (100 / float(Count2) * float(Count1))
    CloudinessSynop = int(round(8 * (float(Count1) / float(Count2))))

    image = Image.fromarray(image_array.astype(np.uint8))

    # ----------Mirror Image-----------------------------
    image = ImageOps.mirror(image)  # Mirror Image back
    # ---------Add Text-----------------------------------
    if debugger == True:
        print("Adding text")

    sza = "{:5.1f}".format(sza_orig)
    azimuth = "{:5.1f}".format(azi_orig)
    CloudinessPercent = "{:5.1f}".format(CloudinessPercent)

    #            draw = ImageDraw.Draw(image)
    #            draw.text((20*scale, 20*scale),"BCO All-Sky Camera",(255,255,255),font=font)
    #            draw.text((20*scale, 200*scale),Hour_str+":"+Minute_str+' UTC',(255,255,255),font=font)
    #
    #            draw.text((20*scale, 1700*scale),"SZA = "+str(sza)+u'\u00B0',(255,255,255),font=font)
    #            draw.text((20*scale, 1820*scale),"Azimuth = "+str(azimuth)+u'\u00B0',(255,255,255),font=font)
    #
    #            draw.text((1940*scale, 1700*scale),"Cloudiness: ",(255,255,255),font=font)
    #            draw.text((1930*scale, 1820*scale),str(CloudinessPercent)+'%   '+ str(CloudinessSynop)+'/8',(255,255,255),font=font)
    #
    #            draw.text((1990*scale, 20*scale),Day_str+'.'+Month_str+'.20'+Year_str,(255,255,255),font=font)

    # -------------Save values to csv-File---------------------------------------
    #            if debugger == True:
    #                print "Saving values to csv-File"

    #            EpochTime=(datetime.datetime(2000+Year,Month,Day,Hour,Minute,Second) - datetime.datetime(1970,1,1)).total_seconds()
    #            f.write(str(EpochTime)+', '+Hour_str+':'+Minute_str+', '+str(sza)+', '+str(azimuth)+', '+str(CloudinessPercent)+', '+str(CloudinessSynop)+'\n')
    # -------------Save picture--------------------------------------------------
    if Save_image == True:
        if debugger == True:
            print("saving picture")

        image = convert(InputFile, image, OutputPath)
        image.save(
            OutputPath + Year_str + Month_str + Day_str + '_' + Hour_str + Minute_str + Second_str + '_ASCA.jpg')

    # image.show()
    time2 = clock()
    time = time2 - time1
    cloudiness_value.append(CloudinessPercent)
    ASCAtime = ((datetime.datetime(Year + 2000, Month, Day, Hour, Minute, Second)))

    cloudmask = [c_array == 255]
    cloudmask = cloudmask[0] * 1
    cloudmask[np.where(c_mask == False)] = -1
    clodmask = np.fliplr(cloudmask)
    cloudmasks.append(cloudmask)

    #               print "Berechnungszeit: ", time
    return (CloudinessPercent, ASCAtime, cloudmask, set_scale_factor)


def uncompressTGZ(tar_url, extract_path='.'):
    tar = tarfile.open(tar_url, 'r')
    counter = 0
    for item in tar:
        counter += 1
        tar.extract(item, extract_path)
        if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
            tar.extract(item.name, "./" + item.name[:item.name.rfind('/')])
    print(counter, " Elemente enpackt.")


# %%
def create_netCDF(nc_name, path_name, cc, asca_time, scale_factor, cloudmask=None):
    MISSING_VALUE = 999
    epoche_time_start = datetime.datetime(1970, 1, 1)
    bco_time_start = datetime.datetime(2010, 4, 1)

    # converting asca_time to epoche-time and bco-time
    bco_time = []
    epoche_time = []
    time_str = []
    for element in asca_time:
        bco_time.append((element - bco_time_start).total_seconds())
        epoche_time.append((element - epoche_time_start).total_seconds())
        time_str.append(int(element.strftime("%Y%m%d%H%M%S")))

    time_str = np.asarray(time_str, dtype="S14")
    bco_time = np.asarray(bco_time, dtype="f8")
    epoche_time = np.asarray(epoche_time, dtype="f8")

    nc = Dataset(path_name + nc_name, mode='w', format='NETCDF4')

    # Create global attributes
    nc.location = "The Barbados Cloud Observatory, Deebles Point, Barbados"
    nc.instrument = "Allsky-Cloud-Imager (Resolution: 2048x1536)"
    nc.downsaling = str(scale_factor) + "% used of Resolution for calculations"
    nc.converted_by = "Tobias Machnitzki (tobias.machnitzki@mpimet.mpg.de)"
    nc.institution = "Max Planck Institute for Meteorology, Hamburg"
    nc.created_with = os.path.basename(__file__) + " with its last modification on " + time.ctime(
        os.path.getmtime(os.path.realpath(__file__)))
    nc.creation_date = time.asctime()
    nc.version = "1.0.0"

    # Create dimensions
    time_dim = nc.createDimension('time', len(asca_time))

    if cloudmask != None:
        y_size, x_size = np.shape(cloudmask[0])
        x_dim = nc.createDimension('x-pixel', x_size)
        y_dim = nc.createDimension('y-pixel', y_size)

    # Create variables
    time_var = nc.createVariable('time', 'f8', ('time',), fill_value=MISSING_VALUE, zlib=True)
    time_var.units = "Seconds since 1970-1-1 0:00:00 UTC"
    time_var._CoordinateAxisType = "Time"
    time_var.calendar = "Standard"

    # YYYYMMDDhhmmss_var = nc.createVariable('YYYYMMDDhhmmss', 'S14', ('time',), fill_value=MISSING_VALUE, zlib=True)
    # YYYYMMDDhhmmss_var.long_name = "Year_Month_Day_Hour_Minute_Second"
    # YYYYMMDDhhmmss_var.units = "UTC"

    bco_day_var = nc.createVariable('bco_day', 'f8', ('time',), fill_value=MISSING_VALUE, zlib=True)
    bco_day_var.long_name = "Days since start of Barbados Cloud Observatory measurements"
    bco_day_var.units = "Days since 2010-4-1 0:00:00 UTC"

    cc_var = nc.createVariable('cc', 'f4', ('time'), fill_value=MISSING_VALUE, zlib=True)
    cc_var.long_name = "Cloudcoverage"
    cc_var.units = "Percent"

    if cloudmask != None:
        cloudmask_var = nc.createVariable('cloudmask', 'i1', ('time', 'y-pixel', 'x-pixel'))
        cloudmask_var.long_name = "Boolean mask for where Clouds were detected. 1=cloud, 0=clearsky, -1=out_of_picture"

    # Fill with vlaues
    # YYYYMMDDhhmmss_var[:] = time_str[:]
    bco_day_var[:] = bco_time[:]
    time_var[:] = epoche_time[:]
    cc_var[:] = cc[:]

    if cloudmask != None:
        cloudmask_var[:] = cloudmask[:]

    # Close netCDF-file
    nc.close()


def calc_sza(InputFile):
    date_str = InputFile[len(InputFile) - 19:len(InputFile) - 19 + 12]
    Year_str = date_str[0:2]
    Month_str = date_str[2:4]
    Day_str = date_str[4:6]
    Hour_str = date_str[6:8]
    Minute_str = date_str[8:10]
    Second_str = date_str[10:12]

    Year = int(date_str[0:2])
    Month = int(date_str[2:4])
    Day = int(date_str[4:6])
    Hour = int(date_str[6:8])
    Minute = int(date_str[8:10])
    Second = int(date_str[10:12])

    tus = Location(13.164, -59.433, 'UTC', 70,
                   'BCO')  # This is the location of the Cloud camera used for calculating the Position of the sun in the picture
    times = pd.date_range(start=datetime.datetime(Year + 2000, Month, Day, Hour, Minute, Second),
                          end=datetime.datetime(Year + 2000, Month, Day, Hour, Minute, Second), freq='10s')
    times_loc = times.tz_localize(tus.pytz)
    pos = pvlib.solarposition.get_solarposition(times_loc, tus.latitude, tus.longitude, method='nrel_numpy',
                                                pressure=101325, temperature=25)
    sza = float(pos.zenith[0])
    if sza <= 85:
        return (InputFile,pos)
    else:
        return "xxx"


# %%
if __name__ == "__main__":

    # =======================================
    #   Set Parameter:
    # =======================================
    save_cc_mask = False  # If True: the cloudmask will be written into the netCDF file (Will result in large netCDF-files)
    ncPath = "./"  # Path were the netCDF-file will be written to
    # =======================================
    # =======================================


    # --------Argparse:------------
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--time", dest="yyyymmdd", type=str,
                        help="time of interest in the format yyyymmdd")

    options = parser.parse_args()

    yyyymmdd = options.yyyymmdd
    date_str = yyyymmdd[2:8]
    year_str = yyyymmdd[0:4]
    month_str = yyyymmdd[4:6]
    day_str = yyyymmdd[6:8]

    # ------build filepath to packed files:----------
    tar_path = "/data/mpi/mpiaes/obs/ACPC/allsky/m" + year_str[
                                                      2:4] + month_str + "/" + year_str + month_str + day_str + "/"
    tar_name = "m" + year_str[2:4] + month_str + day_str + ".tgz"
    if not tar_name in os.listdir(tar_path):
        print(("File " + tar_name + " not found in \n" + tar_path))
        sys.exit('File not found.')

    # ----make temp-folder for extracting the tar file:------
    temp_folder = tempfile.mkdtemp()
    # temp_folder = "/scratch/local1/m300517/tmpu89w80wz"
    print(('Temp-folder ' + temp_folder))

    # ---uncompress Data:-------------
    print("Now Uncompressing " + tar_name)
    uncompressTGZ(tar_path + tar_name, temp_folder)

    # ----preparation for calculation----
    file_list_raw = []
    for element in sorted(glob.glob(temp_folder + '/*.jpg')):
        file_list_raw.append(element)

    print("Sorting out files with SZA > 85")
    file_list = []
    pos_list = []
    # for element in file_list_raw:
    file_elements = Parallel(n_jobs=-1, verbose=5)(delayed(calc_sza)(element) for element in file_list_raw)

    for file_element in file_elements:
        if file_element != "xxx":
            file_list.append(file_element[0])
            pos_list.append(file_element[1])


    if len(file_list) <= 1:
        print("No elements ins file-list")
        sys.exit(1)
    # ----calculate cloudiness----------
    print("Calculating Cloudcoverage")
    result_array = Parallel(n_jobs=-1, verbose=5)(delayed(cloudiness)(file,pos) for file,pos in zip(file_list,pos_list))

    indices = np.where(result_array)[0]
    cc = []
    ASCAtime = []
    cloudmask = []
    for i in indices:
        result_tuple = result_array[i]
        cc.append(result_tuple[0])
        ASCAtime.append(result_tuple[1])
        cloudmask.append(result_tuple[2])
    scale_factor = result_tuple[3]
    # -------Remove temp-folder------
    print("Removing temporaray folder")
    try:
        shutil.rmtree(temp_folder)
        print("Succesfully removed.")
    except:
        print(("Temporary folder {0} could not found or deleted".format(temp_folder)))

    # ------write to netCDF----------
    print("Writing results to netCDF")
    ncName = "CloudCoverage_" + ASCAtime[0].strftime("%Y%m%d") + ".nc"

    if save_cc_mask:
        create_netCDF(ncName, ncPath, cc, ASCAtime, scale_factor, cloudmask)
    else:
        create_netCDF(ncName, ncPath, cc, ASCAtime, scale_factor)
