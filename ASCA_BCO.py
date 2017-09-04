    
def cloudiness(InputFilePath):    
    # -*- coding: utf-8 -*-
    from PIL import Image, ImageDraw, ImageOps, ImageFont
    from scipy.interpolate import interp1d
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
    
    #Information:
    #
    #Code written by Marcus Klingebiel, Max-Planck-Institute for Meteorology
    #E-Mail: marcus.klingebiel@mpimet.mpg.de
    #
    #PLEASE ASK BEFORE SHARING THIS CODE!
    #
    #
    #Preliminary version of the All-Sky Cloud Algorithms (ASCA)
    #The code is based on the analysis of every single pixel on a jpeg-Image. 
    #The used Ski-Index and Brightness-Index base on Letu et al. (2014), Applied Optics, Vol. 53, No. 31. 
    #
    #
    #
    #Marcus Klingebiel, March 2016
    
    #Code eddited by Tobias Machnitzki
    #Email: tobias-machnitzki@web.de
    print("!!!!!!!!!!!!!!!!!!Running ASCA!!!!!!!!!!!!!!!!!")
    
    #--------------------Settings------------------------------------------------------------------------------------------
    
    debugger = False #if true, the program will print a message after each step
    
    TXTFile=False #if True, the program will generate a csv file with several information. Delimiter = ','
    
#    imagefont_size = 20 #Sets the font size of everything written into the picture
    
    Radius_synop = True #If True: not the whole sky will be used, but just the 60 degrees from the middle on (like the DWD does with cloud covering)
    
    Save_image = False #If True: an image will be printed at output-location, where recognized clouds are collored.
    
#    font = ImageFont.truetype("/home/tobias/anaconda3/lib/python3.5/site-packages/matplotlib/mpl-data/fonts/ttf/Vera.ttf",imagefont_size)    # Font 
     
    set_scale_factor = 10   #this factor sets the acuracy of the program. By scaling down the image size the program gets faster but also its acuracy dercreases.
                            #It needs to be between 1 and 100. If set 100, then the original size of the image will be used. 
                            #If set to 50 the image will be scaled down to half its size
                            #
    

    #---------------------Calcutlate the SI-parameter--------------------------------------------------------------------------------------
    #The Parameter gets calculated before the loop over all filse start, to save computing time.
    #To see how the function for the parameter was generated, see the documentation.
    
    size = 100  
    parameter = np.zeros(size)  
    for j in range(size):
            parameter[j] = (0+j*0.4424283716980435-pow(j,2)*0.06676211439554262+pow(j,3)*0.0026358061791573453-pow(j,4)*0.000029417130873311177+pow(j,5)*1.0292852149593944e-7)*0.001
    
    
    
    #----------------------Read files------------------------------------------------------------------------------------------------------

    OutputPath = "/media/MPI/ASCA/images/s160521/out/"

    
    cloudiness_value = []
    ASCAtime = []
    print(InputFilePath)

    for InputFile in sorted(glob.glob(InputFilePath+'/cc*.jpg')):

    
    #-------------------------------------------------------------------------------------------------------------------------------------
    
        print("Aktueller FIle "+InputFile)
    

    
    #---------------------------------------------------------------------------------------------------------
        #--------Get day and time------------
        if debugger == True:
            print "Getting day and time"
            
        date_str = InputFile[len(InputFile)-16:len(InputFile)-16+12]
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
    
    #------------Calculate SZA--------------------------------------------------------------------------------------------------------------
        if debugger == True:
            print "Calculating SZA"
    
    
    
        tus = Location(13.164, -59.433,'UTC', 70, 'BCO')   #This is the location of the Cloud camera used for calculating the Position of the sun in the picture
        times = pd.date_range(start=datetime.datetime(Year+2000,Month,Day,Hour,Minute,Second), end=datetime.datetime(Year+2000,Month,Day,Hour,Minute,Second), freq='1Min')
        times_loc = times.tz_localize(tus.pytz)
        pos = pvlib.solarposition.get_solarposition(times_loc, tus.latitude, tus.longitude, method='nrel_numpy', pressure=101325, temperature=25)
        sza = float(pos.zenith[0])
    
        
        if sza <= 85:            #The program will only process images made at daylight

            time1 = clock()     
            azimuth = float(pos.azimuth[0])
            sza_orig = sza
            azi_orig = azimuth
            azimuth = azimuth + 190        #197 good
            print( str(sza) + '   '+Hour_str+':'+Minute_str)
            if azimuth > 360:
                    azimuth=azimuth-360
    
    #------------Open csv-File-------------------------------------------------------------------------------------------------------------
            if debugger == True:
                print "Open csv-File"
    
            if TXTFile==True:
                f = open(OutputPath+Year_str+Month_str+Day_str+'_'+Hour_str+Minute_str+Second_str+'_ASCA.csv','w')
                f.write('Seconds_since_1970, UTC_Time, SZA_in_degree, Azimuth_in_degree, Cloudiness_in_percent, Cloudiness_in_oktas'+'\n')
                TXTFile=False
    
    #---Read image and set some parameters-------------------------------------------------------------------------------------------------
            if debugger == True:
                print "Reading image and setting parameters"
    
            #------------rescale picture-------------------------------------------
            image = Image.open(InputFile)

            x_size_raw=image.size[0]
            y_size_raw=image.size[1]
            scale_factor = (set_scale_factor/100.)
            NEW_SIZE = (x_size_raw*scale_factor , y_size_raw*scale_factor)
            image.thumbnail(NEW_SIZE, Image.ANTIALIAS)

            image = ImageOps.mirror(image)        #Mirror picture
    

    
            x_size=image.size[0]
            y_size=image.size[1]
            x_mittel = x_size/2    # Detect center of the true image
            y_mittel = y_size/2            
            Radius = 900 #pixel    #  Set area for the true allsky image
    
            scale = x_size/2592.

    
    #-------------convert image to an array and remove unnecessary part araund true allsky image-----------------------------------------------------------------
            if debugger == True:
                print "Drawing circle around image and removing the rest"
    
    
            r=Radius*scale
            y,x = np.ogrid[-y_mittel:y_size-y_mittel, -x_mittel:x_size-x_mittel]   
            x = x + (15*scale)          #move centerpoint manually
            y = y - (40*scale)
            mask = x**2+y**2 <= r**2   #make a circular boolean array which is false in the area outside the true allsky image
              
            image_array = np.asarray(image, order='F')   #converting the image to an array with array[x,y,color]; color: 0=red, 1,green, 2=blue
            image_array.setflags(write=True)    #making it able to work with that array and change it
            image_array[:,:,:][~mask] = [0,0,0]  #using the mask created before on that new made array
    
            if Radius_synop == True:
                mask = x**2+y**2 <= (765*scale)**2 
                image_array[:,:,:][~mask] = [0,0,0]     
    
            del x,y
    #  
    #------------Calculate position of sun on picture---------------------------------------------------------------------------------------
            if debugger == True:
                print "Calculating position of the sun on picture"
    
            sza=sza-90
            if sza < 0:
                sza=sza*(-1)
    
            AzimutWinkel=((2*math.pi)/360)*(azimuth-90)            
            sza = ((2*math.pi)/360)*sza
            x_sol_cen=x_mittel-(15*scale)
            y_sol_cen=y_mittel+(40*scale)
            RadiusBild=r
            sza_dist=RadiusBild*math.cos(sza)
    
            x=x_sol_cen-sza_dist*math.cos(AzimutWinkel)
            y=y_sol_cen-sza_dist*math.sin(AzimutWinkel)
    
    
    ###-----------Draw circle around position of sun-------------------------------------------------------------------------------------------
            if debugger == True:
                print "Drawing circle around position of sun"
    
    
            x_sol_cen=int(x)
            y_sol_cen=int(y)
            Radius_sol=300*scale
            Radius_sol_center=250*scale
            
    
    
            y,x = np.ogrid[-y_sol_cen:y_size-y_sol_cen, -x_sol_cen:x_size-x_sol_cen]
            sol_mask = x**2+y**2 <= Radius_sol**2
            sol_mask_cen = x**2+y**2 <= Radius_sol_center**2
            sol_mask_cen1 = sol_mask_cen
            image_array[:,:,:][sol_mask_cen] = [0,0,0]
    #        image_array[:,:,:][]         
    
    ##-------Calculate Sky Index SI and Brightness Index BI------------Based on Letu et al. (2014)-------------------------------------------------
            if debugger == True:
                print "Calculating Sky Index SI and Brightness Index BI"   
    
            image_array_f = image_array.astype(float) 
       
            SI = ((image_array_f[:,:,2]) - (image_array_f[:,:,0]))/(((image_array_f[:,:,2]) + (image_array_f[:,:,0])))
            where_are_NaNs = np.isnan(SI)
            SI[where_are_NaNs] = 1
      
            mask_sol1 = SI < 0.1  
            Radius = 990*scale
            sol_mask_double = x**2+y**2 <= Radius**2
            mask_sol1 = np.logical_and(mask_sol1, ~sol_mask_double) 
            image_array[:,:,:][mask_sol1]=[255,0,0]
      
    
    ###-------------Include area around the sun----------------------------------------------------------------------------------------------------
            if debugger == True:
                print "Including area around the sun"
         
            y,x = np.ogrid[-y_sol_cen:y_size-y_sol_cen, -x_sol_cen:x_size-x_sol_cen]
            sol_mask = x**2+y**2 <= Radius_sol**2
            sol_mask_cen = x**2+y**2 <= Radius_sol_center**2 
            sol_mask_cen = np.logical_and(sol_mask_cen, sol_mask)
       
            Radius_sol = size*100*2   
            sol_mask = x**2+y**2 <= Radius_sol**2        
            mask2 = np.logical_and(~sol_mask_cen, sol_mask)
       
            image_array_c = copy.deepcopy(image_array) #duplicating array: one for counting one for printing a colored image   
             
            time3 = clock() 
      
            for j in range(size):  
                Radius_sol =j*10*scale  
                sol_mask = (x*x)+(y*y) <= Radius_sol*Radius_sol     
                mask2 = np.logical_and(~sol_mask_cen,sol_mask )    
                sol_mask_cen = np.logical_or(sol_mask, sol_mask_cen)
          
                mask3 = SI < parameter[j]            
                mask3 = np.logical_and(mask2, mask3)                   
                image_array_c[mask3] = [255,0,0]           
                image_array[mask3] = [255,300-3*j,0]

    
    
            
            time4 = clock() 
    #        print 'Schleifenzeit:', time4-time3
    ##---------Count red pixel(clouds) and blue-green pixel(sky)-------------------------------------------------------------------------------------------
            if debugger == True:
                print "Counting red pixel for sky and blue for clouds"
    
    
            c_mask = np.logical_and(~sol_mask_cen1, mask)    
            c_array = (image_array_c[:,:,0]+image_array_c[:,:,1]+image_array_c[:,:,2])  #array just for the counting
            Count1 = np.shape(np.where(c_array == 255))[1]
            Count2 = np.shape(np.where(c_mask == True))[1]
    
            CloudinessPercent=(100/float(Count2)*float(Count1))
            CloudinessSynop=int(round(8*(float(Count1)/float(Count2))))
     
      
            image = Image.fromarray(image_array.astype(np.uint8))
             
    #----------Mirror Image-----------------------------
            image = ImageOps.mirror(image)        #Mirror Image back
    #---------Add Text-----------------------------------
            if debugger == True:
                print "Adding text"
    
    
    
            sza="{:5.1f}".format(sza_orig)
            azimuth="{:5.1f}".format(azi_orig)
            CloudinessPercent="{:5.1f}".format(CloudinessPercent)
    
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
    
    #-------------Save values to csv-File---------------------------------------
#            if debugger == True:
#                print "Saving values to csv-File"
    
#            EpochTime=(datetime.datetime(2000+Year,Month,Day,Hour,Minute,Second) - datetime.datetime(1970,1,1)).total_seconds()
#            f.write(str(EpochTime)+', '+Hour_str+':'+Minute_str+', '+str(sza)+', '+str(azimuth)+', '+str(CloudinessPercent)+', '+str(CloudinessSynop)+'\n')
    #-------------Save picture--------------------------------------------------
            if Save_image == True:
                if debugger == True:
                            print "saving picture"
                  
                image = convert(InputFile, image, OutputPath)   
                image.save(OutputPath+Year_str+Month_str+Day_str+'_'+Hour_str+Minute_str+Second_str+'_ASCA.jpg')
     
            #image.show()
            time2 = clock()
            time = time2 - time1
            cloudiness_value.append(CloudinessPercent)
            ASCAtime.append(mdate.date2num(datetime.datetime(Year+2000,Month,Day,Hour,Minute,Second)))            
        
#               print "Berechnungszeit: ", time
    return cloudiness_value, ASCAtime
