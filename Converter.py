# -*- coding: utf-8 -*-
"""
Created on Tue May 31 16:40:49 2016

@author: tobias
"""
def convert(InputFileRaw, image_ASCA, OutputPath):
    from PIL import Image
#    import glob
    import numpy as np


#    OutputPath = "/home/tobias/Dokumente/MPI/ASCA/Output/"
#    InputFileRaw = ('/home/tobias/Schreibtisch/ASCA/Testbilder/cc150430111806.jpg')
#    InputFileASCA = ('/home/tobias/Dokumente/MPI/ASCA/Output/150430_111806_ASCA.jpg')
    
    image_raw = Image.open(InputFileRaw)
    x_size_raw=image_raw.size[0]
    y_size_raw=image_raw.size[1]
    scale = 1
    NEW_SIZE = (x_size_raw*scale , y_size_raw*scale)
    image_raw.thumbnail(NEW_SIZE, Image.ANTIALIAS)

#    image_ASCA = Image.open(InputFileASCA)
    image_ASCA.thumbnail(NEW_SIZE, Image.ANTIALIAS)



    image_array_raw = np.asarray(image_raw, order='F')
    image_array_raw.setflags(write=True)   


    image_array_ASCA = np.asarray(image_ASCA, order='F')
    image_array_ASCA.setflags(write=True)   

    image_array = np.concatenate((image_array_ASCA, image_array_raw), axis = 1)


    image = Image.fromarray(image_array.astype(np.uint8))
    return image
    
#convert(('/home/tobias/Schreibtisch/ASCA/Testbilder/cc150430111806.jpg'),('/home/tobias/Dokumente/MPI/ASCA/Output/150430_111806_ASCA.jpg'),("/home/tobias/Dokumente/MPI/ASCA/Output/"))    