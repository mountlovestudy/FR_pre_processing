# -*- coding: utf-8 -*-
"""
Created on Mon May 19 20:58:03 2014
Detect the landmarks of the person image
the information of the landmarks are saved in the txt
@author: mountain
"""

#import requests
from facepp import *
import os
import pickle
from shutil import copy

API_KEY = MY FACE++ KEY
API_SECRET = MY FACE++ API_SECRET


api = API(API_KEY, API_SECRET)

#dir_path: the path of the folder containing the images(without other files)
dir_path='../data/lfw_process'

#dest_dir: the path of the folder saving the image which can not be detected the person
dest_dir=ur'../person_unuse'


for root,dirs,files in os.walk(dir_path):
    for file in files:
        img_path=File(os.path.join(root, file))
        result = api.detection.detect(img=img_path, mode = 'oneface')
        if result['face']!=[]:
            face_posit=result['face'][0]['position']
            img_name=file.split('.')[0:-1]
            out_name=img_name[0]+'.txt'
            out_path=os.path.join(root, out_name)
            output=open(out_path,'w')
            pickle.dump(face_posit, output)
            output.close()
        else:
            copy(os.path.join(root, file),dest_dir)
    print root
        


