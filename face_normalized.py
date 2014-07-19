# -*- coding: utf-8 -*-
"""
Created on Fri May 23 10:43:11 2014

Face Normalized

the normalized image size: 150*120
normalized coordinates [x,y]
eye_left:   [36,62]
eye_right:  [84,62]
nose:       [60,93]
mouth_left: [42,117]
mouth_right:[78,117]
 
@author: mountain
"""

import numpy as np
import os
import pickle
import cv2

#the matrix of the normalized coordinates [y',x'] of the landmarks 
#eye_left eye_rigth nose mouth_left mouth_right
eye_left=   np.array([62,36],dtype=float)
eye_right=  np.array([62,84],dtype=float)
nose     =  np.array([93,60],dtype=float)
mouth_left= np.array([117,42],dtype=float)
moutn_right=np.array([117,78],dtype=float)

#M is the matrix composed of the normalized coordinates
m1=np.append(eye_left,[0,0,1,0])
m2=np.append(np.append([0,0],eye_left),[0,1])
M =np.append([m1],[m2],axis=0)

m1=np.append(eye_right,[0,0,1,0])
m2=np.append(np.append([0,0],eye_right),[0,1])
M =np.append(np.append(M,[m1],axis=0),[m2],axis=0)

m1=np.append(nose,[0,0,1,0])
m2=np.append(np.append([0,0],nose),[0,1])
M =np.append(np.append(M,[m1],axis=0),[m2],axis=0)

m1=np.append(mouth_left,[0,0,1,0])
m2=np.append(np.append([0,0],mouth_left),[0,1])
M =np.append(np.append(M,[m1],axis=0),[m2],axis=0)

m1=np.append(moutn_right,[0,0,1,0])
m2=np.append(np.append([0,0],moutn_right),[0,1])
M =np.append(np.append(M,[m1],axis=0),[m2],axis=0)

#transfor M to matrix type
M =np.matrix(M)

#Row_size: is the total number of the rows of the normalized image
#Col_size: is the total number of the columns of the normalized image
Row_size=150
Col_size=120


Row_ori=250
Col_ori=250

#dir_path: the path of the folder containing the image 
dir_path='E:\\GPforFR\\data\\lfw_p'
#the image format in the database
img_fmt='.jpg'

#dest_dir: the path of the folder saving the normalized image
dest_dir=ur'E:\GPforFR\data\a'



def Affain_trans(old_face,h,Row_size=150,Col_size=120):
    '''
    return the new_face Mat by the affain transformation
    old_face: the old_face Mat
    h: the coeffients of the affain transformation
    '''
    new_face=np.zeros((Row_size,Col_size),np.uint8)
    for y in range(Row_size):
        for x in range(Col_size):
            #xynew=[y,x]
            xynew=np.matrix([[y,x,0,0,1,0],[0,0,y,x,0,1]])*h
                        
            
            fy = int(np.floor(xynew[0]))
            cy = int(np.ceil(xynew[0]))
            ry = int(round(xynew[0]))
            fx = int(np.floor(xynew[1]))
            cx = int(np.ceil(xynew[1]))
            rx = int(round(xynew[1]))
            
            '''
            if ry<0 or rx<0 or ry>=Row_ori or rx>=Col_ori:
                new_face[y][x]=0
            else:
                new_face[y][x]=old_face[ry][rx]
            '''
            
            
            #check the interpolation needed or not
            if (abs(xynew[1]-rx)<1e-06) and (abs(xynew[0]-ry)<1e-6):
                #interpolation is not needed
                new_face[y][x]=old_face[ry][rx]

            elif fy<0 or fx<0 or cy>=Row_ori or cx>=Col_ori:
                #or fx<0 or fy<0 or cy>Row_ori or cx>Col_ori:
                new_face[y][x]=0
                
            else:
                    
                #interpolation is needed
                ty = xynew[0]-fy
                
                tx = xynew[1]-fx
            
                #Calculate the interpolation weights of the four neighbors
                w1 = (1-tx)*(1-ty)
                w2 = tx*(1-ty)
                w3 = (1-tx)*ty
                w4 = tx*ty            
                new_face[y][x]=np.uint8(w1*old_face[fy][fx]+w2*old_face[fy][cx]+\
                                w3*old_face[cy][fx]+w4*old_face[cy][cx])
            
            
            
            
    return new_face

for root,dirs,files in os.walk(dir_path):
    #the person name
    person_name=root.split('\\')[-1]
    
    #creat the folder
    out_path=os.path.join(dest_dir,person_name)
    if not os.path.isdir(out_path):
        
        os.makedirs(out_path)
    
        for file in files:
            if file.split('.')[-1]=='txt':
                
                #read the txt
                read_file=open(os.path.join(root,file),'r')
                dictxy=pickle.load(read_file)
                
                
                '''
                calculate the affain transformation coeffcients h=[a b c d e f]
                xy=[y1,x1,y2,x2,...y5,x5] is the coordinate of the five landmarks in the origen image
                x=ax'+by'+c
                y=dx'+ey'+f
                xy'=Mh
                '''        
                xy=np.array([Row_ori*dictxy['eye_left']['y'],Col_ori*dictxy['eye_left']['x'],\
                    Row_ori*dictxy['eye_right']['y'],Col_ori*dictxy['eye_right']['x'],\
                    Row_ori*dictxy['nose']['y'],Col_ori*dictxy['nose']['x'],\
                    Row_ori*dictxy['mouth_left']['y'],Col_ori*dictxy['mouth_left']['x'],\
                    Row_ori*dictxy['mouth_right']['y'],Col_ori*dictxy['mouth_right']['x']])
                xy=0.01*xy
                xy=np.matrix(xy)
                M_T=M.T
                h=np.linalg.inv(M_T*M)*M_T*xy.T
                read_file.close()
                
                # new face based on the affain transformation
                img_name=file.split('.')[0:-1][0]
                old_face=cv2.imread(os.path.join(root,img_name+img_fmt))
                old_face=cv2.cvtColor(old_face,cv2.COLOR_BGR2GRAY)
                new_face =Affain_trans(old_face,h)
                
                
                cv2.imwrite(os.path.join(out_path,img_name+img_fmt),new_face)
        print root
        



    


