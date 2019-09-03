# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


##################################################################
#step 1: import data (trajectory data, car profile)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
from numpy.polynomial.polynomial import polyfit
import os
import math
from matplotlib import cm
import multiprocessing 
from multiprocessing import Pool
os.chdir('/home/yina/Desktop/drone_data')
print(os.getcwd())
print("loadingdtata")
pixel_data=pd.read_csv('out_DJI_0002.csv', skiprows=[0])
pixel_data['carCenterY']=-pixel_data['carCenterY']
pixel_data['frameNUM'].max()
#car_data=pd.read_csv('CarProfile.csv')
print("finished loading")

##################################################################
#step 2: generate leave arrive table
#generate car profile
car_data=pixel_data.groupby('carID').agg({'carID': np.min,'carL': np.min,'carW': np.min})
print("generate car profile")
car_data.to_csv("CarProfile_07130723am.csv",index=True,sep=',')

#for calculate the angle between two vehicles' centerpoints
class Vect:

   def __init__(self, a, b):
        self.a = a
        self.b = b

   def findClockwiseAngle(self, other):
       return -math.degrees(math.asin((self.a * other.b - self.b * other.a)/(self.length()*other.length())))
   def length(self):
       return math.sqrt(self.a**2 + self.b**2)
#define the function of bresenham

def Bresenham(x1,y1,x2,y2):

    dx = x2 - x1
    dy = y2 - y1
 
    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)
 
    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
 
    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
 
    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1
 
    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1
 
    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx
 
    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    points=pd.DataFrame(points)
    points.columns=['X','Y']
    return points

def cell_arrive_leave(index):
    MAR=4
    print("-----------------------------")
    print(index)
    try:
        car_pixel=pixel_data.loc[(pixel_data['carID']==index)]
        test_car=car_data.loc[(car_data['carID']==index)] #get the vehicle l w from vehicle data
        carL=test_car['carL'].iloc[2]
        carW=test_car['carW'].iloc[3]

    #plot the original data
        
            #cell_arrive_leave_temp=cell_arrive_leave(car_pixel,carW,carL,MAR=4)
            #
    except:
        return pd.DataFrame()
    if car_pixel.shape[0]<=60:
        return pd.DataFrame()
    print("================================")
    test=car_pixel
    test['carCenterX1']=test['carCenterX'].rolling(window=60).mean()
    test['carCenterY1']=test['carCenterY'].rolling(window=60).mean()
    test['course']=test['course'].rolling(window=10).mean()
    #plt.plot(test['course'])
    test=test[pd.notnull(test['carCenterX1'])]
    L=MAR
    test['frameNUM_L']=test['frameNUM']%L
    test=test.loc[(test['frameNUM_L']==0)]
    test['angle']=(test['course']/180)*(np.pi) #change ganle to radius
    test['cos']=np.cos(test['angle'])
    test['sin']=np.sin(test['angle'])
    ##set up twelve points at the original point
    test['L_X11']=-carW/2 
    test['L_Y11']=carL/2
    test['L_X21']=-carW/2  
    test['L_Y21']=carL/4
    test['L_X31']=-carW/2  
    test['L_Y31']=carL/4+1
    test['L_X41']=-carW/2
    test['L_Y41']=-carL/4
    test['L_X51']=-carW/2
    test['L_Y51']=-carL/4+1
    test['L_X61']=-carW/2
    test['L_Y61']=-carL/2
    test['R_X11']=carW/2
    test['R_Y11']=carL/2
    test['R_X21']=carW/2
    test['R_Y21']=carL/4
    test['R_X31']=carW/2
    test['R_Y31']=carL/4+1
    test['R_X41']=carW/2
    test['R_Y41']=-carL/4
    test['R_X51']=carW/2
    test['R_Y51']=-carL/4+1
    test['R_X61']=carW/2
    test['R_Y61']=-carL/2
    test['L_X1']=test['L_X11']*test['cos']+test['L_Y11']*test['sin']+test['carCenterX']
    test['L_Y1']=test['L_Y11']*test['cos']-test['L_X11']*test['sin']+test['carCenterY']
    test['L_X2']=test['L_X21']*test['cos']+test['L_Y21']*test['sin']+test['carCenterX']
    test['L_Y2']=test['L_Y21']*test['cos']-test['L_X21']*test['sin']+test['carCenterY']
    test['L_X3']=test['L_X31']*test['cos']+test['L_Y31']*test['sin']+test['carCenterX']
    test['L_Y3']=test['L_Y31']*test['cos']-test['L_X31']*test['sin']+test['carCenterY']
    test['L_X4']=test['L_X41']*test['cos']+test['L_Y41']*test['sin']+test['carCenterX']
    test['L_Y4']=test['L_Y41']*test['cos']-test['L_X41']*test['sin']+test['carCenterY']
    test['L_X5']=test['L_X51']*test['cos']+test['L_Y51']*test['sin']+test['carCenterX']
    test['L_Y5']=test['L_Y51']*test['cos']-test['L_X51']*test['sin']+test['carCenterY']
    test['L_X6']=test['L_X61']*test['cos']+test['L_Y61']*test['sin']+test['carCenterX']
    test['L_Y6']=test['L_Y61']*test['cos']-test['L_X61']*test['sin']+test['carCenterY']
    test['R_X1']=test['R_X11']*test['cos']+test['R_Y11']*test['sin']+test['carCenterX']
    test['R_Y1']=test['R_Y11']*test['cos']-test['R_X11']*test['sin']+test['carCenterY']
    test['R_X2']=test['R_X21']*test['cos']+test['R_Y21']*test['sin']+test['carCenterX']
    test['R_Y2']=test['R_Y21']*test['cos']-test['R_X21']*test['sin']+test['carCenterY']
    test['R_X3']=test['R_X31']*test['cos']+test['R_Y31']*test['sin']+test['carCenterX']
    test['R_Y3']=test['R_Y31']*test['cos']-test['R_X31']*test['sin']+test['carCenterY']
    test['R_X4']=test['R_X41']*test['cos']+test['R_Y41']*test['sin']+test['carCenterX']
    test['R_Y4']=test['R_Y41']*test['cos']-test['R_X41']*test['sin']+test['carCenterY']
    test['R_X5']=test['R_X51']*test['cos']+test['R_Y51']*test['sin']+test['carCenterX']
    test['R_Y5']=test['R_Y51']*test['cos']-test['R_X51']*test['sin']+test['carCenterY']
    test['R_X6']=test['R_X61']*test['cos']+test['R_Y61']*test['sin']+test['carCenterX']
    test['R_Y6']=test['R_Y61']*test['cos']-test['R_X61']*test['sin']+test['carCenterY']
    #get all twelve points data
    test1=test.iloc[:,45:69]
    test1[['carCenterX','carCenterY','frameNUM','course','carID']]=test[['carCenterX','carCenterY','frameNUM','course','carID']]
    test1.iloc[:,0:24]=np.around(test1.iloc[:,0:24]).astype(np.int64)
    #select all cells
    Len_test1=test1.shape[0]
    points=pd.DataFrame()
    print("]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]")
    for i in range(0,Len_test1):
        print("car "+str(index)+" - "+ str(i)+"/"+str(Len_test1))
        for j in range(0,5):
            x_start=test1.iloc[i,(j*2)]
            y_start=test1.iloc[i,(j*2+1)]
            x_end=test1.iloc[i,((j+1)*2)]
            y_end=test1.iloc[i,((j+1)*2+1)]
            points_temp=Bresenham(x_start,y_start,x_end,y_end)
            points_temp['part_id']=(j+1)*100+(j+2)
            points_temp['part_id']=(j+1)*100+(j+2)
            points_temp['carCenterX']=test1.iloc[i,24]
            points_temp['carCenterY']=test1.iloc[i,25]
            points_temp['frameNUM']=test1.iloc[i,26]
            points_temp['course']=test1.iloc[i,27]
            points_temp['carID']=test1.iloc[i,28]
            points=points.append(points_temp)
        for j in range(6,11):
            x_start=test1.iloc[i,(j*2)]
            y_start=test1.iloc[i,(j*2+1)]
            x_end=test1.iloc[i,((j+1)*2)]
            y_end=test1.iloc[i,((j+1)*2+1)]
            points_temp=Bresenham(x_start,y_start,x_end,y_end)
            points_temp['part_id']=(j+1)*100+(j+2)
            points_temp['carCenterX']=test1.iloc[i,24]
            points_temp['carCenterY']=test1.iloc[i,25]
            points_temp['frameNUM']=test1.iloc[i,26]
            points_temp['course']=test1.iloc[i,27]
            points_temp['carID']=test1.iloc[i,28]
            points=points.append(points_temp)
        for j in range(0,6):
            x_start=test1.iloc[i,j*2]
            y_start=test1.iloc[i,j*2+1]
            x_end=test1.iloc[i,(j+6)*2]
            y_end=test1.iloc[i,(j+6)*2+1]
            points_temp=Bresenham(x_start,y_start,x_end,y_end)
            points_temp['part_id']=(j+1)*100+(j+7)
            points_temp['carCenterX']=test1.iloc[i,24]
            points_temp['carCenterY']=test1.iloc[i,25]
            points_temp['frameNUM']=test1.iloc[i,26]
            points_temp['course']=test1.iloc[i,27]
            points_temp['carID']=test1.iloc[i,28]
            points=points.append(points_temp)
    points['part']=0
    points['part'][(points['part_id'] ==102)] = 1 #1 head 2 middle 3 rear
    points['part'][(points['part_id'] ==107)] = 1
    points['part'][(points['part_id'] ==208)] = 1  
    points['part'][(points['part_id'] ==708)] = 1 
    points['part'][(points['part_id'] ==304)] = 2 #1 head 2 middle 3 rear
    points['part'][(points['part_id'] ==309)] = 2
    points['part'][(points['part_id'] ==410)] = 2  
    points['part'][(points['part_id'] ==910)] = 2 
    points['part'][(points['part_id'] ==506)] = 3 #1 head 2 middle 3 rear
    points['part'][(points['part_id'] ==511)] = 3
    points['part'][(points['part_id'] ==1112)] =3  
    points['part'][(points['part_id'] ==612)] = 3 
    points = points[points['part'] != 0]
    #get all arrival and leave carid, framenum, and part
    #arrive_points=points.groupby(['X','Y','carID','part'], as_index=False)['frameNUM'].min()
    arrive_points=points[points['frameNUM'].isin(points.groupby(['X','Y','carID','part']).min()['frameNUM'].values)]
    arrive_points['status']=1 #1 is arrive and 2 is leave
    leave_points=points[points['frameNUM'].isin(points.groupby(['X','Y','carID','part']).max()['frameNUM'].values)]
    leave_points['status']=2
    cell_arrive_leave=arrive_points.append(leave_points)
    #output the cell data
    #cell_arrive_leave_final.append(cell_arrive_leave)
    #print(cell_arrive_leave)
    with open('cell_leave_arrive_07.csv', 'a') as f:
        cell_arrive_leave.to_csv(f, header=False)
    # cell_arrive_leave.to_csv(str(index)+".csv")
    return cell_arrive_leave

cell_arrive_leave_final=pd.DataFrame()
carnum=car_data['carID'].max()+1
pool = Pool(os.cpu_count()-1)
#pool = Pool(2)
results = [pool.apply_async(cell_arrive_leave, args=(x,)) for x in range(1,carnum)]
cell_arrive_leave_final = [p.get() for p in results]
print("finished reading cars =====================")   
print(os.cpu_count())


