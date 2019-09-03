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

os.chdir('/home/yina/Desktop/drone_data')
print(os.getcwd())

#import pixel data -
# 8/24/2019 add three column to be consistent with previous format
pixel_data=pd.read_csv('out_DJI_0014.csv', skiprows=[0])
pixel_data['carCenterY']=-pixel_data['carCenterY']
pixel_data['frameNUM'].max()
pixel_data['stop']=0
pixel_data['lostCounter']=0
pixel_data['missDetection Count']=0

#import car data
car_data=pixel_data.groupby('carID').agg({'carID': np.min,'carL': np.min,'carW': np.min})
cell_arrive_leave_final1=pd.read_csv('cell_out_DJI_0014.csv', names=["a","X","Y","part_id","carCenterX","carCenterY","frameNUM","course","carID","part","status"])
print("finished loading")

###################################################################33
#step 3: get pet
 
#cell_arrive_leave_final1=pd.read_csv('my_csv.csv', names=["a","X","Y","part_id","carCenterX","carCenterY","frameNUM","course","carID","part","status"])
#cell_arrive_leave_final1=cell_arrive_leave_final1[cell_arrive_leave_final1['frameNUM']>=15000]
#print("finished loading")

cell_arrive_leave_final_sort=cell_arrive_leave_final1.sort_values(['X', 'Y','frameNUM'], ascending=[True, True,True])
cell_arrive_leave_final_sort2=cell_arrive_leave_final_sort.shift(-1,axis=0)
cell_arrive_leave_final_sort['X2']=cell_arrive_leave_final_sort2['X']
cell_arrive_leave_final_sort['Y2']=cell_arrive_leave_final_sort2['Y']
cell_arrive_leave_final_sort['carID2']=pd.to_numeric(cell_arrive_leave_final_sort2['carID'])
cell_arrive_leave_final_sort['part2']=pd.to_numeric(cell_arrive_leave_final_sort2['part'])
cell_arrive_leave_final_sort['carCenterX1']=cell_arrive_leave_final_sort2['carCenterX']
cell_arrive_leave_final_sort['carCenterY1']=cell_arrive_leave_final_sort2['carCenterY']
cell_arrive_leave_final_sort['frameNUM1']=cell_arrive_leave_final_sort['frameNUM']
cell_arrive_leave_final_sort=cell_arrive_leave_final_sort.drop(['frameNUM','carCenterX','carCenterY'], axis=1)
cell_arrive_leave_final_sort['frameNUM2']=cell_arrive_leave_final_sort2['frameNUM']
#get the following vehicle position before it 
cell_arrive_leave_final_sort['frameNUM2_previous']=cell_arrive_leave_final_sort['frameNUM2']-20
cell_arrive_leave_final_sort=cell_arrive_leave_final_sort[pd.notnull(cell_arrive_leave_final_sort['carID2'])]
cell_arrive_leave_final_sort['status2']=cell_arrive_leave_final_sort2['status']
cell_arrive_leave_final_sort['course2']=cell_arrive_leave_final_sort2['course']
cell_arrive_leave_final_sort['carCenterX2']=cell_arrive_leave_final_sort2['carCenterX']
cell_arrive_leave_final_sort['carCenterY2']=cell_arrive_leave_final_sort2['carCenterY']
cell_arrive_leave_final_sort=cell_arrive_leave_final_sort[pd.notnull(cell_arrive_leave_final_sort['carID2'])]
pixel_data2=pixel_data[['frameNUM','carID','carCenterX','carCenterY']]
pixel_data2.columns=['frameNUM2_previous','carID2_previous','carCenterX2_previous','carCenterY2_previous']
cell_arrive_leave_final_sort = pd.merge(cell_arrive_leave_final_sort, pixel_data2,
                                        how='left', left_on=['frameNUM1','carID2'],right_on=['frameNUM2_previous','carID2_previous'])
cell_arrive_leave_final_sort['diff_X']=cell_arrive_leave_final_sort['X2']-cell_arrive_leave_final_sort['X']
cell_arrive_leave_final_sort['diff_Y']=cell_arrive_leave_final_sort['Y2']-cell_arrive_leave_final_sort['Y']
cell_arrive_leave_final_sort['diff_frame']=cell_arrive_leave_final_sort['frameNUM2']-cell_arrive_leave_final_sort['frameNUM1']
cell_arrive_leave_final_sort['diff_carID']=cell_arrive_leave_final_sort['carID2']-cell_arrive_leave_final_sort['carID']
#get the angle of the moving direction of the two vehicles
cell_arrive_leave_final_sort['diff_course']=(cell_arrive_leave_final_sort['course2']-cell_arrive_leave_final_sort['course']).abs()
cell_arrive_leave_final_sort['diff_carCenterX']=cell_arrive_leave_final_sort['carCenterX1']-cell_arrive_leave_final_sort['carCenterX2_previous']
cell_arrive_leave_final_sort['diff_carCenterY']=cell_arrive_leave_final_sort['carCenterY1']-cell_arrive_leave_final_sort['carCenterY2_previous']
#get the angle between the carcenter of the following vehicle and the leading vehicle 
cell_arrive_leave_final_sort['carCenter_angle']=np.arctan2(cell_arrive_leave_final_sort['diff_carCenterY'],cell_arrive_leave_final_sort['diff_carCenterX'])*180/(np.pi)
cell_arrive_leave_final_sort.loc[cell_arrive_leave_final_sort['carCenter_angle']<0,'carCenter_angle']=cell_arrive_leave_final_sort['carCenter_angle']+360

#cell_arrive_leave_final_sort['diff_course'][(cell_arrive_leave_final_sort.diff_course>180)]=360-cell_arrive_leave_final_sort['diff_course']
#keep only the same pixel
cell_arrive_leave_final_sort=cell_arrive_leave_final_sort[cell_arrive_leave_final_sort['diff_X']==0]
cell_arrive_leave_final_sort=cell_arrive_leave_final_sort[cell_arrive_leave_final_sort['diff_Y']==0]
#keep the ID is different
cell_arrive_leave_final_sort=cell_arrive_leave_final_sort[cell_arrive_leave_final_sort['diff_carID']!=0]
cell_arrive_leave_final_sort=cell_arrive_leave_final_sort[pd.notnull(cell_arrive_leave_final_sort['carID2'])]
#keep the PET threshold 
PET_threshold=10*60 #set PET=3
cell_arrive_leave_final_sort=cell_arrive_leave_final_sort[cell_arrive_leave_final_sort['diff_frame']<PET_threshold]
#combine the IDs and parts
cell_arrive_leave_final_sort['part_combine']=cell_arrive_leave_final_sort['part2']*100+cell_arrive_leave_final_sort['part']
cell_arrive_leave_final_sort['carID_min']=pd.DataFrame([cell_arrive_leave_final_sort['carID'],cell_arrive_leave_final_sort['carID2']]).min()
cell_arrive_leave_final_sort['carID_max']=pd.DataFrame([cell_arrive_leave_final_sort['carID'],cell_arrive_leave_final_sort['carID2']]).max()
cell_arrive_leave_final_sort['newID']=cell_arrive_leave_final_sort['carID_min']*10000+cell_arrive_leave_final_sort['carID_max']
#exclude two id vehicle if their conflicts number is less than 15 as conflict pixel should not be 1 or very less
cell_arrive_leave_final_sort_count=cell_arrive_leave_final_sort.groupby(['newID','part_combine']).size().to_frame().reset_index()
cell_arrive_leave_final_sort_count.columns=['newID','part_combine','count']
cell_arrive_leave_final_sort_count=cell_arrive_leave_final_sort_count[cell_arrive_leave_final_sort_count['count']>15]
cell_arrive_leave_final_sort_count=cell_arrive_leave_final_sort_count.loc[cell_arrive_leave_final_sort_count.groupby('newID')['part_combine'].idxmin()]

#pixel_data_count=pixel_data.groupby(['carID']).size().to_frame().reset_index()

#find the earliest frame
cell_arrive_leave_final_sort_first=cell_arrive_leave_final_sort[cell_arrive_leave_final_sort['frameNUM1'].isin(cell_arrive_leave_final_sort.groupby(['newID']).min()['frameNUM1'].values)]
cell_arrive_leave_final_sort_first=cell_arrive_leave_final_sort[cell_arrive_leave_final_sort['frameNUM2'].isin(cell_arrive_leave_final_sort.groupby(['newID']).min()['frameNUM1'].values)]

#delete if the conflict pixel number is less than 15
cell_arrive_leave_final_sort_first=pd.merge(cell_arrive_leave_final_sort_first,cell_arrive_leave_final_sort_count,
                                                how='left',left_on=['newID','part_combine'],
                                                right_on=['newID','part_combine'])
cell_arrive_leave_final_sort_first=cell_arrive_leave_final_sort_first[pd.notnull(cell_arrive_leave_final_sort_first['count'])]
#group by pair id find the smallest part
cell_arrive_leave_final_sort_first2=cell_arrive_leave_final_sort_first[cell_arrive_leave_final_sort_first['part_combine'].isin(cell_arrive_leave_final_sort_first.groupby(['carID','carID2']).min()['part_combine'].values)]
#get the distance of two vehicles when the leading vehicle arrived at the conflict points. delete if the center distance is too close which means it is actually the same vehicle
pixel_data3=pixel_data2
pixel_data3.columns=['frameNUM2_previous2','carID2_previous2','carCenterX2_previous2','carCenterY2_previous2']
cell_arrive_leave_final_sort_first2 = pd.merge(cell_arrive_leave_final_sort_first2, pixel_data3,
                                        how='left', left_on=['frameNUM1','carID2'],right_on=['frameNUM2_previous2','carID2_previous2'])
cell_arrive_leave_final_sort_first2=cell_arrive_leave_final_sort_first2[pd.notnull(cell_arrive_leave_final_sort_first2['carID2_previous2'])]
cell_arrive_leave_final_sort_first2['distance_car12']=((cell_arrive_leave_final_sort_first2['carCenterX1']-cell_arrive_leave_final_sort_first2['carCenterX2_previous2'])**2+
                                   (cell_arrive_leave_final_sort_first2['carCenterY1']-cell_arrive_leave_final_sort_first2['carCenterY2_previous2'])**2)**0.5

cell_arrive_leave_final_sort_first2['distance_car12']
cell_arrive_leave_final_sort_first2=cell_arrive_leave_final_sort_first2[cell_arrive_leave_final_sort_first2['distance_car12']>10]

#delete some area has some problems
#cell_arrive_leave_final_sort_first2=cell_arrive_leave_final_sort_first2[cell_arrive_leave_final_sort_first2['X']>500]
#cell_arrive_leave_final_sort_first2=cell_arrive_leave_final_sort_first2[((cell_arrive_leave_final_sort_first2['X']<1400)&(cell_arrive_leave_final_sort_first2['Y']<-500))|
       # ((cell_arrive_leave_final_sort_first2['X']<1400)&(cell_arrive_leave_final_sort_first2['Y']>-400))|
       # ((cell_arrive_leave_final_sort_first2['X']>1700)&(cell_arrive_leave_final_sort_first2['Y']<-500))|
       # ((cell_arrive_leave_final_sort_first2['X']>1700)&(cell_arrive_leave_final_sort_first2['Y']>-400))]
cell_arrive_leave_final_sort_first2['center_course_angle']=(cell_arrive_leave_final_sort_first2['carCenter_angle']-cell_arrive_leave_final_sort_first2['course']).abs()

#get the IOU between arrive the conflict points                                 
cell_arrive_leave_final_sort_first3=cell_arrive_leave_final_sort_first2[['carID','carID2','newID','frameNUM1','frameNUM2','X','Y','diff_course','part_combine','carCenter_angle','center_course_angle','distance_car12']]
L_cell_arrive_leave_final_sort_first3=cell_arrive_leave_final_sort_first3.shape[0]
print("start for loop")
pair_IDs12=pd.DataFrame()
for i in range(0,L_cell_arrive_leave_final_sort_first3):
    print(i)
    ID1_temp=cell_arrive_leave_final_sort_first3.iloc[i,0]
    frameNUM1_temp=cell_arrive_leave_final_sort_first3.iloc[i,3]
    ID2_temp=cell_arrive_leave_final_sort_first3.iloc[i,1]   
    frameNUM2_temp=cell_arrive_leave_final_sort_firsƒt3.iloc[i,4]
    pair_IDs=cell_arrive_leave_final1.loc[(cell_arrive_leave_final1['carID']==ID1_temp) | (cell_arrive_leave_final1['carID']==ID2_temp)]
    pair_IDs=pair_IDs[['frameNUM','X','Y','carID']]
    pair_IDs1=pair_IDs.loc[(pair_IDs['carID']==ID1_temp)&(pair_IDs['frameNUM']<=frameNUM1_temp)&(pair_IDs['frameNUM']>=(frameNUM1_temp-60))]
    #pair_IDs1=pair_IDs1[pair_IDs1['frameNUM'].isin(pair_IDs1.groupby(['X','Y']).min()['frameNUM'].values)]
    pair_IDs1=pair_IDs1.drop_duplicates(subset=['X','Y'])
    pair_IDs2=pair_IDs.loc[(pair_IDs['carID']==ID2_temp)&(pair_IDs['frameNUM']<=frameNUM2_temp)&(pair_IDs['frameNUM']>=(frameNUM2_temp-60))]
    pair_IDs2.columns=['frameNUM2','X2','Y2','carID2']
    pair_IDs2=pair_IDs2.drop_duplicates(subset=['X2','Y2'])
    #pair_IDs2=pair_IDs2[pair_IDs2['frameNUM2'].isin(pair_IDs2.groupby(['X2','Y2']).min()['frameNUM2'].values)]
    pair_IDs12_temp=pd.merge(pair_IDs1,pair_IDs2,how='left',left_on=['X','Y'],
                                                right_on=['X2','Y2'])
    pair_IDs12_temp=pair_IDs12_temp[pd.notnull(pair_IDs12_temp['frameNUM2'])]
    pair_IDs12_temp_size = {'ID1': [ID1_temp], 'ID2': [ID2_temp],
                            'frameNUM1':frameNUM1_temp,'frameNUM2':frameNUM2_temp,
                            'conflict_pixel_count':[pair_IDs12_temp.shape[0]],
                            'n_ID1':[pair_IDs1.shape[0]],'IOU':[pair_IDs12_temp.shape[0]/(pair_IDs1.shape[0]+pair_IDs2.shape[0])]}
    pair_IDs12_temp_size = pd.DataFrame(pair_IDs12_temp_size)
    
    pair_IDs12=pair_IDs12.append(pair_IDs12_temp_size)
pair_IDs12=pair_IDs12.drop_duplicates()
cell_arrive_leave_final_sort_first3=pd.merge(cell_arrive_leave_final_sort_first3,pair_IDs12,how='left',left_on=['carID','carID2','frameNUM1','frameNUM2'],
                                                right_on=['ID1','ID2','frameNUM1','frameNUM2'])
print("start conflict type identify")
#define collision type
#angle:1 sideswipe:2 head on: 3; rear end:4;
cell_arrive_leave_final_sort_first3['collision_type']=1
cell_arrive_leave_final_sort_first3['collision_type'][(cell_arrive_leave_final_sort_first3.diff_course<=45)]=2
cell_arrive_leave_final_sort_first3['collision_type'][(cell_arrive_leave_final_sort_first3.diff_course>315)]=2
cell_arrive_leave_final_sort_first3['collision_type'][(cell_arrive_leave_final_sort_first3.diff_course>135)&(cell_arrive_leave_final_sort_first3.diff_course<=225)]=2
#cell_arrive_leave_final_sort_first3['collision_type'][(cell_arrive_leave_final_sort_first3.diff_course<=45)&(cell_arrive_leave_final_sort_first3.center_course_angle<=5)]=4
#cell_arrive_leave_final_sort_first3['collision_type'][(cell_arrive_leave_final_sort_first3.diff_course<=45)&(cell_arrive_leave_final_sort_first3.center_course_angle>355)]=4
cell_arrive_leave_final_sort_first3['collision_type'][(cell_arrive_leave_final_sort_first3.diff_course<=45)&(cell_arrive_leave_final_sort_first3.IOU>0.15)]=4
#cell_arrive_leave_final_sort_first3['collision_type'][(cell_arrive_leave_final_sort_first3.diff_course>315)&(cell_arrive_leave_final_sort_first3.center_course_angle<=5)]=4
#cell_arrive_leave_final_sort_first3['collision_type'][(cell_arrive_leave_final_sort_first3.diff_course>315)&(cell_arrive_leave_final_sort_first3.center_course_angle>355)]=4
cell_arrive_leave_final_sort_first3['collision_type'][(cell_arrive_leave_final_sort_first3.diff_course>315)&(cell_arrive_leave_final_sort_first3.IOU>0.15)]=4
cell_arrive_leave_final_sort_first3['collision_type'][(cell_arrive_leave_final_sort_first3.diff_course>135)&(cell_arrive_leave_final_sort_first3.diff_course<=225)&(cell_arrive_leave_final_sort_first2.part_combine==101)]=3
cell_arrive_leave_collision=cell_arrive_leave_final_sort_first3.drop_duplicates('newID')
print("skip plotting conflict map")
#plot conflict location
#cdict = {1: 'red', 2: 'blue', 3: 'green',4: 'yellow'}
#collision_type = {1: 'angle', 2: 'sideswipe', 3: 'head on',4: 'rear end'}
#fig, ax = plt.subplots(figsize=(12,6))
#ax.set_xlim([0,2000])
#ax.set_ylim([-1000,-0])
#ax.set_xticks(np.arange(1000, 1100, 1))
#ax.set_yticks(np.arange(-450,-400, 1))
#print("start second loop")
#for i in range(1,5):
    #print(i)
    #cell_arrive_leave_final_sort_PET_temp=cell_arrive_leave_collision[cell_arrive_leave_collision.collision_type==i]
    #plt.scatter(cell_arrive_leave_final_sort_PET_temp['X'],cell_arrive_leave_final_sort_PET_temp['Y'],c=cdict[i],label=i,s=30) 
#ax.legend()
#plt.grid(True)
#plt.show()
print("saving to CSV")

cell_arrive_leave_collision['PET'] = (cell_arrive_leave_collision['frameNUM2']-cell_arrive_leave_collision['frameNUM1'])/60
cell_arrive_leave_collision['conflict_type'] = "rearend"
#angle:1 sideswipe:2 head on: 3; rear end:4;
def f(cell_arrive_leave_collision):
    if cell_arrive_leave_collision['collision_type']==1:
        val="angle"
    elif cell_arrive_leave_collision['collision_type']==2:
        val="sideswipe"
    elif cell_arrive_leave_collision['collision_type']==3:
        val="headon"  
    elif cell_arrive_leave_collision['collision_type']==4:
        val="rearend"  
    else:
        val="missing"
    return val

cell_arrive_leave_collision['conflict_type']=cell_arrive_leave_collision.apply(f,axis=1)

cell_arrive_leave_collision.to_csv("cell_arrive_leave_collision_DJI_0014.csv")

#cell_arrive_leave_collision.groupby(['collision_type']).size()


#cell plot to visualize theresult
import seaborn as sns

f, ax = plt.subplots(figsize=(12, 6))
cmap = sns.cubehelix_palette(as_cmap=True, dark=1, light=0, reverse=True)
#sns.kdeplot(cell_arrive_leave_final_sort_PET_final.X_x, cell_arrive_leave_final_sort_PET_final.Y_y, cmap=cmap, n_levels=60, shade=True);
sns.kdeplot(cell_arrive_leave_collision.X, cell_arrive_leave_collision.Y, cmap=cmap, n_levels=60, shade=True);



#visualize the cell 
#cell_arrive_leave_temp['X']=cell_arrive_leave_temp['X']-cell_arrive_leave_temp['X'].min()
#cell_arrive_leave_temp['Y']=-cell_arrive_leave_temp['Y']
#cell_arrive_leave_temp['Y']=cell_arrive_leave_temp['Y']-cell_arrive_leave_temp['Y'].min()
#cell_arrive_leave_temp['Z']=1
#a=cell_arrive_leave_temp.pivot_table(values='Z', index='Y', columns='X')
#a=a.fillna(0) #change na to 0
#cmap = plt.cm.OrRd
#cmap.set_bad(color='white')
#plt.imshow(a, cmap=cmap, interpolation='nearest')



#plot point
#plot_data=pixel_data.loc[(pixel_data['carID']==121) | (pixel_data['carID']==140)]

#, ax = plt.subplots(figsize=(12,6))
#ax.set_xlim([0,2000])
#ax.set_ylim([-1000,-0])
#ax.set_xticks(np.arange(1000, 1100, 1))
#ax.set_yticks(np.arange(-450,-400, 1))
#plt.scatter(pixel_data['carCenterX'],pixel_data['carCenterY'],c=pixel_data['carID'],s=1) 
#plt.scatter(pixel_data['carCenterX'],pixel_data['carCenterY'],c=pixel_data['carID'],s=1) 
#ax.get_legend().remove()
#plt.grid(True)
#plt.show()


