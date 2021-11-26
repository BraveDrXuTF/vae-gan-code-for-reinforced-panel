# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 10:38:47 2020
@author: dcliu
生成训练集
"""

from scipy.interpolate import pchip
import numpy as np
from scipy.interpolate import PPoly
import matplotlib.pyplot as plt
import matplotlib
import cv2
import math
import time
matplotlib.use('Agg')
import random


#排列组合函数
def combination(n,c,com=1,limit=0,per=[]):
    for pos in range(limit,n):
        t = per + [pos]
        if len(set(t)) == len(t):
            if len(t) == c:
                    yield [pos,]
            else:
                    for result in combination(n,c,com,com*pos, per + [pos,]):
                            yield [pos,] + result

#计算控制点数组
def PointSet():
    #范围：[0，100]
    
    #上
    PointSet_1 = [ [0 for col in range(2)] for row in range(HengNumb)]
    for i in range(HengNumb) :
       PointSet_1[i][0] = 100.0/(HengNumb+1)*(i+1) 
       PointSet_1[i][1] = 100.0
    
    #左
    PointSet_2 = [ [0 for col in range(2)] for row in range(ZongNumb)]
    for i in range(ZongNumb) :
       PointSet_2[i][0] = 100.0
       PointSet_2[i][1] = 100.0/(ZongNumb+1)*(i+1) 
    
    #下
    PointSet_3 = [ [0 for col in range(2)] for row in range(HengNumb)]
    for i in range(HengNumb) :
       PointSet_3[i][0] = 100.0/(HengNumb+1)*(i+1) 
       PointSet_3[i][1] = 0.0
    
    #右
    PointSet_4 = [ [0 for col in range(2)] for row in range(ZongNumb)]
    for i in range(ZongNumb) :
       PointSet_4[i][0] = 0.0
       PointSet_4[i][1] = 100.0/(ZongNumb+1)*(i+1) 
    
    PointSet_sum = PointSet_1 + PointSet_2 + PointSet_3 + PointSet_4
    return PointSet_sum


#计算连线数组
def Pairing():
    Tota_Numb = HengNumb*(2*ZongNumb+HengNumb) + ZongNumb*(HengNumb+ZongNumb) + HengNumb*ZongNumb
    Pairing_points = [ [0 for col in range(4)] for row in range(Tota_Numb)]
    #1到234
    i = 0
    for m in range(HengNumb) :
        for n in range(HengNumb,ZongNumb*2+HengNumb*2) :
            Pairing_points[i][0] = PointSet_sum[m][0]
            Pairing_points[i][1] = PointSet_sum[m][1]
            
            Pairing_points[i][2] = PointSet_sum[n][0]
            Pairing_points[i][3] = PointSet_sum[n][1]
            i = i+1
    
    #2到34
    for m in range(HengNumb,ZongNumb+HengNumb) :
        for n in range(HengNumb+ZongNumb,ZongNumb*2+HengNumb*2) :
            Pairing_points[i][0] = PointSet_sum[m][0]
            Pairing_points[i][1] = PointSet_sum[m][1]
            
            Pairing_points[i][2] = PointSet_sum[n][0]
            Pairing_points[i][3] = PointSet_sum[n][1]
            i = i+1
    
    #3到4
    for m in range(ZongNumb+HengNumb,ZongNumb+HengNumb*2) :
        for n in range(HengNumb*2+ZongNumb,ZongNumb*2+HengNumb*2) :
            Pairing_points[i][0] = PointSet_sum[m][0]
            Pairing_points[i][1] = PointSet_sum[m][1]
            
            Pairing_points[i][2] = PointSet_sum[n][0]
            Pairing_points[i][3] = PointSet_sum[n][1]
            i = i+1
    return Pairing_points



if __name__ == '__main__':
    #存储位置
    cwd = "I:\\zhangkunpeng\\bian_Cjin\\"
    #控制点数目
    HengNumb = 2
    ZongNumb = 2
    #允许的筋条数目
    min_num = 5
    max_num = 5
    #允许的重量(总长度)
    min_weight = 0
    max_weitht = 500
    #所保存的图片编号
    pic_num = 0
    #此控制点数目下总0-1选择个数
    Number_sum = ZongNumb*HengNumb*4 + HengNumb*HengNumb + ZongNumb*ZongNumb
      
    time_start = time.time()
    #对所允许的不同筋条数目循环（线的总数为排列组合C[线数][NUmber_sum]）
    for Number_selected in range(min_num,max_num+1):
        #排列组合数组
        Combined_data = []
        Combined_data.clear()
        
        #根据0-1选择总数，和筋条数目生成排列组合
        for res in combination(Number_sum,Number_selected):
            Combined_data.append(res)
        
        #计算控制点数组
        PointSet_sum = PointSet()
        #计算连线数组
        Pairing_points = Pairing()
        
        #对图(设计方案)循环
        for j in range(len(Combined_data)):
        #for j in range(20):
            sum_weight = 0
            plt.clf()
            #对线循环
            for index in range(len(Pairing_points)):
                #判断此线画不画(是否在此设计方案中)
                if index in Combined_data[j]:
                    #画线
                    plt.plot([Pairing_points[index][0],Pairing_points[index][2]],\
                             [Pairing_points[index][1],Pairing_points[index][3]],linewidth = 2.0,color=(0,0,0))
                    #计算线长度
                    p1 = np.array([Pairing_points[index][0],Pairing_points[index][1]])
                    p2 = np.array([Pairing_points[index][2],Pairing_points[index][3]])
                    p3 = p2-p1
                    length = math.hypot(p3[0],p3[1])
                    sum_weight = sum_weight + length
            #输出每个方案的重量
            time_end = time.time()
            print('*******************************************')
            print(pic_num)
            print(sum_weight,time_end -time_start)
            plt.ylim(0, 100)
            plt.xlim(0, 100)
            
#            if sum_weight <= max_weitht and sum_weight >= min_weight :
                
            ##坐标取消
            plt.axis('off')
            fig = matplotlib.pyplot.gcf()
            #设置当前窗口大小
            fig.set_size_inches(0.9*64/70,0.9*64/70)
            #保存图片
            plt.savefig(cwd+'\\line_wid_pic\\%.5d.tiff'%pic_num ,dpi=100)
            plt.close()
            #重新读取图片进行区域裁剪
            img = cv2.imread(cwd+'\\line_wid_pic\\%.5d.tiff'%pic_num)
            img_coordinate = img[10:74,10:74,:]
            cv2.imwrite(cwd+'\\line_wid_pic\\%.5d.tiff'%pic_num,img_coordinate)
            pic_num = pic_num + 1
                
    #输出总图片数目
    print(pic_num)






