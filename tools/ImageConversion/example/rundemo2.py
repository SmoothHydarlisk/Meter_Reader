#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Meter_Reader_development 
@File    ：rundemo2.py
@Author  ：光溜溜的刺蛇
@Date    ：2021/9/10 21:04 
"""
import os
import cv2 as cv
import tools.ImageConversion.function.pretreatment as fup
if __name__ == '__main__':
    path = 'G:/zzuliWorkSpace/MeterOverview/digitalImg/11-JCQ-10_800A(5.0)'
    for root, dirs, files in os.walk(path):
        for file in files:
            filePath = root + '/' + file
            img = cv.imread(filePath)
            h, w = img.shape[0], img.shape[1]
            if h > 150 and w > 150:
                # 删除表计照片
                os.remove(filePath)
            elif h > w:
                # 逆时针旋转
                img = fup.image_whirl(img)
                cv.imwrite(filePath, img)
