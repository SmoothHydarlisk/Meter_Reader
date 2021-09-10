#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：ImageConversion 
@File    ：MyConversion.py
@Author  ：光溜溜的刺蛇
@Date    ：2021/8/25 10:21 
"""
import os
import tools.ImageConversion.function.pretreatment as fup
import cv2
import numpy as np


def im_done_1(path, savePath):
    """
    本函数的功能有意愿日后对其升级, 即增加其泛用性, 但目前姑且就这么用吧U•ェ•*U
    :param path: 目标路径
    :param savepath: 存储路径
    :return: 无返回值
    """
    # 支持的图片格式后缀校准, 可添加
    suffix = ['jpg', 'png', 'JPG', 'jpeg']
    length = len(path)
    for root, dirs, files in os.walk(path):
        for file in files:
            # 获取图片格式
            imSuffix = file[-3:]
            # / or \ 自己改吧
            filePath = root + '\\' + file
            if imSuffix in suffix:
                # 获取图片
                im = cv2.imread(filePath)
                if imSuffix == 'JPG':
                    fup.imageConversionjpg(im)
                    file = file[:-3] + 'jpg'
                #获取图片的高和宽
                h, w = im.shape[0], im.shape[1]
                if h < w:
                    im = fup.image_whirl(im, 1)

                im = cv2.resize(im, (1080, 1440))
                if not os.path.exists(savePath + root[length:]):
                    os.makedirs(savePath + root[length:])
                res = savePath + root[length:] + '\\' + file
                # print(res)
                cv2.imwrite(res, im)
                print((res + '已完成'))
    pass

if __name__ == '__main__':
    path = "G:\\zzuliWorkSpace\\MeterOverview\\Meter"
    savePath = "G:\\zzuliWorkSpace\\MeterOverview\\OUT"
    im_done_1(path, savePath)