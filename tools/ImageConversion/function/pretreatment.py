#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：ImageConversion 
@File    ：pretreatment.py
@Author  ：光溜溜的刺蛇
@Date    ：2021/8/24 9:38 
"""
import numpy as np
import cv2
import os
from PIL import Image, ExifTags


def image_whirl(img, x = 0):
    """
    主要将图片进行旋转, 默认逆时针旋转90°
    :param image: 传入的图像数组:h, w, c 三个维度
    :param x: 0为逆时针旋转, 1为顺时针旋转
    :return: 返回处理完成的数组
    """
    trans_img = cv2.transpose(img)
    new_img = cv2.flip(trans_img, x)
    return new_img


def imageConversionjpg(im):
    """
    参照此处文档
    https://paddlex.readthedocs.io/zh_CN/release-1.3/data/annotation.html
    此函数可以将JPG格式的图片转为jpg
    :param im:传入的图像数组 : h w c 三个维度
    :return:此处直接通过引用改变了图片内容, 所以不需要返回值
    """
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(im._getexif().items())
        if exif[orientation] == 3:
            im = im.rotate(180, expand=True)
        if exif[orientation] == 6:
            im = im.rotate(270, expand=True)
        if exif[orientation] == 8:
            im = im.rotate(90, expand=True)
    except:
        pass


def batch_conversion_jpg(path, savePath='.\\out'):
    """
    批量处理图片转化格式的函数
    - 🔥支持多级文件下的扫描, 一键转化🔥
    - 🔥凡是path下的.JPG文件均会被转化并另存为.jpg文件🔥
    :param path: 要处理的目标图片
    :param savePath:预保存的目标位置
    :return:
    """
    length = len(path)
    for root, dirs, files in os.walk(path):
        for file in files:
            filePath = root + '\\' + file
            if filePath[-3:] == 'JPG':
                im = cv2.imread(filePath)
                imageConversionjpg(im)
                res = savePath + root[length:] + '\\' + file[:-3] + 'jpg'
                if not os.path.exists(savePath + root[length:]):
                    os.makedirs(savePath + root[length:])
                cv2.imwrite(res, im)
                print('已保存至' + res)


if __name__ == '__main__':
    # 一个小测试样例 😁
    batch_conversion_jpg('G:\MeterOverview', 'G:\\out')