#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Meter_reader_development 
@File    ：mybeing_img.py
@Author  ：光溜溜的刺蛇
@Date    ：2021/9/3 16:03 
"""
import os
import os.path as osp
import numpy as np
import math
import cv2
import argparse
from paddlex import transforms as T
import paddlex as pdx
# export CUDA_VISIBLE_DEVICES=0
METER_SHAPE = [512, 512]  # 高x宽
# 模型路径



class MeterReader:
    """检测表盘位置的执行类"""
    def __init__(self, del_model_dir):
        """初始化类"""
        if not osp.exists(del_model_dir):
            raise Exception("del模型路径不存在!")
        self.detector = pdx.load_model(del_model_dir)

    def decode(self, img_file):
        """图像解码

        参数:
            img_file(strlnp.array):图像路径,或者时以解码的BGR图像数组.

        返回:
            img(np.array):BGR图像数组.
        """
        if isinstance(img_file, str):
            img = cv2.imread(img_file).astype('float32')
        else:
            print(img_file)
            img = img_file.copy()
        return img

    def filter_bboxes(self, det_results, score_threshold):
        """过滤置信度低于阈值的检测框

        参数:
            det_results(list[dict]):检测模型预测接口的返回值.
            score_threshold(float):置信度阈值

        返回:
            filtered_results(list[dict]):过滤后的检测框

        """
        filtered_results = list()
        for res in det_results:
            if res['score'] > score_threshold:
                filtered_results.append(res)
        return filtered_results

    def roi_crop(self, img, det_results):
        """扣出检测框中的图像

        :param img(np.array):BGR图像数组
        :param det_results(list[dic]):检测模型预测接口的返回值
        :return:sub_imgs(list[np.array]):各检测框的图像区域
        """
        sub_imgs = []
        for res in det_results:
            xmin, ymin, w, h = res['bbox']
            xmin = max(0, int(xmin))
            ymin = max(0, int(ymin))
            xmax = min(img.shape[1], int(xmin + w - 1))
            ymax = min(img.shape[0], int(ymin + h - 1))
            sub_img = img[ymin:(ymax + 1), xmin:(xmax + 1), :]
            sub_imgs.append(sub_img)
        return sub_imgs

    def resize(self, imgs, target_size, interp=cv2.INTER_LINEAR):
        """图像缩放至固定大小

        参数：
            imgs (list[np.array])：批量BGR图像数组。
            target_size (list|tuple)：缩放后的图像大小，格式为[高, 宽]。
            interp (int)：图像差值方法。默认值为cv2.INTER_LINEAR。

        返回：
            resized_imgs (list[np.array])：缩放后的批量BGR图像数组。

        """

        resized_imgs = list()
        for img in imgs:
            img_shape = img.shape
            scale_x = float(target_size[1]) / float(img_shape[1])
            scale_y = float(target_size[0]) / float(img_shape[0])
            resize_img = cv2.resize(
                img, None, None, fx=scale_x, fy=scale_y, interpolation=interp)
            resized_imgs.append(resize_img)
        return resized_imgs

    def predict(self,
                img_file,
                score_threshold=0.5):
        """检测图像中的表盘，而后分割出各表盘中的指针和刻度，对分割结果进行读数后处理后得到各表盘的读数。

        参数：
            img_file (str)：待预测的图片路径。
            save_dir (str): 可视化结果的保存路径。
            use_erode (bool, optional): 是否对分割预测结果做图像腐蚀。默认值：True。
            erode_kernel (int, optional): 图像腐蚀的卷积核大小。默认值: 4。
            score_threshold (float, optional): 用于滤除检测框的置信度阈值。默认值：0.5。
            seg_batch_size (int, optional)：分割模型前向推理一次时输入表盘图像的批量大小。默认值为：2。
        """

        img = self.decode(img_file)
        det_results = self.detector.predict(img)
        filtered_results = self.filter_bboxes(det_results, score_threshold)
        sub_imgs = self.roi_crop(img, filtered_results)
        sub_imgs = self.resize(sub_imgs, METER_SHAPE)
        return sub_imgs


if __name__ == '__main__':
    # 模型路径
    det_model_dir = './MODEL/Object_det_model/'
    # 图片路径
    image_dir = './TestImg'
    # 保存路径
    save_dir = './OUT'
    # 获取模型
    meter_reader = MeterReader(det_model_dir)
    # 目标图片后缀
    suffix = ['jpg', 'png', 'JPG', 'jpeg']
    length = len(image_dir)
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            # 获取图片格式
            imSuffix = file[-3:]
            # 图片完整的相对路径
            filePath = root + '/' + file
            if imSuffix in suffix:
                # 获得预测结果
                resImgs = meter_reader.predict(filePath)
                if not os.path.exists(save_dir + root[length:]):
                    os.makedirs(save_dir + root[length:])

                savePath = save_dir + root[length:] + '/'
                img_name = file[:-4]
                for idx in range(len(resImgs)):
                    if idx != 0:
                        img_name = img_name + '_' + str(idx)
                    img_name = img_name + '.png'
                    cv2.imwrite(savePath + img_name, resImgs[idx])
                    print(savePath + img_name + '已完成')


