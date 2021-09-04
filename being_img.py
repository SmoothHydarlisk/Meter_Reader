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
idx = 0

def parse_args():
    parser = argparse.ArgumentParser(description='Meter Reader Infering')
    parser.add_argument(
        '--det_model_dir',
        dest='det_model_dir',
        help='The directory of the detection model',
        type=str,
        # default='./meter_det_inference_model/'
        default='./meter_det_inference_model')
    parser.add_argument(
        '--seg_model_dir',
        dest='seg_model_dir',
        help='The directory of the segmentation model',
        type=str,
        default='./meter_seg_inference_model/')
    parser.add_argument(
        '--image_dir',
        dest='image_dir',
        help='The directory of images to be inferred',
        type=str,
        default=None)
    parser.add_argument(
        '--image',
        dest='image',
        help='The image to be inferred',
        type=str,
        default='./det_test/123456.jpg')
    parser.add_argument(
        '--use_erode',
        dest='use_erode',
        help='Whether erode the lable map predicted from a segmentation model',
        action='store_true')
    parser.add_argument(
        '--erode_kernel',
        dest='erode_kernel',
        help='Erode kernel size',
        type=int,
        default=4)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the predicted results',
        type=str,
        default='./output/result')
    parser.add_argument(
        '--score_threshold',
        dest='score_threshold',
        help="Predicted bounding boxes whose scores are lower than this threshlod are filtered",
        type=float,
        default=0.5)
    parser.add_argument(
        '--seg_batch_size',
        dest='seg_batch_size',
        help="The number of images fed into the segmentation model during one forward propagation",
        type=int,
        default=4)

    print(parser.parse_args())
    return parser.parse_args()

def is_pic(img_name):
    """判断是否是图片

    参数：
        img_name (str): 图片路径

    返回：
        flag (bool): 判断值。
    """
    valid_suffix = ['JPEG', 'jpeg', 'JPG', 'jpg', 'BMP', 'bmp', 'PNG', 'png']
    suffix = img_name.split('.')[-1]
    flag = True
    if suffix not in valid_suffix:
        flag = False
    return flag


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
                save_dir='./',
                use_erode=True,
                erode_kernel=4,
                score_threshold=0.5,
                seg_batch_size=2):
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
        # 保存扣取的图片
        for res in sub_imgs:
            global idx
            file_name = './datasetdetVOC_image/_'+str(idx)+'OIT.png'
            print('%d------>%s已保存' % (idx, file_name))
            cv2.imwrite(file_name, res)
            idx += 1

def infer(args):
    image_lists = list()
    if args.image is not None:
        if not osp.exists(args.image):
            raise Exception("Image {} does not exist.".format(args.image))
        if not is_pic(args.image):
            raise Exception("{} is not a picture.".format(args.image))
        image_lists.append(args.image)
    elif args.image_dir is not None:
        if not osp.exists(args.image_dir):
            raise Exception("Directory {} does not exist.".format(
                args.image_dir))
        for im_file in os.listdir(args.image_dir):
            if not is_pic(im_file):
                continue
            im_file = osp.join(args.image_dir, im_file)
            image_lists.append(im_file)

    meter_reader = MeterReader(args.det_model_dir)
    if len(image_lists) > 0:
        for image in image_lists:
            print('\n\n-------------------------%s--------------' % image)
            meter_reader.predict(image, args.save_dir, args.use_erode,
                                 args.erode_kernel, args.score_threshold,
                                 args.seg_batch_size)

if __name__ == '__main__':
    args = parse_args()
    infer(args)
