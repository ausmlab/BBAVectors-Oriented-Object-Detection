import torch
from reclassification.MACNN import clustering
from prepare_classification_dataset import crop_image, align_image, align_pnts
from os import listdir, mkdir
from os.path import isfile, join, exists
import cv2
import numpy as np
import pathlib
import math
import copy
import argparse
from tqdm import tqdm

class RCModule(object):
    def __init__(self, model, indicator):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.indicators = np.load(indicator)

    def align_bbxs(self, args):
        dst_aligned_path = join(args.dst_path, 'aligned')
        src_path = args.src_path
        pathlib.Path(dst_aligned_path).mkdir(parents=True, exist_ok=True)
        src_images_path = join(src_path, 'images')
        src_label_path = join(src_path, 'labelTxt')

        image_names = [f for f in listdir(src_images_path) if isfile(join(src_images_path, f))]

        print('-= aligning images =-')
        for image_name in tqdm(image_names):
            image = cv2.imread(join(src_images_path, image_name))
            with open(join(src_label_path, image_name.split('.')[0] + '.txt')) as f:
                labels = f.readlines()
                for label in labels:
                    id, x1, y1, x2, y2, x3, y3, x4, y4, _ = label.split(' ')
                    x1, y1, x2, y2, x3, y3, x4, y4 = int(float(x1)), int(float(y1)), int(float(x2)), int(
                        float(y2)), int(float(x3)), int(float(y3)), int(float(x4)), int(float(y4))
                    x = np.zeros(image.shape[:2], dtype=np.uint8)
                    pts = np.array([[x1, y1], [x2, y2],
                                    [x3, y3], [x4, y4]],
                                   np.int32).reshape(-1, 1, 2)
                    mask = cv2.fillPoly(x, [pts], 255)
                    object = cv2.bitwise_or(image, image, mask=mask)

                    obj_img = align_image(object, pts.reshape([4, 2]), obj_center=True)
                    pts_ = pts.reshape([4, 2])
                    center = tuple(np.array([pts_[1, 0] + pts_[0, 0], pts_[1, 1] + pts_[0, 1]])/2)
                    pts = align_pnts(pts.reshape([4, 2]), center)

                    y1, x1, y2, x2, y3, x3, y4, x4 = pts.reshape(8)

                    obj_img = crop_image(obj_img, (y1, x1, y2, x2, y3, x3, y4, x4))
                    image_final_name = join(dst_aligned_path, '{img}_{id}.jpg'.format(
                        img=image_name.split('.')[0], id=id))
                    if obj_img.shape[0] < obj_img.shape[1]:
                        obj_img = cv2.rotate(obj_img, cv2.ROTATE_90_CLOCKWISE)
                    cv2.imwrite(image_final_name, obj_img)

    def reclassify(self, args):
        indicators_list = clustering(self.indicators)
        self.model.load_state_dict(torch.load(args.model_path))

        object_margin = 10
        image_names = [f for f in listdir(args.src_path) if isfile(join(args.lable_dir, f))]

        for image_name in tqdm(image_names):
            image = cv2.imread(join(args.data_dir, image_name))
            with open(join(args.lable_dir, image_name.split('.')[0] + '.txt')) as f:
                labels = f.readlines()
                for i, label in enumerate(labels):
                    x1, y1, x2, y2, x3, y3, x4, y4, category, _ = label.split(' ')
                    x1, y1, x2, y2, x3, y3, x4, y4 = int(float(x1)), int(float(y1)),int(float(x2)), int(float(y2)),int(float(x3)), int(float(y3)),int(float(x4)), int(float(y4))
                    x = np.zeros(image.shape[:2], dtype=np.uint8)
                    pts = np.array([[x1, y1], [x2, y2],
                                    [x3, y3], [x4, y4]],
                                    np.int32).reshape(-1, 1, 2)
                    mask = cv2.fillPoly(x,[pts],255)
                    object = cv2.bitwise_or(image, image, mask=mask)
                    obj_img = align_image(object, pts.reshape([4, 2]))
                    center = tuple(np.array(image.shape[1::-1]) / 2)
                    pts = align_pnts(pts.reshape([4, 2]), center)
                    y1, x1, y2, x2, y3, x3, y4, x4 = pts.reshape(8)
                    if max(pts.reshape((4, 2))[:,0]) > image.shape[1] or max(pts.reshape((4, 2))[:,1]) > image.shape[0] or min(pts.reshape((8))) < 0:
                        continue
                    obj_img = crop_image(obj_img, (y1, x1, y2, x2, y3, x3, y4, x4))
