from os import listdir, mkdir
from os.path import isfile, join, exists
import cv2
import numpy as np
import pathlib
import math
import copy
import argparse
from tqdm import tqdm



def rotate_image(image, angle, image_center=None):
    if image_center is None:
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def align_pnts(pts, center):
    delta_x = pts[1, 0] - pts[0, 0]
    delta_y = pts[1, 1] - pts[0, 1]
    theta = math.degrees(np.arctan(delta_y / delta_x))
    rotation_mtrx = cv2.getRotationMatrix2D(center, theta, 1.0)
    shited_pnts = pts.reshape([4, 2])
    newrow = [1 for i in range(4)]
    shited_pnts = np.vstack([shited_pnts.T, newrow]).T
    rotated_pnts = np.matmul(rotation_mtrx, shited_pnts.T).T
    readjusted_pnts = rotated_pnts

    return readjusted_pnts.astype(int)


def align_image(image, pts, obj_center=False):
    # This is assuming the point are from top left and clockwise:
    delta_x = pts[1, 0] - pts[0, 0]
    delta_y = pts[1, 1] - pts[0, 1]
    theta = math.degrees(np.arctan(delta_y / delta_x))
    if obj_center:
        rotation_center = tuple(np.array([pts[1, 0] + pts[0, 0], pts[1, 1] + pts[0, 1]]) / 2)
    else:
        rotation_center = None
    image = rotate_image(image, theta, rotation_center)
    return image


def crop_image(image, pts, object_margin=10):
    x1, y1, x2, y2, x3, y3, x4, y4 = pts
    max_y = max([y1, y2, y3, y4])
    min_y = min([y1, y2, y3, y4])
    max_x = max([x1, x2, x3, x4])
    min_x = min([x1, x2, x3, x4])
    left = max([min_x - object_margin, 0])
    right = min([max_x + object_margin, image.shape[1]])
    top = max([min_y - object_margin, 0])
    bottom = min([max_y + object_margin, image.shape[0]])
    crp_img = image[top: bottom, left: right]
    return crp_img


def prepare_image_files(args):
    dst_aligned_path = join(args.dst_path, 'aligned')
    dst_unaligned_path = join(args.dst_path, 'unaligned')
    src_path = args.src_path

    src_images_path = join(src_path, 'images')
    src_label_path = join(src_path, 'labelTxt')

    object_margin = 10

    image_names = [f for f in listdir(src_images_path) if isfile(join(src_images_path, f))]

    for image_name in tqdm(image_names):
        image = cv2.imread(join(src_images_path, image_name))
        with open(join(src_label_path, image_name.split('.')[0] + '.txt')) as f:
            labels = f.readlines()
            for i, label in enumerate(labels):
                x1, y1, x2, y2, x3, y3, x4, y4, category, _ = label.split(' ')
                x1, y1, x2, y2, x3, y3, x4, y4 = int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2)), int(
                    float(x3)), int(float(y3)), int(float(x4)), int(float(y4))
                x = np.zeros(image.shape, dtype=np.uint8)
                pts = np.array([[x1, y1], [x2, y2],
                                [x3, y3], [x4, y4]],
                               np.int32).reshape(-1, 1, 2)
                mask = cv2.fillPoly(x, [pts], 255)
                object = cv2.bitwise_or(image, image, mask=mask)
                obj_img = crop_image(object, (x1, y1, x2, y2, x3, y3, x4, y4))
                or_pts = copy.deepcopy(pts)
                pathlib.Path(join(dst_unaligned_path, category)).mkdir(parents=True, exist_ok=True)
                image_final_name = join(dst_unaligned_path, category,
                                        image_name.split('.')[0] + str(i) + '.' + image_name.split('.')[1])
                if obj_img.shape[0] < obj_img.shape[1]:
                    obj_img = cv2.rotate(obj_img, cv2.ROTATE_90_CLOCKWISE)
                cv2.imwrite(image_final_name, obj_img)

                obj_img = align_image(object, pts.reshape([4, 2]))
                center = tuple(np.array(image.shape[1::-1]) / 2)
                pts = align_pnts(pts.reshape([4, 2]), center)
                y1, x1, y2, x2, y3, x3, y4, x4 = pts.reshape(8)
                if max(pts.reshape((4, 2))[:, 0]) > image.shape[1] or max(pts.reshape((4, 2))[:, 1]) > image.shape[
                    0] or min(pts.reshape((8))) < 0:
                    continue
                obj_img = crop_image(obj_img, (y1, x1, y2, x2, y3, x3, y4, x4))
                pathlib.Path(join(dst_aligned_path, category)).mkdir(parents=True, exist_ok=True)
                image_final_name = join(dst_aligned_path, category,
                                        image_name.split('.')[0] + str(i) + '.' + image_name.split('.')[1])
                if obj_img.shape[0] < obj_img.shape[1]:
                    obj_img = cv2.rotate(obj_img, cv2.ROTATE_90_CLOCKWISE)
                cv2.imwrite(image_final_name, obj_img)


def merge_classes(src_path, dst_path, new_class_name='object'):
    pathlib.Path(dst_path).mkdir(parents=True, exist_ok=True)
    label_files = [f for f in listdir(src_path) if isfile(join(src_path, f))]
    for label_file in label_files:
        with open(join(src_path, label_file)) as f_in:
            labels = f_in.readlines()
            dst_file = join(dst_path, label_file)
            with open(dst_file, 'w') as f_out:
                for label in labels:
                    words = label.strip('\n').split(' ')
                    co_ordinations = ' '.join(words[:-2])
                    difficulty = words[-1]
                    new_label = '{} {} {}\n'.format(co_ordinations, new_class_name, difficulty)
                    f_out.write(new_label)


def prepare_labels(args):
    merge_classes(args.src_path, args.dst_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates dataset for classification')
    parser.add_argument('--src_path', type=str, default=None, help='path to DOTA dataset')
    parser.add_argument('--dst_path', type=str, default=None, help='path to where images should be saved')
    parser.add_argument('--phase', type=str, default='images',
                        help='images: extracts objects and re-aligns them in a new dir,'
                             'labels: creates new labels where all objects '
                             'have the same label')

    args = parser.parse_args()
    if args.phase == 'image':
        prepare_image_files(args)
    elif args.phase =='label':
        prepare_labels(args)
