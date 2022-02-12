import os
import random
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
import pandas
from PIL import Image
from collections import namedtuple
import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, join



class_id_to_new_class_id = {0: 0,
                            1: 0,
                            2: 0,
                            3: 0,
                            4: 0,
                            5: 0,
                            6: -1,
                            7: 0,
                            8: 0,
                            9: 0,
                            10: 0,
                            11: 1,
                            12: -1,
                            13: 1,
                            14: 0,
                            15: 0,
                            16: 0,
                            17: 0,
                            18: 1,
                            19: 1,
                            20: 1,
                            21: 1,
                            22: 1,
                            23: 1,
                            24: 1,
                            25: 1,
                            26: 1,
                            27: 1,
                            28: 1,
                            29: 1,
                            30: 1,
                            31: 1,
                            32: -1,
                            33: 2,
                            34: 2,
                            35: 2,
                            36: 2,
                            37: 2,
                            38: 2,
                            39: 2,
                            40: 2,
                            41: -1,
                            42: -1}


def load_data(path, filename):
    """
    Loads data from disk.
    @param path: Path to dataset directory.
    @param filename: Filename of csv file with information about samples.
    @return: List of dictionaries, one for every sample, with entries "image" (np.array with image) and "label" (class_id).
    """
    entry_list = pandas.read_csv(os.path.join(path, filename))

    data = []
    for idx, entry in entry_list.iterrows():
        class_id = class_id_to_new_class_id[entry['ClassId']]
        image_path = entry['Path']

        if class_id != -1:
            image = cv2.imread(os.path.join(path, image_path))
            data.append({'image': image, 'label': class_id})

    return data


def draw_grid(images, n_classes, grid_size, h, w):
    """
    Draws images on a grid, with columns corresponding to classes.
    @param images: Dictionary with images in a form of (class_id, list of np.array images).
    @param n_classes: Number of classes.
    @param grid_size: Number of samples per class.
    @param h: Height in pixels.
    @param w: Width in pixels.
    @return: Rendered image
    """
    image_all = np.zeros((h, w, 3), dtype=np.uint8)
    h_size = int(h / grid_size)
    w_size = int(w / n_classes)

    col = 0
    for class_id, class_images in images.items():
        for idx, cur_image in enumerate(class_images):
            row = idx

            if col < n_classes and row < grid_size:
                image_resized = cv2.resize(cur_image, (w_size, h_size))
                image_all[row * h_size: (row + 1) * h_size, col * w_size: (col + 1) * w_size, :] = image_resized

        col += 1

    return image_all


def display(data):
    """
    Displays samples of correct and incorrect classification.
    @param data: List of dictionaries, one for every sample, with entries "image" (np.array with image), "label" (class_id),
                    "desc" (np.array with descriptor), and "label_pred".
    @return: Nothing.
    """
    n_classes = 3

    corr = {}
    incorr = {}

    for idx, sample in enumerate(data):
        if sample['desc'] is not None:
            if sample['label_pred'] == sample['label']:
                if sample['label_pred'] not in corr:
                    corr[sample['label_pred']] = []
                corr[sample['label_pred']].append(idx)
            else:
                if sample['label_pred'] not in incorr:
                    incorr[sample['label_pred']] = []
                incorr[sample['label_pred']].append(idx)

            # print('ground truth = %s, predicted = %s' % (sample['label'], pred))
            # cv2.imshow('image', sample['image'])
            # cv2.waitKey()

    grid_size = 8

    # sort according to classes
    corr = dict(sorted(corr.items(), key=lambda item: item[0]))
    corr_disp = {}
    for key, samples in corr.items():
        idxs = random.sample(samples, min(grid_size, len(samples)))
        corr_disp[key] = [data[idx]['image'] for idx in idxs]
    # sort according to classes
    incorr = dict(sorted(incorr.items(), key=lambda item: item[0]))
    incorr_disp = {}
    for key, samples in incorr.items():
        idxs = random.sample(samples, min(grid_size, len(samples)))
        incorr_disp[key] = [data[idx]['image'] for idx in idxs]

    image_corr = draw_grid(corr_disp, n_classes, grid_size, 800, 600)
    image_incorr = draw_grid(incorr_disp, n_classes, grid_size, 800, 600)

    cv2.imshow('images correct', image_corr)
    cv2.imshow('images incorrect', image_incorr)
    cv2.waitKey()

    # this function does not return anything
    return


def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou



def main():

    allfiles = [f for f in listdir("../train/annotations") if isfile(join("../train/annotations", f))]
    speedlimit = []
    other = []
    tree = []
    all_data = []

    for file in range(len(allfiles)):
        pth = '../train/annotations/road' + str(file) + '.xml'
        tree.append(ET.parse(pth))

    Detection = namedtuple("Detection", ["image_name", "numb_of_obj", "box_coords"])

    print(len(allfiles))
    for n_file in range(len(allfiles)):
        root = tree[n_file].getroot()
        print(root[1].text)
        ctr = 0
        box = []
        boxes = []
        for child in root.findall('object'):
            ctr += 1
        print(ctr)

        for child2 in root.findall('object'):
            for child3 in child2.findall('bndbox'):
                xmin = int(child3[0].text)
                ymin = int(child3[1].text)
                xmax = int(child3[2].text)
                ymax = int(child3[3].text)
                box = [xmin, ymin, xmax, ymax]
                boxes.append(box)
                print(box[0], box[1], box[2], box[3])
        print('\n')
        all_data.append(Detection(root[1].text, ctr, boxes))

    for detection in all_data:
        # load the image
        image_path = '../train/images/' + detection.image_name
        image_plus_box = cv2.imread(image_path)

        # draw the ground-truth bounding box along with the predicted
        # bounding box
        for sing_box in range(len(detection.box_coords)):
            cv2.rectangle(image_plus_box, tuple(detection.box_coords[sing_box][:2]), tuple(detection.box_coords[sing_box][2:]), (0, 255, 0), 2)
        # cv2.rectangle(image, tuple(detection.pred[:2]), tuple(detection.pred[2:]), (0, 0, 255), 2)

        # compute the intersection over union and display it
        # iou = bb_intersection_over_union(detection.gt, detection.pred)
        # cv2.putText(image, "IoU: {:.4f}".format(iou), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # print("{}: {:.4f}".format(image_path, iou))

        # show the output image
        cv2.imshow("Image", image_plus_box)
        cv2.waitKey(0)




    # img = cv2.imread('../train/images/road0.png')

    # col = cv2.cvtColor(img, None)
    # sift = cv2.SIFT_create()
    # kp = sift.detect(col, None)
    # img = cv2.drawKeypoints(col, kp, img)
    #
    # cv2.imwrite('sift_keypoints.png', img)
    # cv2.imshow('test image', img)
    # cv2.waitKey()
    # print("Hello!")

    return


if __name__ == '__main__':
    main()
