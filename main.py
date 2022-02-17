import os
import random
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, join
import time

# TODO Jakość kodu i raport (2.5/4)
# TODO Raport malo przejrzysty.

# TODO Skuteczność klasyfikacji 0.0 (/4)
# TODO [0.00, 0.50) - 0.0
# TODO [0.50, 0.55) - 0.5
# TODO [0.55, 0.60) - 1.0
# TODO [0.60, 0.65) - 1.5
# TODO [0.65, 0.70) - 2.0
# TODO [0.70, 0.75) - 2.5
# TODO [0.75, 0.80) - 3.0
# TODO [0.80, 0.85) - 3.5
# TODO [0.85, 1.00) - 4.0

# stderr:
# Traceback (most recent call last):
#   File "main.py", line 259, in <module>
#     main()
#   File "main.py", line 230, in main
#     allfiles_train, allfiles_test, tree1, tree2, train_path, test_path = set_data()
#   File "main.py", line 180, in set_data
#     tree1.append(ET.parse(pth1))
#   File "/usr/lib/python3.8/xml/etree/ElementTree.py", line 1202, in parse
#     tree.parse(source, parser)
#   File "/usr/lib/python3.8/xml/etree/ElementTree.py", line 584, in parse
#     source = open(source, "rb")
# FileNotFoundError: [Errno 2] No such file or directory: '../train/annotations/road0.xml'

# TODO Skuteczność detekcji 0.0 (/2)

# stderr:
# Traceback (most recent call last):
#   File "main.py", line 277, in <module>
#     main()
#   File "main.py", line 248, in main
#     allfiles_train, allfiles_test, tree1, tree2, train_path, test_path = set_data()
#   File "main.py", line 198, in set_data
#     tree1.append(ET.parse(pth1))
#   File "/usr/lib/python3.8/xml/etree/ElementTree.py", line 1202, in parse
#     tree.parse(source, parser)
#   File "/usr/lib/python3.8/xml/etree/ElementTree.py", line 584, in parse
#     source = open(source, "rb")
# FileNotFoundError: [Errno 2] No such file or directory: '../train/annotations/road0.xml'

# TODO max(0, 0+0) = 0

def bb_intersection_over_union(boxA, boxB):
    x_a = max(boxA[0], boxB[0])
    y_a = max(boxA[1], boxB[1])
    x_b = min(boxA[2], boxB[2])
    y_b = min(boxA[3], boxB[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    boxa_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxb_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = inter_area / float(boxa_area + boxb_area - inter_area)

    return iou


def learn_bovw(data):
    dict_size = 92
    bow = cv2.BOWKMeansTrainer(dict_size)
    sift = cv2.SIFT_create()
    ctrr = 1

    for sample in data:
        img = cv2.imread('../train/images/' + sample.get("image_name"))
        sample_m = cv2.cvtColor(img, None)

        k_pts = sift.detect(sample['image_name'], None)
        k_pts, desc = sift.compute(sample_m, k_pts)

        if desc is not None:
            bow.add(desc)

        # TODO Program nie powinien nic wypisywac.
        print('Progress: ', ctrr, '/', len(data))
        ctrr += 1

    vocabulary = bow.cluster()
    np.save('voc.npy', vocabulary)


def extract_features(data, path):
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)
    vocabulary = np.load('voc.npy')
    bow.setVocabulary(vocabulary)

    for sample in data:
        img = cv2.imread(path + sample["image_name"])
        sample_m = cv2.cvtColor(img, None)
        k_pts = sift.detect(sample_m, None)
        img_des = bow.compute(sample_m, k_pts)
        if img_des is not None:
            sample.update({'desc': img_des})
        else:
            sample.update({'desc': np.zeros((1, 92))})
    return data


def build_data(_all_data_, allfiles, tree):
    for n_file in range(len(allfiles)):
        collect_data = {"image_name": None, "numb_of_obj": None, "box_coords": None, "type": None}
        root = tree[n_file].getroot()
        collect_data.update({"image_name": root[1].text})

        ctr = 0
        boxes = []
        types = []
        for child in root.findall('object'):
            ctr += 1
        collect_data.update({"numb_of_obj": ctr})

        for child2 in root.findall('object'):
            for child3 in child2.findall('bndbox'):
                xmin = int(child3[0].text)
                ymin = int(child3[1].text)
                xmax = int(child3[2].text)
                ymax = int(child3[3].text)
                box = [xmin, ymin, xmax, ymax]
                boxes.append(box)
                # TODO A co w przypadku gdy bedzie wiecej niz jeden obiekt na zdjeciu?
                collect_data.update({"box_coords": boxes})

            for child4 in child2.findall('name'):
                if child4.text == "speedlimit":
                    types.append(child4.text)
                    collect_data.update({'label': 1})
                else:
                    types.append("other")
                    collect_data.update({'label': 0})
                collect_data.update({"type": types})
        _all_data_.append(collect_data)
    return _all_data_


def train(data):
    clf = RandomForestClassifier(92)
    x_matrix = np.empty((1, 92))
    y_vector = []
    for sample in data:
        y_vector.append(sample['label'])
        x_matrix = np.vstack((x_matrix, sample['desc']))
    clf.fit(x_matrix[1:], y_vector)

    return clf


def predict(rf, data):
    for sample in data:
        sample.update({'label_pred': rf.predict(sample['desc'])[0]})

    return data


def input_data(data_test):
    det_or_class = input()
    outpt = []
    if det_or_class == "classify":
        n_files = input()
        for n_f in range(int(n_files)):
            boxes = []
            file_name = input()
            n_obj = input()
            for n in range(int(n_obj)):
                bx = np.array(input())
                boxes.append(bx)
            # TODO Ta petla jest niepotrzebna.
            for sample in data_test:
                if file_name == sample["image_name"]:
                    # TODO Podczas testowania nie mozna wykorzystywac oznaczen z pliku XML!!!
                    # TODO Liczba wycinkow nie koniecznie bedzie rowna liczbie obiektow na zdjeciu.
                    if int(n_obj) == sample["numb_of_obj"]:
                        for x in range(int(n_obj)):
                            xxx = str(sample["box_coords"][x][0]) + ' ' + str(sample["box_coords"][x][1]) + ' ' + \
                                  str(sample["box_coords"][x][2]) + ' ' + str(sample["box_coords"][x][3])
                            if boxes[x] == xxx:
                                outpt.append(sample["type"][0])
        for opt in outpt:
            print(opt)

    elif det_or_class == "detect":
        for sample in data_test:
            print(sample["image_name"])
            print(sample["numb_of_obj"])
            for x in range(int(sample["numb_of_obj"])):
                # TODO Podczas testowania nie mozna wykorzystywac oznaczen z pliku XML!!!
                xxx = str(sample["box_coords"][x][0]) + ' ' + str(sample["box_coords"][x][1]) + ' ' + \
                      str(sample["box_coords"][x][2]) + ' ' + str(sample["box_coords"][x][3])
                print(xxx)

def create_rectangles(_all_data_train, _all_data_test, path):
    img_with_boxes = []
    for curr_data in range(len(_all_data_train)):
        image_path = path + _all_data_train[curr_data].get("image_name")
        image_plus_box = cv2.imread(image_path)

        for sing_box in range(len((_all_data_train[curr_data].get("box_coords")))):
            cv2.rectangle(image_plus_box, _all_data_train[curr_data].get("box_coords")[sing_box][:2], _all_data_train[curr_data].get("box_coords")[sing_box][2:], (0, 255, 0), 2)
            cv2.rectangle(image_plus_box, _all_data_test[curr_data].get("box_coords")[sing_box][:2], _all_data_test[curr_data].get("box_coords")[sing_box][2:], (0, 0, 255), 2)

        # iou = bb_intersection_over_union(_all_data_train[curr_data].get("box_coords"), _all_data_test[curr_data].get("box_coords"))
        # cv2.putText(image_plus_box, "IoU: {:.4f}".format(iou), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # print("{}: {:.4f}".format(image_path, iou))
        img_with_boxes.append(image_plus_box)
    return img_with_boxes


def set_data():
    allfiles_train = [f1 for f1 in listdir("../train/annotations") if isfile(join("../train/annotations", f1))]
    allfiles_test = [f2 for f2 in listdir("../test/annotations") if isfile(join("../test/annotations", f2))]
    tree1 = []
    tree2 = []
    # TODO Obrazy nie koniecznie beda mialy kolejne numery.
    for file1 in range(len(allfiles_train)):
        pth1 = '../train/annotations/road' + str(file1) + '.xml'
        tree1.append(ET.parse(pth1))

    for file2 in range(len(allfiles_test)):
        pth2 = '../test/annotations/road' + str(file2) + '.xml'
        tree2.append(ET.parse(pth2))

    train_path = '../train/images/'
    test_path = '../test/images/'

    return allfiles_train, allfiles_test, tree1, tree2,  train_path, test_path


def display(data):
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

    print("Number of correct images:", len(corr[0]))
    print("Number of incorrect images:", len(incorr[1]))
    grid_size = 8

    corr = dict(sorted(corr.items(), key=lambda item: item[0]))
    corr_disp = {}
    for key, samples in corr.items():
        idxs = random.sample(samples, grid_size)
        corr_disp[key] = [data[idx]['image_name'] for idx in idxs]
    # sort according to classes
    incorr = dict(sorted(incorr.items(), key=lambda item: item[0]))
    incorr_disp = {}
    for key, samples in incorr.items():
        idxs = random.sample(samples, grid_size)
        incorr_disp[key] = [data[idx]['image_name'] for idx in idxs]
    return


def main():

    all_data_train = []
    all_data_test = []

    allfiles_train, allfiles_test, tree1, tree2, train_path, test_path = set_data()

    # print('Load data')
    build_data(all_data_train, allfiles_train, tree1)
    build_data(all_data_test, allfiles_test, tree2)

    if not os.path.isfile('voc.npy'):
        learn_bovw(all_data_train)

    # print('extracting train features')
    all_data_train = extract_features(all_data_train, train_path)

    # print('training')
    rf = train(all_data_train)

    # print('extracting test features')
    all_data_test = extract_features(all_data_test, test_path)

    # print('testing')
    all_data_test = predict(rf, all_data_test)

    input_data(all_data_test)

    # display(all_data_test)

    return


if __name__ == '__main__':
    main()
