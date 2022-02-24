import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, join

# TODO Jakość kodu i raport (4/4)


# TODO Skuteczność klasyfikacji 0.929 (4/4)
# TODO [0.00, 0.50) - 0.0
# TODO [0.50, 0.55) - 0.5
# TODO [0.55, 0.60) - 1.0
# TODO [0.60, 0.65) - 1.5
# TODO [0.65, 0.70) - 2.0
# TODO [0.70, 0.75) - 2.5
# TODO [0.75, 0.80) - 3.0
# TODO [0.80, 0.85) - 3.5
# TODO [0.85, 1.00) - 4.0


# TODO Skuteczność detekcji mAP = 0.0 (0/2) (0/6)

# TODO max(0, 4+0) = 4

def learn_bovw(data, t_p):
    dict_size = 128
    bow = cv2.BOWKMeansTrainer(dict_size)
    sift = cv2.SIFT_create()

    for sample in data:
        img = cv2.imread(t_p + sample["image_name"])
        sample_m = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        k_pts = sift.detect(sample_m, None)
        k_pts, desc = sift.compute(sample_m, k_pts)

        if desc is not None:
            bow.add(desc)

    vocabulary = bow.cluster()
    np.save('voc.npy', vocabulary)


def extract_features(data, path, cropped_box_):
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)
    vocabulary = np.load('voc.npy')
    bow.setVocabulary(vocabulary)

    if cropped_box_ is None:
        for sample in data:
            for x in range(len(sample["box_coords"])):
                img = cv2.imread(path + sample["image_name"])
                y_min = sample["box_coords"][x][1]
                y_max = sample["box_coords"][x][3]
                x_min = sample["box_coords"][x][0]
                x_max = sample["box_coords"][x][2]
                cropped_image = img[y_min:y_max, x_min:x_max]
                sample_m = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                k_pts = sift.detect(sample_m, None)
                img_des = bow.compute(sample_m, k_pts)
                if img_des is not None:
                    sample.update({'desc': img_des})
                else:
                    # TODO Lepiej w ogole pominac takie przypadki.
                    sample.update({'desc': np.zeros((1, 128))})

    elif cropped_box_ == "detect":
        img = cv2.imread(path + data["image_name"])
        sample_m = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        k_pts = sift.detect(sample_m, None)
        img_des = bow.compute(sample_m, k_pts)
        if img_des is not None:
            data.update({'desc': img_des})
        else:
            data.update({'desc': np.zeros((1, 128))})
    else:
        img = cv2.imread(path + data["image_name"])
        y_min = data["box_coords"][2]
        y_max = data["box_coords"][3]
        x_min = data["box_coords"][0]
        x_max = data["box_coords"][1]
        cropped_image = img[y_min:y_max, x_min:x_max]
        sample_m = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        k_pts = sift.detect(sample_m, None)
        img_des = bow.compute(sample_m, k_pts)
        if img_des is not None:
            data.update({'desc': img_des})
        else:
            data.update({'desc': np.zeros((1, 128))})

    return data


def build_data(_all_data_, allfiles, tree):
    for n_file in range(len(allfiles)):
        collect_data = {"image_name": None, "numb_of_obj": None, "box_coords": None, "type": None}
        root = tree[n_file].getroot()
        collect_data.update({"image_name": root[1].text})

        wid = int(root[2][0].text)
        hei = int(root[2][1].text)
        ctr = len(root.findall('object'))
        boxes = []
        types = []
        xmin = 0
        xmax = 0
        ymin = 0
        ymax = 0

        collect_data.update({"numb_of_obj": ctr})
        for child2 in root.findall('object'):
            for child3 in child2.findall('bndbox'):
                xmin = int(child3[0].text)
                ymin = int(child3[1].text)
                xmax = int(child3[2].text)
                ymax = int(child3[3].text)
                box = [xmin, ymin, xmax, ymax]
                boxes.append(box)
                collect_data.update({"box_coords": boxes})

            for child4 in child2.findall('name'):
                if child4.text == "speedlimit":
                    diffx = xmax - xmin
                    diffy = ymax - ymin
                    if diffx >= 0.1*wid and diffy >= 0.1*hei:
                        types.append(child4.text)
                        collect_data.update({'label': 1})
                    else:
                        types.append("other")
                        collect_data.update({'label': 0})
                else:
                    types.append("other")
                    collect_data.update({'label': 0})
                collect_data.update({"type": types})
        _all_data_.append(collect_data)
    return _all_data_


def train(data):
    rfc = RandomForestClassifier(128)
    # TODO Mozna tez zrobic "np.empty((0, 128))".
    x_m = np.empty((1, 128))
    y_v = []
    for sample in data:
        y_v.append(sample['label'])
        x_m = np.vstack((x_m, sample['desc']))
    rfc.fit(x_m[1:], y_v)

    return rfc


def train_object_number(data):
    rfc = RandomForestClassifier(128)
    x_m = np.empty((1, 128))
    y_v = []
    for sample in data:
        y_v.append(sample['numb_of_obj'])
        x_m = np.vstack((x_m, sample['desc']))
    # TODO To jest zadanie regresji, a nie klasyfikacji.
    rfc.fit(x_m[1:], y_v)

    return rfc


def train_box(data):
    rfc = RandomForestClassifier(128)
    x_m = np.empty((1, 128))
    y_v = []
    for sample in data:
        for x in range(len(sample['box_coords'])):
            y_v.append(sample['box_coords'][x])
            x_m = np.vstack((x_m, sample['desc']))

    # TODO To jest zadanie regresji, a nie klasyfikacji.
    rfc.fit(x_m[1:], y_v)

    return rfc


def predict(rf, data):
    data.update({'label_pred': rf.predict(data['desc'])[0]})
    if int(data['label_pred']) == 1:
        data.update({'type': "speedlimit"})
    elif int(data['label_pred']) == 0:
        data.update({'type': "other"})
    return data


def predict_object_number(rfobj, data):
    data.update({'numb_of_obj': rfobj.predict(data['desc'])[0]})
    return data


def predict_box(rfbox, data):
    boxes = []
    for x in range(data['numb_of_obj']):
        boxes.append(rfbox.predict(data['desc'])[0])
    data.update({'box_coords': boxes})
    return data


def input_data(data, test_path, rf, rf_obj, rf_box):
    det_or_class = input()
    outpt = []

    if det_or_class == "classify":
        n_files = input()
        for curr_file in range(int(n_files)):
            file_name = input()
            n_obj = input()
            for n in range(int(n_obj)):
                bx = list(map(int, input().strip().split()))[:4]
                curr_data = {"image_name": file_name, "numb_of_obj": n_obj, "box_coords": bx, "type": None}
                samp = extract_features(curr_data, test_path, 1)
                outpt.append(predict(rf, samp))
        for opt in outpt:
            print(opt['type'])
        return outpt

    elif det_or_class == "detect":
        for sample in data:
            samp = extract_features(sample, test_path, det_or_class)
            samp = predict(rf, samp)
            samp = predict_object_number(rf_obj, samp)
            samp = predict_box(rf_box, samp)
            print(samp["image_name"])
            print(samp["numb_of_obj"])
            for x in range(int(samp["numb_of_obj"])):
                xxx = str(samp["box_coords"][x][0]) + ' ' + str(samp["box_coords"][x][1]) + ' ' + \
                      str(samp["box_coords"][x][2]) + ' ' + str(samp["box_coords"][x][3])
                print(xxx)


def set_data():
    allfiles_train = [f1 for f1 in listdir("../train/annotations") if isfile(join("../train/annotations", f1))]
    test_img = [f2 for f2 in listdir("../test/images") if isfile(join("../test/images", f2))]
    tree1 = []
    all_data_train = []
    all_test_images_data = []

    for file1 in allfiles_train:
        pth1 = '../train/annotations/' + file1
        tree1.append(ET.parse(pth1))

    train_path = '../train/images/'
    test_path = '../test/images/'

    for single_img in test_img:
        test_images_data = {"image_name": None, "numb_of_obj": None, "box_coords": None, "type": None}
        test_images_data.update({"image_name": single_img})
        all_test_images_data.append(test_images_data)

    return all_data_train, allfiles_train, all_test_images_data, tree1,  train_path, test_path, test_img


def main():

    all_data_train, allfiles_train, all_test_images_data, tree1, train_path, test_path, test_img = set_data()

    # Load data
    build_data(all_data_train, allfiles_train, tree1)

    if not os.path.isfile('voc.npy'):
        learn_bovw(all_data_train, train_path)

    # extracting train features
    all_data_train = extract_features(all_data_train, train_path, None)

    # training
    rf = train(all_data_train)
    # TODO Pomysl dobry, ale w przypadku klasycznego ML nie zadziala tak dobrze jak w przypadku DNN.
    rf_obj = train_object_number(all_data_train)
    rf_box = train_box(all_data_train)
    input_data(all_test_images_data, test_path, rf, rf_obj, rf_box)

    return


if __name__ == '__main__':
    main()
