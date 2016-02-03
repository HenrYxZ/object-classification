import cv2
import time
import pickle
import numpy as np
from matplotlib import pyplot as plt

#----------------------------------
# Local dependencies
import descriptors
import utils
from dataset import Dataset

def test_dataset():
    dataset = Dataset("dataset")
    pickle.dump(dataset, open("dataset.obj", "wb"), protocol=2)
    classes = dataset.get_classes()
    print("Dataset generated with {0} classes.".format(len(classes)))
    print(classes)
    train = dataset.get_train_set()
    test = dataset.get_test_set()
    for i in range(len(classes)):
        print(
            "There are {0} training files and {1} testing files for class number {2} ({3})".format(
                len(train[i]), len(test[i]), i, classes[i]
            )
        )

def test_des_type():
    img = cv2.imread("dataset/cassava/n12926689_5139.JPEG")
    kp, des = descriptors.orb(img)
    return des

def test_descriptors():

    img = cv2.imread("dataset/cassava/n12926689_5139.JPEG")
    cv2.imshow("Normal Image", img)
    print(
        "Normal Image\n"\
        "Press [1] to use ORB features or other key to use SIFT features"
    )
    option = cv2.waitKey()
    start = time.time()
    key_one = ord('1')
    if option == key_one:
        kp, des = descriptors.orb(img)
    else:
        kp, des = descriptors.sift(img)
    end = time.time()
    elapsed_time = utils.humanize_time(end - start)
    des_name = "ORB" if option == 49 else "SIFT"
    print("Elapsed time getting descriptors {0}".format(elapsed_time))
    print("Number of descriptors found {0}".format(len(des)))
    if des is not None and len(des) > 0:
        print("Dimension of descriptors {0}".format(len(des[0])))
    print("Name of descriptors used is {0}".format(des_name))
    img2 = cv2.drawKeypoints(img, kp)
    # plt.imshow(img2), plt.show()
    cv2.imshow("{0} descriptors".format(des_name), img2)
    print("Press any key to exit ...")
    cv2.waitKey()

def test_codebook():
    dataset = pickle.load(open("dataset.obj", "rb"))
    #print("Press [1] to use ORB features or any other key to use SIFT features")
    #option = cv2.waitKey()
    option = ord('1')
    start = time.time()
    des = descriptors.all_descriptors(dataset, dataset.get_train_set(), option)
    end = time.time()
    elapsed_time = utils.humanize_time(end - start)
    print("Elapsed time getting all the descriptors is {0}".format(elapsed_time))
    k = 64
    start = time.time()
    codebook = descriptors.gen_codebook(dataset, des, k)
    end = time.time()
    elapsed_time = utils.humanize_time(end - start)
    print("Elapsed time calculating the k means for the codebook is {0}".format(elapsed_time))
    np.savetxt("codebook64.csv", codebook, delimiter = ",")
    print("Codebook loaded in codebook64.csv, press any key to exit ...")
    cv2.waitKey()

if __name__ == '__main__':
    test_codebook()