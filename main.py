import pickle
import cv2
import numpy as np
import time
import os

# Local dependencies
from dataset import Dataset
import vlad
import descriptors
import constants
import utils
import filenames

def main():
    if not os.path.exists("files"):
        os.makedirs("files")
    if not os.path.exists("dataset"):
        print("Dataset not found, please copy one.")
    # dataset_option = input("Enter [1] to generate a new dataset or [2] to load one.\n")
    dataset_option = 2
    if dataset_option == constants.GENERATE_OPTION:
        print("Generating a new dataset.")
        dataset = Dataset(constants.DATASET_PATH)
        dataset.generate_sets()
        pickle.dump(dataset, open(constants.DATASET_OBJ_FILENAME, "wb"), protocol=constants.PICKLE_PROTOCOL)
        dataset.store_listfile()
    else:
        print("Loading dataset.")
        dataset = pickle.load(open(constants.DATASET_OBJ_FILENAME, "rb"))
    des_option = input("Enter [1] for using ORB features or [2] to use SIFT features.\n")
    des_name = constants.ORB_FEAT_NAME if des_option == constants.ORB_FEAT_OPTION else constants.SIFT_FEAT_NAME
    k = input("Enter the number of cluster centers you want for the codebook.\n")
    codebook_filename = filenames.codebook(k, des_name)
    codebook_option = input("Enter [1] for generating a new codebook or [2] to load one.\n")
    if codebook_option == constants.GENERATE_OPTION:
        # Calculate all the training descriptors to generate the codebook
        start = time.time()
        des = descriptors.all_descriptors(dataset, dataset.get_train_set(), des_option)
        end = time.time()
        elapsed_time = utils.humanize_time(end - start)
        print("Elapsed time getting all the descriptors is {0}".format(elapsed_time))
        # Generates the codebook using K Means
        start = time.time()
        codebook = descriptors.gen_codebook(dataset, des, k)
        end = time.time()
        elapsed_time = utils.humanize_time(end - start)
        print("Elapsed time calculating the k means for the codebook is {0}".format(elapsed_time))
        # Stores the codebook in a file
        utils.save(codebook_filename, codebook)
        print("Codebook saved in {0}".format(codebook_filename))
    else:
        print("Loading codebook ...")
        codebook = utils.load(codebook_filename)
        print("Codebook with shape = {0} loaded.".format(codebook.shape))
    # Train and test the dataset
    svm = train(dataset.get_train_set(), codebook, des_option=des_option)
    result, labels = test(dataset.get_test_set(), codebook, svm, des_option=des_option)
    # Store the results from the test
    result_filename = filenames.result(k, des_name)
    labels_filename = filenames.labels(k, des_name)
    utils.save(result_filename, result)
    utils.save(labels_filename, labels)
    #TODO Show a confusion matrix

def train(train_set, codebook, des_option = constants.ORB_FEAT_OPTION):
    """
    Gets the descriptors for the training set and then calculates the SVM for them.

    Args:
        train_set (list of string arrays): Each element has all the image paths for it corresponding class.
        des_option (integer): The option of the feature that is going to be used as local descriptor.

    Returns:
        cv2.SVM: The Support Vector Machine obtained in the training phase.
    """
    des_name = constants.ORB_FEAT_NAME if des_option == constants.ORB_FEAT_OPTION else constants.SIFT_FEAT_NAME
    k = len(codebook)
    X_filename = filenames.X_train(k, des_name)
    y_filename = filenames.y_train(k, des_name)
    data_option = input("Enter [1] to calculate VLAD vectors for the training set or [2] to load them.\n")
    if data_option == constants.GENERATE_OPTION:
        # Getting the global vectors for all of the training set
        print("Getting global descriptors for the training set.")
        start = time.time()
        X, y = get_data_and_labels(train_set, codebook, des_option)
        utils.save(X_filename, X)
        utils.save(y_filename, y)
        end = time.time()
        elapsed_time = utils.humanize_time(end - start)
        print("Elapsed time calculating VLAD vectors for training set is {0}.".format(elapsed_time))
    else:
        # Loading the global vectors for all of the training set
        print("Loading global descriptors for the training set.")
        X = utils.load(X_filename)
        y = utils.load(y_filename)
        X = np.matrix(X, dtype=np.float32)
        y = np.float32(y)[:, np.newaxis]
    # Calculating the Support Vector Machine for the training set
    print("Calculating the Support Vector Machine for the training set...")
    svm_params = dict(kernel_type=cv2.SVM_LINEAR, svm_type=cv2.SVM_C_SVC, C=1)
    svm = cv2.SVM()
    start = time.time()
    #svm.train(X, y, params=svm_params)
    svm.train_auto(X, y, None, None, svm_params)
    end = time.time()
    elapsed_time = utils.humanize_time(end - start)
    print("Elapsed time calculating the SVM for the training set is {0}".format(elapsed_time))
    # Storing the SVM in a file
    k = len(codebook)
    svm_filename = filenames.svm(k, des_name)
    svm.save(svm_filename)
    return svm

def test(test_set, codebook, svm, des_option = constants.ORB_FEAT_OPTION):
    """
    Gets the descriptors for the testing set and use the svm given as a parameter to predict all the elements

    Args:
        test_set (list of string arrays): Each element has all the image paths for its corresponding class.
        svm (cv2.SVM): The Support Vector Machine obtained in the training phase.
        des_option (integer): The option of the feature that is going to be used as local descriptor.

    Returns:
        numpy float array: The result of the predictions made.
        numpy float array: The real labels for the testing set.
    """
    des_name = constants.ORB_FEAT_NAME if des_option == constants.ORB_FEAT_OPTION else constants.SIFT_FEAT_NAME
    k = len(codebook)
    X_filename = filenames.X_test(k, des_name)
    y_filename = filenames.y_test(k, des_name)
    data_option = input("Enter [1] to calculate VLAD vectors for the testing set or [2] to load them.\n")
    if data_option == constants.GENERATE_OPTION:
        # Getting the global vectors for all of the testing set
        print("Getting global descriptors for the testing set...")
        start = time.time()
        X, y = get_data_and_labels(test_set, codebook, des_option)
        utils.save(X_filename, X)
        utils.save(y_filename, y)
        end = time.time()
        elapsed_time = utils.humanize_time(end - start)
        print("Elapsed time calculating VLAD vectors for testing set is {0}".format(elapsed_time))
    else:
        # Loading the global vectors for all of the testing set
        print("Loading global descriptors for the testing set.")
        X = utils.load(X_filename.format(des_name))
        y = utils.load(y_filename.format(des_name))
        X = np.matrix(X, dtype=np.float32)
        y = np.float32(y)[:, np.newaxis]
    # Predicting the testing set using the SVM
    start = time.time()
    result = svm.predict_all(X)
    end = time.time()
    elapsed_time = utils.humanize_time(end - start)
    print("Elapsed time predicting the testing set is {0}".format(elapsed_time))
    mask = result == y
    correct = np.count_nonzero(mask)
    accuracy = (correct * 100.0 / result.size)
    print("Accuracy: {0}".format(accuracy))
    return result, y

def get_data_and_labels(img_set, codebook, des_option = constants.ORB_FEAT_OPTION):
    """
    Calculates all the local descriptors for an image set and then uses a codebook to calculate the VLAD global
    descriptor for each image and stores the label with the class of the image.
    Args:
        img_set (string array): The list of image paths for the set.
        codebook (numpy float matrix): Each row is a center and each column is a dimension of the centers.
        des_option (integer): The option of the feature that is going to be used as local descriptor.

    Returns:
        numpy float matrix: Each row is the global descriptor of an image and each column is a dimension.
        numpy float array: Each element is the number of the class for the corresponding image.
    """
    y = []
    X = None
    for class_number in range(len(img_set)):
        img_paths = img_set[class_number]
        step = round(5 * len(img_paths) / 100)
        for i in range(len(img_paths)):
            if i % step == 0:
                percentage = (100 * i) / len(img_paths)
                print("Calculating descriptors for image number {0} of {1}({2}%)".format(i, len(img_paths), percentage))
            img = cv2.imread(img_paths[i])
            if des_option == constants.ORB_FEAT_OPTION:
                des = descriptors.orb(img)
            else:
                des = descriptors.sift(img)
            if des is not None:
                des = np.array(des, dtype=np.float32)
                vlad_vector = vlad.vlad(des, codebook)
                if X is None:
                    X = vlad_vector
                    y.append(class_number)
                else:
                    X = np.vstack((X, vlad_vector))
                    y.append(class_number)
            else:
                print("Img with None descriptor: {0}".format(img_paths[i]))
    y = np.float32(y)[:, np.newaxis]
    X = np.matrix(X)
    return X, y

if __name__ == '__main__':
    main()