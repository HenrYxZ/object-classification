import pickle
import cv2
import numpy as np
import time

# Local dependencies
from dataset import Dataset
import vlad
import descriptors
import constants
import utils

def main():
    dataset_option = input("Enter [1] to generate a new dataset or [2] to load one.\n")
    if dataset_option == constants.GENERATE_OPTION:
        print("Generating a new dataset.")
        dataset = Dataset(constants.DATASET_PATH)
        dataset.generate_sets()
        pickle.dump(dataset, open(constants.DATASET_OBJ_FILENAME, "wb"), protocol=constants.PICKLE_PROTOCOL)
    else:
        print("Loading dataset.")
        dataset = pickle.load(open(constants.DATASET_OBJ_FILENAME, "rb"))
    dataset.store_listfile()
    svm = train(dataset.get_train_set())
    test(dataset.get_test_set(), svm)

def train(train_set, des_option = constants.ORB_FEAT_OPTION):
    codebook = np.loadtxt(constants.CODEBOOK_FILE_NAME, delimiter=constants.NUMPY_DELIMITER)
    data_option = input("Enter [1] to calculate VLAD vectors for the training set or [2] to load them.\n")
    if data_option == constants.GENERATE_OPTION:
        # Getting the global vectors for all of the training set
        print("Getting global descriptors for the training set.")
        start = time.time()
        X, y = get_data_and_labels(train_set, codebook, des_option)
        np.savetxt("X_train.csv", X, delimiter=constants.NUMPY_DELIMITER)
        np.savetxt("y_train.csv", y, delimiter=constants.NUMPY_DELIMITER)
        end = time.time()
        elapsed_time = utils.humanize_time(end - start)
        print("Elapsed time calculating VLAD vectors for training set is {0}.".format(elapsed_time))
    else:
        print("Loading global descriptors for the training set.")
        X = np.loadtxt("X_train.csv", delimiter=constants.NUMPY_DELIMITER)
        X = np.matrix(X, dtype=np.float32)
        y = np.loadtxt("y_train.csv", delimiter=constants.NUMPY_DELIMITER)
        y = np.float32(y)[:, np.newaxis]
    # Calculating the Support Vector Machine for the training set
    print("Calculating the Support Vector Machine for the training set...")
    svm_params = dict(kernel_type=cv2.SVM_LINEAR, svm_type=cv2.SVM_C_SVC, C=2.67, gamma=5.383)
    svm = cv2.SVM()
    start = time.time()
    svm.train(X, y, params=svm_params)
    end = time.time()
    elapsed_time = utils.humanize_time(end - start)
    print("Elapsed time calculating the SVM for the training set is {0}".format(elapsed_time))
    # Storing the SVM in a file
    svm.save(constants.SVM_DATA_FILENAME)
    return svm

def test(test_set, svm, des_option = constants.ORB_FEAT_OPTION):
    codebook = np.loadtxt(constants.CODEBOOK_FILE_NAME, delimiter=constants.NUMPY_DELIMITER)
    # Getting the global vectors for all of the testing set
    print("Getting global descriptors for the testing set...")
    start = time.time()
    X, y = get_data_and_labels(test_set, codebook, des_option)
    np.savetxt("X_test.csv", X, delimiter=constants.NUMPY_DELIMITER)
    np.savetxt("y_test.csv", y, delimiter=constants.NUMPY_DELIMITER)
    end = time.time()
    elapsed_time = utils.humanize_time(end - start)
    print("Elapsed time calculating VLAD vectors for testing set is {0}".format(elapsed_time))
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

def get_data_and_labels(img_set, codebook, des_option = constants.ORB_FEAT_OPTION):
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