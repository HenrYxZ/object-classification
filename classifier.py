import cv2
import numpy as np
import time


# Local dependencies

import constants
import descriptors
import filenames
import utils


class Classifier:
    """
    Class for making training and testing in image classification.
    """
    def __init__(self, dataset, log):
        """
        Initialize the classifier object.
        Args:
            dataset (Dataset): The object that stores the information about the dataset.
            log (Log): The object that stores the information about the times and the results of the process.

        Returns:
            void
        """
        self.dataset = dataset
        self.log = log

    def train(self, svm_kernel, codebook, des_option=constants.ORB_FEAT_OPTION, is_interactive=True):
        """
        Gets the descriptors for the training set and then calculates the SVM for them.

        Args:
            svm_kernel (constant): The kernel of the SVM that will be created.
            codebook (NumPy float matrix): Each row is a center of a codebook of Bag of Words approach.
            des_option (integer): The option of the feature that is going to be used as local descriptor.
            is_interactive (boolean): If it is the user can choose to load files or generate.

        Returns:
            cv2.SVM: The Support Vector Machine obtained in the training phase.
        """
        des_name = constants.ORB_FEAT_NAME if des_option == constants.ORB_FEAT_OPTION else constants.SIFT_FEAT_NAME
        k = len(codebook)
        x_filename = filenames.vlads_train(k, des_name)
        if is_interactive:
            data_option = input("Enter [1] to calculate VLAD vectors for the training set or [2] to load them.\n")
        else:
            data_option = constants.GENERATE_OPTION
        if data_option == constants.GENERATE_OPTION:
            # Getting the global vectors for all of the training set
            print("Getting global descriptors for the training set.")
            start = time.time()
            x, y = self.get_data_and_labels(self.dataset.get_train_set(), codebook, des_option)
            utils.save(x_filename, x)
            end = time.time()
            print("VLADs training vectors saved on file {0}".format(x_filename))
            self.log.train_vlad_time(end - start)
        else:
            # Loading the global vectors for all of the training set
            print("Loading global descriptors for the training set.")
            x = utils.load(x_filename)
            y = self.dataset.get_train_y()
            x = np.matrix(x, dtype=np.float32)
        svm = cv2.SVM()
        svm_filename = filenames.svm(k, des_name, svm_kernel)
        if is_interactive:
            svm_option = input("Enter [1] for generating a SVM or [2] to load one\n")
        else:
            svm_option = constants.GENERATE_OPTION
        if svm_option == constants.GENERATE_OPTION:
            # Calculating the Support Vector Machine for the training set
            print("Calculating the Support Vector Machine for the training set...")
            svm_params = dict(kernel_type=svm_kernel, svm_type=cv2.SVM_C_SVC, C=1)
            start = time.time()
            svm.train_auto(x, y, None, None, svm_params)
            end = time.time()
            self.log.svm_time(end - start)
            # Storing the SVM in a file
            svm.save(svm_filename)
        else:
            svm.load(svm_filename)
        return svm

    def test(self, codebook, svm, des_option = constants.ORB_FEAT_OPTION, is_interactive=True):
        """
        Gets the descriptors for the testing set and use the svm given as a parameter to predict all the elements

        Args:
            codebook (NumPy matrix): Each row is a center of a codebook of Bag of Words approach.
            svm (cv2.SVM): The Support Vector Machine obtained in the training phase.
            des_option (integer): The option of the feature that is going to be used as local descriptor.
            is_interactive (boolean): If it is the user can choose to load files or generate.

        Returns:
            NumPy float array: The result of the predictions made.
            NumPy float array: The real labels for the testing set.
        """
        des_name = constants.ORB_FEAT_NAME if des_option == constants.ORB_FEAT_OPTION else constants.SIFT_FEAT_NAME
        k = len(codebook)
        x_filename = filenames.vlads_test(k, des_name)
        if is_interactive:
            data_option = input("Enter [1] to calculate VLAD vectors for the testing set or [2] to load them.\n")
        else:
            data_option = constants.GENERATE_OPTION
        if data_option == constants.GENERATE_OPTION:
            # Getting the global vectors for all of the testing set
            print("Getting global descriptors for the testing set...")
            start = time.time()
            x, y = self.get_data_and_labels(self.dataset.get_test_set(), codebook, des_option)
            utils.save(x_filename, x)
            end = time.time()
            print("VLADs testing vectors saved on file {0}".format(x_filename))
            self.log.test_vlad_time(end - start)
        else:
            # Loading the global vectors for all of the testing set
            print("Loading global descriptors for the testing set.")
            x = utils.load(x_filename.format(des_name))
            y = self.dataset.get_test_y()
            x = np.matrix(x, dtype=np.float32)
        # Predicting the testing set using the SVM
        start = time.time()
        result = svm.predict_all(x)
        end = time.time()
        self.log.predict_time(end - start)
        mask = result == y
        correct = np.count_nonzero(mask)
        accuracy = (correct * 100.0 / result.size)
        self.log.accuracy(accuracy)
        return result, y

    def get_data_and_labels(self, img_set, codebook, des_option = constants.ORB_FEAT_OPTION):
        """
        Calculates all the local descriptors for an image set and then uses a codebook to calculate the VLAD global
        descriptor for each image and stores the label with the class of the image.
        Args:
            img_set (string array): The list of image paths for the set.
            codebook (numpy float matrix): Each row is a center and each column is a dimension of the centers.
            des_option (integer): The option of the feature that is going to be used as local descriptor.

        Returns:
            NumPy float matrix: Each row is the global descriptor of an image and each column is a dimension.
            NumPy float array: Each element is the number of the class for the corresponding image.
        """
        y = []
        x = None
        for class_number in range(len(img_set)):
            img_paths = img_set[class_number]
            step = round(constants.STEP_PERCENTAGE * len(img_paths) / 100)
            for i in range(len(img_paths)):
                if (step > 0) and (i % step == 0):
                    percentage = (100 * i) / len(img_paths)
                    print("Calculating global descriptors for image number {0} of {1}({2}%)".format(
                        i, len(img_paths), percentage)
                    )
                img = cv2.imread(img_paths[i])
                if des_option == constants.ORB_FEAT_OPTION:
                    des = descriptors.orb(img)
                else:
                    des = descriptors.sift(img)
                if des is not None:
                    des = np.array(des, dtype=np.float32)
                    vlad_vector = descriptors.vlad(des, codebook)
                    if x is None:
                        x = vlad_vector
                        y.append(class_number)
                    else:
                        x = np.vstack((x, vlad_vector))
                        y.append(class_number)
                else:
                    print("Img with None descriptor: {0}".format(img_paths[i]))
        y = np.float32(y)[:, np.newaxis]
        x = np.matrix(x)
        return x, y