import cv2
import numpy as np
import time
import os

# Local dependencies
from dataset import Dataset
import descriptors
import constants
import utils
import filenames
from log import Log


def main(is_interactive=True, k_opt=64, des_opt=constants.ORB_FEAT_OPTION):
    if not is_interactive:
        experiment_start = time.time()
    # Check for the dataset of images
    if not os.path.exists(constants.DATASET_PATH):
        print("Dataset not found, please copy one.")
        return
    dataset = Dataset(constants.DATASET_PATH)
    dataset.generate_sets()

    # Check for the directory where stores generated files
    if not os.path.exists(constants.FILES_DIR_NAME):
        os.makedirs(constants.FILES_DIR_NAME)

    if is_interactive:
        des_option = input("Enter [1] for using ORB features or [2] to use SIFT features.\n")
    else:
        des_option = des_opt
    des_name = constants.ORB_FEAT_NAME if des_option == constants.ORB_FEAT_OPTION else constants.SIFT_FEAT_NAME

    if is_interactive:
        k = input("Enter the number of cluster centers you want for the codebook.\n")
    else:
        k = k_opt

    log = Log(k, des_name)

    codebook_filename = filenames.codebook(k, des_name)
    if is_interactive:
        codebook_option = input("Enter [1] for generating a new codebook or [2] to load one.\n")
    else:
        codebook_option = constants.GENERATE_OPTION
    if codebook_option == constants.GENERATE_OPTION:
        # Calculate all the training descriptors to generate the codebook
        start = time.time()
        des = descriptors.all_descriptors(dataset, dataset.get_train_set(), des_option)
        end = time.time()
        log.train_des_time(end - start)
        # Generates the codebook using K Means
        print("Generating a codebook using K-Means with k={0}".format(k))
        start = time.time()
        codebook = descriptors.gen_codebook(dataset, des, k)
        end = time.time()
        log.codebook_time(end - start)
        # Stores the codebook in a file
        utils.save(codebook_filename, codebook)
        print("Codebook saved in {0}".format(codebook_filename))
    else:
        # Load a codebook from a file
        print("Loading codebook ...")
        codebook = utils.load(codebook_filename)
        print("Codebook with shape = {0} loaded.".format(codebook.shape))

    # Train and test the dataset
    svm = train(dataset.get_train_set(), codebook, log, des_option=des_option, is_interactive=is_interactive)
    result, labels = test(
        dataset.get_test_set(), codebook, svm, log, des_option=des_option, is_interactive=is_interactive
    )

    # Store the results from the test
    result_filename = filenames.result(k, des_name)
    labels_filename = filenames.labels(k, des_name)
    utils.save(result_filename, result)
    utils.save(labels_filename, labels)

    classes = dataset.get_classes()
    log.classes(classes)
    log.classes_counts(dataset.get_classes_counts())


    # Create a confusion matrix
    confusion_matrix = np.zeros((len(classes), len(classes)), dtype=np.uint32)
    for i in range(len(result)):
        predicted_id = int(result[i])
        real_id = int(labels[i])
        confusion_matrix[real_id][predicted_id] += 1

    print("Confusion Matrix =\n{0}".format(confusion_matrix))
    log.confusion_matrix(confusion_matrix)
    log.save()
    print("Log saved on {0}.".format(filenames.log(k, des_name)))
    if not is_interactive:
        experiment_end = time.time()
        elapsed_time = utils.humanize_time(experiment_end - experiment_start)
        print("Total time during the experiment was {0}".format(elapsed_time))
    else:
        # Show a plot of the confusion matrix on interactive mode
        utils.show_conf_mat(confusion_matrix)
        raw_input("Press [Enter] to exit ...")

def train(train_set, codebook, log, des_option=constants.ORB_FEAT_OPTION, is_interactive=True):
    """
    Gets the descriptors for the training set and then calculates the SVM for them.

    Args:
        train_set (list of string arrays): Each element has all the image paths for it corresponding class.
        codebook (numpy matrix): Each row is a center of a codebook of Bag of Words approach.
        log (Log): It helps to store the information about the time that each part take.
        des_option (integer): The option of the feature that is going to be used as local descriptor.
        is_interactive (boolean): If it is the user can choose to load files or generate.

    Returns:
        cv2.SVM: The Support Vector Machine obtained in the training phase.
    """
    des_name = constants.ORB_FEAT_NAME if des_option == constants.ORB_FEAT_OPTION else constants.SIFT_FEAT_NAME
    k = len(codebook)
    X_filename = filenames.X_train(k, des_name)
    y_filename = filenames.y_train(k, des_name)
    if is_interactive:
        data_option = input("Enter [1] to calculate VLAD vectors for the training set or [2] to load them.\n")
    else:
        data_option = constants.GENERATE_OPTION
    if data_option == constants.GENERATE_OPTION:
        # Getting the global vectors for all of the training set
        print("Getting global descriptors for the training set.")
        start = time.time()
        X, y = get_data_and_labels(train_set, codebook, des_option)
        utils.save(X_filename, X)
        utils.save(y_filename, y)
        end = time.time()
        log.train_vlad_time(end - start)
    else:
        # Loading the global vectors for all of the training set
        print("Loading global descriptors for the training set.")
        X = utils.load(X_filename)
        y = utils.load(y_filename)
        X = np.matrix(X, dtype=np.float32)
    svm = cv2.SVM()
    svm_filename = filenames.svm(k, des_name)
    if is_interactive:
        svm_option = input("Enter [1] for generating a SVM or [2] to load one\n")
    else:
        svm_option = constants.GENERATE_OPTION
    if svm_option == constants.GENERATE_OPTION:
        # Calculating the Support Vector Machine for the training set
        print("Calculating the Support Vector Machine for the training set...")
        svm_params = dict(kernel_type=cv2.SVM_LINEAR, svm_type=cv2.SVM_C_SVC, C=1)
        start = time.time()
        svm.train_auto(X, y, None, None, svm_params)
        end = time.time()
        log.svm_time(end - start)
        # Storing the SVM in a file
        svm.save(svm_filename)
    else:
        svm.load(svm_filename)
    return svm

def test(test_set, codebook, svm, log, des_option = constants.ORB_FEAT_OPTION, is_interactive=True):
    """
    Gets the descriptors for the testing set and use the svm given as a parameter to predict all the elements

    Args:
        test_set (list of string arrays): Each element has all the image paths for its corresponding class.
        svm (cv2.SVM): The Support Vector Machine obtained in the training phase.
        log (Log): It helps to store the information about the time that each part take.
        des_option (integer): The option of the feature that is going to be used as local descriptor.
        is_interactive (boolean): If it is the user can choose to load files or generate.

    Returns:
        numpy float array: The result of the predictions made.
        numpy float array: The real labels for the testing set.
    """
    des_name = constants.ORB_FEAT_NAME if des_option == constants.ORB_FEAT_OPTION else constants.SIFT_FEAT_NAME
    k = len(codebook)
    X_filename = filenames.X_test(k, des_name)
    y_filename = filenames.y_test(k, des_name)
    if is_interactive:
        data_option = input("Enter [1] to calculate VLAD vectors for the testing set or [2] to load them.\n")
    else:
        data_option = constants.GENERATE_OPTION
    if data_option == constants.GENERATE_OPTION:
        # Getting the global vectors for all of the testing set
        print("Getting global descriptors for the testing set...")
        start = time.time()
        X, y = get_data_and_labels(test_set, codebook, des_option)
        utils.save(X_filename, X)
        utils.save(y_filename, y)
        end = time.time()
        log.test_vlad_time(end - start)
    else:
        # Loading the global vectors for all of the testing set
        print("Loading global descriptors for the testing set.")
        X = utils.load(X_filename.format(des_name))
        y = utils.load(y_filename.format(des_name))
        X = np.matrix(X, dtype=np.float32)
    # Predicting the testing set using the SVM
    start = time.time()
    result = svm.predict_all(X)
    end = time.time()
    log.predict_time(end - start)
    mask = result == y
    correct = np.count_nonzero(mask)
    accuracy = (correct * 100.0 / result.size)
    log.accuracy(accuracy)
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
        step = round(constants.STEP_PERCENTAGE * len(img_paths) / 100)
        for i in range(len(img_paths)):
            if i % step == 0:
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