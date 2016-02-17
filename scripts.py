import constants
import scipy.io as sio
import numpy as np
import cv2
import os

# Local dependencies
import filenames
import utils


def codebook_to_csv(k=128, des_name=constants.ORB_FEAT_NAME):
    if not os.path.exists(constants.FILES_DIR_NAME):
        os.makedirs(constants.FILES_DIR_NAME)
    codebook = utils.load(filenames.codebook(k, des_name))
    filename = "{0}/codebook_{1}_{2}.csv".format(constants.FILES_DIR_NAME, k, des_name)
    utils.save_csv(filename, codebook)
    print("Copied codebook into the file with name {0}. Press any key to exit...".format(filename))
    cv2.waitKey()

if __name__ == '__main__':
    codebook_to_csv()
