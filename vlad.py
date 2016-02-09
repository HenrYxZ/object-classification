import numpy as np
import cv2

# Local dependencies
import utils

def vlad(descriptors, centers):
    """
    Calculate the Vector of Locally Aggregated Descriptors (VLAD) which is a global descriptor from a group of
    descriptors and centers that are codewords of a codebook, obtained for example with K-Means.

    Args:
        descriptors (numpy float matrix): The local descriptors.
        centers (numpy float matrix): The centers are points representatives of the classes.

    Returns:
        numpy float array: The VLAD vector.
    """
    dimensions = len(descriptors[0])
    vlad_vector = np.zeros((len(centers), dimensions), dtype=np.float32)
    for descriptor in descriptors:
        nearest_center, center_idx = utils.find_nn(descriptor, centers)
        for i in range(dimensions):
            vlad_vector[center_idx][i] += (descriptor[i] - nearest_center[i])
    # L2 Normalization
    vlad_vector = cv2.normalize(vlad_vector)
    vlad_vector = vlad_vector.flatten()
    return vlad_vector
