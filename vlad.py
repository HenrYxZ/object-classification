import numpy as np

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
    vlad_vector = np.zeros(dimensions, dtype=np.float32)
    for descriptor in descriptors:
        nearest_center = utils.find_nn(descriptor, centers)
        for i in range(dimensions):
            vlad_vector[i] += (descriptor[i] - nearest_center[i])
    return vlad_vector
