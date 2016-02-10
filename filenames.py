import constants

def codebook(k, des_name):
    return "{0}/codebook_{1}_{2}.mat".format(constants.FILES_DIR_NAME, k, des_name)

def result(k, des_name):
    return "{0}/result_{1}_{2}.mat".format(constants.FILES_DIR_NAME, k, des_name)

def labels(k, des_name):
    return "{0}/labels_{1}_{2}.mat".format(constants.FILES_DIR_NAME, k, des_name)

def X_train(k, des_name):
    return "{0}/X_train_{1}_{2}.mat".format(constants.FILES_DIR_NAME, k, des_name)

def y_train(k, des_name):
    return "{0}/y_train_{1}_{2}.mat".format(constants.FILES_DIR_NAME, k, des_name)

def X_test(k, des_name):
    return "{0}/X_test_{1}_{2}.mat".format(constants.FILES_DIR_NAME, k, des_name)

def y_test(k, des_name):
    return "{0}/y_test_{1}_{2}.mat".format(constants.FILES_DIR_NAME, k, des_name)

def svm(k, des_name):
    return "{0}/svm_data_{1}_{2}.dat".format(constants.FILES_DIR_NAME, k, des_name)