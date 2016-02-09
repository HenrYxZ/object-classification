# object-classification
Python and OpenCV code for object classification using images.
The program calculates local descriptors on images (it can be ORB or SIFT) and then gets a codebook for the training set using K-Means.
After that it uses the codebook to calculate VLAD (Vector of Locally Aggregated Descriptors) as a global descriptor for each
of the images. This vectors are used on a SVM (Support Vector Machine) in the training phase. In the testing phase for each
testing image its VLAD has to be calculated and the class is determined using that vector in the SVM. The program gives the
accuracy of the predictions for the testing phase.

## Usage
To run the main program run ``python main.py``

The images have to be in a folder inside this project with the name "dataset". The images must be separeted between classes
in different folders. For example inside "database" can be the folders "lion", "elephant" and "monkey" each one with its
corresponding images. The images don't have to be separeted between training and testing, the selection of that is done by
the class Dataset. The class will randomly select some images for training and other for testing, leaving 1/3 of the images
of each class for testing and the rest for training. After a Dataset object is generated and separeted the images it is stored
in a file using pickle. You should only generate one Dataset object and then reuse it to have consistent experiments.

When the program already selected the images it will ask you to decide the kind of features to use. It supports ORB and SIFT
features. Then it gets the descriptors of all the training images to create a codebook using K-Means. You can calculate the
codebook or use another previously calculated. After that it gets VLAD global descriptors for every training image, they can
be calculated or loaded if they were previously calculated. Using that information it generates a SVM, if there is one previously
calculated it can be loaded too. The next step is to get VLAD vectors for the testing set, they can be calculated or loaded.
When all the testing vectors are ready it uses them as data in the Support Vector Machine to predict their classes. Finally
the program show the accuracy obtained.

## Dependencies
Used with OpenCV 2.4 and Python 2.7. Python libraries required are scipy, numpy and matplotlib.
