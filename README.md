# object-classification
Python and OpenCV code for object classification using images.
The program calculates local descriptors on images (it can be ORB or SIFT) and then gets a codebook for the training set using K-Means.
After that it uses the codebook to calculate VLAD (Vector of Locally Aggregated Descriptors) as a global descriptor for each
of the images. This vectors are used on a SVM (Support Vector Machine) in the training phase. In the testing phase for each
testing image its VLAD has to be calculated and the class is determined using that vector in the SVM. The program gives the
accuracy of the predictions for the testing phase.

I used this for a research project, you can see more of it here http://aggiecv.blogspot.com/

## Usage
To run the main program run ``python main.py``

The images have to be in a folder inside this project with the name "dataset". The images must be separeted between classes
in different folders. For example inside "dataset" can be the folders "lion", "elephant" and "monkey" each one with its
corresponding images.

The images have to be separated between training and testing in different folders, "train" and "test" inside the folder for the
class. For example, inside the folder "lion" there will be the folder "train" which contains the training set for the lion images
and the folder "test" which contains the testing images of lions. After a Dataset object is generated and separated the images
it is stored in a file using pickle. You should only generate one Dataset object and then reuse it to have consistent experiments.

When the program already selected the images it will ask you to decide the kind of features to use. It supports ORB and SIFT
features. Then it gets the descriptors of all the training images to create a codebook using K-Means. You can calculate the
codebook or use another previously calculated. After that it gets VLAD global descriptors for every training image, they can
be calculated or loaded if they were previously calculated. Using that information it generates a SVM, if there is one previously
calculated it can be loaded too. The next step is to get VLAD vectors for the testing set, they can be calculated or loaded.
When all the testing vectors are ready it uses them as data in the Support Vector Machine to predict their classes. Finally
the program show the accuracy obtained.

## Dependencies
Used with OpenCV 2.4 and Python 2.7. Python libraries required are scipy, numpy and matplotlib.
