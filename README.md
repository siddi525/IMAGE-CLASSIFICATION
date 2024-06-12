
INTRODUCTION:
The purpose of this report is to present the results and main strategies of an image classification project using Rice Image Dataset. The goal of the project was to achieve a high accuracy of over 90% in classifying rice image into different categories. The Rice Image Dataset, obtained from (www.kaggle.com), consists of a collection of images depicting various types of rice grains.


DATASET OVERVIEW:
The Rice Image Dataset comprises (75,000) images, divided into (5 classes) different classes representing different rice grain types. Each image in the dataset manually labeled with the corresponding rice grain category. The dataset is balanced, that it contains a relatively equal number of samples for each class. 5 different rice grains, they are: [Arborio, Basmati, Ipsala, Jasmine and Karacadag].


IMPORTING LIBRARIES:
To accomplish the image classification task on the Rice Image Dataset, several libraries were utilized. These libraries provided essential functionalities for data handling, model creation, training, and evaluation. The following libraries were imported:


PREPROCESSING:
Data Split: The dataset was split into training and validation sets using a split ratio of 0.2. This division ensured that the model was trained on a subset of the data and evaluated on unseen data for better generalization.
Image Resizing: The images were resized to a consistent size of [224x224] pixels to ensure uniformity and facilitate model training.
Data Augmentation: Data augmentation techniques were applied to the training set using the ‘ImageDataGenerator‘ from Keras. Techniques such as random rotations, flips, and shifts were utilized to artificially increase the size and diversity of the training data, improving the model's ability to generalize.
Normalization: The pixel values of the images were normalized to a range of [0, 1] by dividing them by 255. This normalization step ensured that the model could effectively learn from the data without being biased by varying pixel intensities.



MODEL ARCHITECTURE:
The image classification model used in this project is a Convolutional Neural Network (CNN). The first layer is a 2D convolutional layer with 32 filters, a kernel size of 3x3, and ‘relu’ activation. This layer extracts relevant features from the input images.
A max-pooling layer with a pool size of 2x2 and stride of 2x2 follows the convolutional layer. This layer reduces the spatial dimensions and retains the most important features. The flatten layer converts the output from the previous layer into a 1D vector, preparing it for the fully connected layers. Two dense layers were added to the model. The first dense layer consists of 40 units and ‘relu’ activation, serving as a hidden layer. The second dense layer, with 5 units and sigmoid activation, produces class probabilities for multi-label classification. A dropout layer with a dropout rate of 0.1 was introduced after the first dense layer. Dropout helps prevent overfitting by randomly disabling a fraction of the neurons during training.



TRAINING AND TESTING:
The model was trained using the training set and evaluated on the validation set by (‘ImageDataGenerator’). The training was performed using a batch size of [32] and an initial learning rate of [0.2]. To optimize the model's performance, a [Adam] optimizer was utilized. The number of epochs used for training was determined based on the validation curves and early stopping to prevent overfitting.


In conclusion, the CNN model trained on the Rice Image Dataset yielded promising results. By utilizing the Adam optimizer, categorical cross-entropy loss, and appropriate evaluation metrics, the model demonstrated high accuracy on both the training and validation sets. The successful classification of rice grain images using this model holds potential for various applications in the field of agriculture and rice crop analysis.

