# KaggleFace
ECE 50024 Mini Challenge

# FILES
test - Raw Testing Data (Provide this yourself)

train - Raw Training Data (Provide this yourself)

test_proc - Testing Images preprocessed by preproc_test.py (Requires testing image folder)

train_proc and val_proc - Training and Validation images preprocessed by preproc_train_val.py (Requires training image folder)

deploy.prototxt and res10 network - Both of these are used for the face detecting DNN from: https://github.com/gopinath-balu/computer_vision/tree/master/CAFFE_DNN

preproc_test.py and preproc_train_val.py preprocess the testing and training images, respectively.


category.csv - Contains mapping from celebrity name to numerical label
train.csv - Contains ground truth labels for training/validation data

training.py - Contains logic for training the prebuilt vgg16 neural network. Some code adapted from: https://www.kaggle.com/code/carloalbertobarbano/vgg16-transfer-learning-pytorch

big_classifier_new.pth - Contains the latest, most-accurate model. You may need to delete this before running your own training.py.

gen_csv.py - Generates submission CSV file
test_sub_new.csv - Best current submission, you may need to delete this before running your own gen_csv.py
