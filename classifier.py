import tensorflow as tf
from tensorflow.python.platform import gfile
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedShuffleSplit
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle
import collections
import itertools
import time
import os
import re

# what and where
inceptionV3_dir = 'graph'
images_dir = 'images2'

# inception-v3


def create_graph():
    
    with tf.io.gfile.GFile(os.path.join(inceptionV3_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def extract_features(list_images):
    
    nb_features = 2048
    features = np.empty((len(list_images), nb_features))
    labels = []
    
    create_graph()
    
    # pool_3:0: next-to-last layer containing 2048 float description of the image.
    # DecodeJpeg/contents:0:JPEG encoding of the image.
    
    with tf.compat.v1.Session() as sess:
        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        
        for ind, image in enumerate(list_images):
            imlabel = image.split('/')[1]
            
            # rough indication of progress
            if ind % 100 == 0:
                print('Processing', image, imlabel)
            if not gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)
                
            image_data = gfile.FastGFile(image, 'rb').read()
            predictions = sess.run(next_to_last_tensor, {'DecodeJpeg/contents:0': image_data})
            features[ind, :] = np.squeeze(predictions)
            labels.append(imlabel)
    
    return features, labels


# Graphics


def plot_features(feature_labels, t_sne_features):
    
    plt.figure(figsize=(9, 9), dpi=100)
    
    uniques = {x: labels.count(x) for x in feature_labels}
    od = collections.OrderedDict(sorted(uniques.items()))
    
    colors = itertools.cycle(["r", "b", "g", "c", "m", "y",
                              "slategray", "plum", "cornflowerblue",
                              "hotpink", "darkorange", "forestgreen",
                              "tan", "firebrick", "sandybrown"])
    n = 0
    for label in od:
        count = od[label]
        m = n + count
        plt.scatter(t_sne_features[n:m, 0], t_sne_features[n:m, 1], c=next(colors), s=10, edgecolors='none')
        c = (m + n) // 2
        plt.annotate(label, (t_sne_features[c, 0], t_sne_features[c, 1]))
        n = m
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, matrix_title):
    plt.figure(figsize=(20, 20), dpi=100)
    cf_matrix = confusion_matrix(y_true, y_pred)
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    x_axis_labels = np.arange(len(true_labels))
    y_axis_labels = np.arange(len(pred_labels))
    sns.heatmap(cf_matrix/np.sum(cf_matrix, axis = 0), annot=True, fmt='.2%', cmap='Blues')
    plt.title(matrix_title, fontsize=12)
    plt.xticks(x_axis_labels, true_labels, rotation=90)
    plt.yticks(y_axis_labels, pred_labels, rotation=0)
    plt.ylabel('True label', fontsize=10)
    plt.xlabel('Predicted label', fontsize=10)
    plt.show()

    


#classifier function to run classifier and output results


def run_classifier(clfr, x_train_data, y_train_data, x_test_data, y_test_data, acc_str, matrix_header_str):
    start_time = time.time()
    clfr.fit(x_train_data, y_train_data)
    y_pred = clfr.predict(x_test_data)
    print("%f seconds" % (time.time() - start_time))
    
    # confusion matrix 
    print(acc_str.format(accuracy_score(y_test_data, y_pred) * 100))
    plot_confusion_matrix(y_test_data, y_pred, matrix_header_str)


# get the images and the labels 
dir_list = [x[0] for x in os.walk(images_dir)]
dir_list = dir_list[1:]
list_images = []
for image_sub_dir in dir_list:
	sub_dir_images = [image_sub_dir + '/' + f for f in os.listdir(image_sub_dir) if re.search('jpg|JPG', f)]
	list_images.extend(sub_dir_images)

# extract features
features, labels = extract_features(list_images)




# Classification

tsne_features = TSNE().fit_transform(features)


plot_features(labels, tsne_features)

# training and test datasets
X_train, X_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size=0.2, random_state=42, stratify = labels)


# Support Vector Machine
print('Support Vector Machine starting ...')
cl = LinearSVC()
run_classifier(cl, X_train, y_train, X_test, y_test, "CNN-SVM Accuracy: {0:0.1f}%", "SVM Confusion matrix")

#Extra Trees
print('Extra Trees Classifier starting ...')
cl = ExtraTreesClassifier(n_jobs=1,  n_estimators=10, criterion='gini', min_samples_split=2,
                           max_features=50, max_depth=None, min_samples_leaf=1)
run_classifier(cl, X_train, y_train, X_test, y_test, "CNN-ET Accuracy: {0:0.1f}%", "Extra Trees Confusion matrix")

# Random Forest
print('Random Forest Classifier starting ...')
cl = RandomForestClassifier(n_jobs=1, criterion='entropy', n_estimators=10, min_samples_split=2)
run_classifier(cl, X_train, y_train, X_test, y_test, "CNN-RF Accuracy: {0:0.1f}%", "Random Forest Confusion matrix")

#knn
print('K-Nearest Neighbours Classifier starting ...')
cl = KNeighborsClassifier(n_neighbors=1, n_jobs=4)
run_classifier(cl, X_train, y_train, X_test, y_test, "CNN-KNN Accuracy: {0:0.1f}%",
               "K-Nearest Neighbor Confusion matrix")

#MyLittlePony
print('Multi-layer Perceptron Classifier starting ...')
clf = MLPClassifier()
run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-MLP Accuracy: {0:0.1f}%",
               "Multi-layer Perceptron Confusion matrix")


#Gaussian Naive Bayes Classifier
print('Gaussian Naive Bayes Classifier starting ...')
clf = GaussianNB()
run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-GNB Accuracy: {0:0.1f}%",
               "Gaussian Naive Bayes Confusion matrix")

#LDA
print('Linear Discriminant Analysis Classifier starting ...')
clf = LinearDiscriminantAnalysis()
run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-LDA Accuracy: {0:0.1f}%",
               "Linear Discriminant Analysis Confusion matrix")

#QDA
print('Quadratic Discriminant Analysis Classifier starting ...')
clf = QuadraticDiscriminantAnalysis()
run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-QDA Accuracy: {0:0.1f}%",
               "Quadratic Discriminant Analysis Confusion matrix")




