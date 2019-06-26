import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from sklearn import svm
import threading
start_time = time.time()
np.set_printoptions(suppress=True)

def retrieveFiles(dataPath, dataTrainFile, labelTrainFile, dataTestFile, labelTestFile):
    train_imgs = pd.read_csv(dataPath + dataTrainFile, header=None, dtype = np.int32)
    df1 = pd.DataFrame(train_imgs)
    train_imgs = df1.values.tolist()
    train_labels1 = pd.read_csv(dataPath + labelTrainFile, header=None, dtype=np.int32)
    df2 = pd.DataFrame(train_labels1)
    train_labels1 = df2.values.tolist()
    train_labels = []
    for i in range(len(train_labels1)):
        train_labels.append(train_labels1[i][0])
    test_imgs = pd.read_csv(dataPath + dataTestFile, header=None, dtype = np.int32)
    df3 = pd.DataFrame(test_imgs)
    test_imgs = df3.values.tolist()
    test_labels = pd.read_csv(dataPath + labelTestFile, header=None, dtype = np.int32)
    df4 = pd.DataFrame(test_labels)
    test_labels = df4.values.tolist()
    return train_imgs, train_labels, test_imgs, test_labels

@profile
def runMain(range1, range2):
    test = pd.read_csv("foo.csv", header=None, dtype = np.int32)
    df1 = pd.DataFrame(test)
    test = df1.values.tolist()
    train_imgs, train_labels, test_imgs, test_labels = retrieveFiles("data/mnist/", "test.csv", "labels.csv", "newdatatest.csv", "testLabels.csv")
    print("Done loading images, beginning classification")
    classifier = svm.SVC(gamma = 'auto',verbose=1)
    classifier.fit(train_imgs[range1:range2][:], train_labels[range1:range2])
    print(classifier.score(test_imgs[:][:], test_labels[:]))
    elapsed_time = time.time() - start_time
    print(elapsed_time)

runMain(0,60000)