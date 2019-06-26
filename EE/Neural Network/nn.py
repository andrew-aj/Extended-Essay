import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
#from keras.layers import Input, Dense
#from keras.models import Model
#import keras
#from keras.layers import Dense
#from keras.models import Sequential
import tensorflow as tf
from tensorflow import keras
start_time = time.time()
np.set_printoptions(suppress=True)

@profile
def retrieveFiles(dataPath, dataTrainFile, labelTrainFile, dataTestFile, labelTestFile):
	train_imgs = pd.read_csv(dataPath + dataTrainFile, header=None, dtype = np.int32).to_numpy()
	train_labels = pd.read_csv(dataPath + labelTrainFile, header=None, dtype=np.int32).to_numpy()
	train_imgs = np.asfarray(train_imgs, dtype=np.int32)
	train_labels = np.asfarray(train_labels, dtype=np.int32).reshape(-1)
	test_imgs = pd.read_csv(dataPath + dataTestFile, header=None, dtype = np.int32).to_numpy()
	test_labels = pd.read_csv(dataPath + labelTestFile, header=None, dtype=np.int32).to_numpy()
	test_imgs = np.asfarray(test_imgs, dtype=np.int32)
	test_labels = np.asfarray(test_labels, dtype=np.int32).reshape(-1)
	return train_imgs, train_labels, test_imgs, test_labels

def createModel():
    model = keras.Sequential()
    model.add(keras.layers.Dense(units=2500,input_shape=(784,)))
    model.add(keras.layers.Dropout(.2))
    model.add(keras.layers.Dense(units=2000,activation="relu"))
    model.add(keras.layers.Dropout(.2))
    model.add(keras.layers.Dense(units=1500,activation="relu"))
    model.add(keras.layers.Dropout(.2))
    model.add(keras.layers.Dense(units=1000,activation="relu"))
    model.add(keras.layers.Dropout(.2))
    model.add(keras.layers.Dense(units=500,activation="relu"))
    model.add(keras.layers.Dropout(.2))
    model.add(keras.layers.Dense(10,activation="softmax"))
    return model

"""
def createModel():
    model = Sequential()
    model.add(Dense(units=128,input_shape=(784,)))
    model.add(Dense(units=128,activation="relu"))
    model.add(Dense(units=128,activation="relu"))
    model.add(Dense(10,activation="softmax"))
    return model
"""
@profile
def trainModel(train_imgs, train_labels, test_imgs, test_labels, model):
    model.compile(optimizer='adamax', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit(train_imgs,train_labels, epochs=5, validation_data=(test_imgs,test_labels))
    #print(hist.history)
    #model.summary()

def testModel(test_imgs, test_labels, start_time, test, model):
    test_loss, test_acc = model.evaluate(test_imgs, test_labels)
    print('Test accuracy:', test_acc)
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    print(model.predict_classes(test[0,:].reshape(1,784)))
    print(model.predict(test[0,:].reshape(1,784)))
    a = np.reshape(test, (28,28))
    plt.imshow(a, cmap='gray')
    plt.show()

@profile
def doModel(train_imgs, train_labels, start_time, test, test_imgs, test_labels):
    model = createModel()
    trainModel(train_imgs, train_labels, test_imgs, test_labels, model)
    testModel(test_imgs, test_labels, start_time, test, model)    

test = pd.read_csv("foo.csv", header=None, dtype = np.int32).to_numpy()
test = np.asfarray(test, dtype=np.int32)
train_imgs, train_labels, test_imgs, test_labels = retrieveFiles("data/mnist/", "test.csv", "labels.csv", "newdatatest.csv", "testLabels.csv")

doModel(train_imgs, train_labels, start_time, test, test_imgs, test_labels)