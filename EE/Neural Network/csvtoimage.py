import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from PIL import Image
np.set_printoptions(suppress=True)
image_size = 28
no_of_different_labels = 10
image_pixels = image_size * image_size
data_path = "data/mnist/"

"""
test_data = pd.read_csv(data_path + "mnist_train.csv", delimiter=",", header=None)
test_data = test_data.to_numpy()
train_imgs = np.asfarray(test_data[:, 1:])
train_labels = np.asfarray(test_data[:, 0]) 


y = train_imgs
with np.nditer(y, op_flags=['readwrite']) as it:
	for x in it:
		if(x[...] > 128):
			x[...] = 1
			#in reality will be set to 1 to take up less memory
			#only used 255 here so i could visualize it
		else:
			x[...] = 0
"""


threshold_data = pd.read_csv(data_path + "postThresholdTest.csv", delimiter=",", header=None)
threshold_data = threshold_data.to_numpy()


#np.savetxt("data/mnist/postThresholdTest.csv", y, delimiter=",", fmt='%i')
#np.savetxt("data/mnist/testLabels.csv", train_labels, delimiter=",", fmt='%i')

#x = Image.open('out.jpg','r').resize((28,28))
#x = x.convert('L') 
#z = np.asarray(x.getdata(), dtype=np.float64) * 255
#b = np.reshape(z, (28,28))
#plt.imshow(b, cmap='gray')
#plt.show()

#thing = Image.open('test2.png', 'r')
#d = thing.resize((28,28))
#d.save('test3.png')

#a = np.reshape(y, (28,28))
#plt.imshow(a, cmap='gray')
#plt.show()

def createBoundingBox(img):
    xlist = []
    ylist = []
    right = 0
    left = 0
    bottom = 0
    top = 0
    for i in range(len(img)):
        for j in range(len(img[0])):
            if(img[i][j] == 1):
                ylist.append(i)
                xlist.append(j)
    right = max(xlist)+1
    bottom = max(ylist)+1
    left = min(xlist)
    top = min(ylist)
    #print(img.shape)
    #temp = img.tolist()
    imdone = Image.fromarray(img.astype('uint8'))
    imdone = imdone.crop((left,top,right,bottom))
    imdone = imdone.resize((28,28))
    imdone = np.array(imdone)
    return imdone

"""
thing = np.zeros((10000,784))



for x in range(10000):
    temp = threshold_data[x,:]
    temp = np.reshape(temp, (28,28))
    #temp = temp.tolist()
    im = createBoundingBox(temp)
    im = np.asarray(im)
    im = np.reshape(im, (1,784))
    thing[x,:] = im

    
np.savetxt(data_path + "newdatatest.csv", thing, delimiter=",", fmt='%i')
"""
thing = pd.read_csv("foo.csv", delimiter=",", header=None)
thing = thing.to_numpy()

#train_imgs = train_imgs.resize((28,28))
#createBoundingBox(train_imgs) 
##print(y.shape)
#y = np.reshape(y,(28,28))
#print(y)


im = createBoundingBox(thing.reshape(28,28))
plt.imshow(thing.reshape(28,28), cmap='gray')
plt.show()
plt.imshow(im, cmap='gray')
plt.show()
im = np.asarray(im)
im = np.reshape(im,(1,784))
#print(im)
np.savetxt("foo.csv", im, delimiter=",", fmt='%i')