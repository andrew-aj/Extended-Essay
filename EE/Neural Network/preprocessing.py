import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

x = Image.open('test.png','r')#.resize((28,28))
x = x.convert('L') 
y = np.asarray(x.getdata(), dtype=np.float64)#.reshape((x.size[1],x.size[0]))
#to have just a list of pixed, comment out .reshape

y = np.asarray(y,dtype=np.uint8)
with np.nditer(y, op_flags=['readwrite']) as it:
	for x in it:
		if(x[...] > 128):
			x[...] = 1
			#in reality will be set to 1 to take up less memory
			#only used 255 here so i could visualize it
		else:
			x[...] = 0

#to stop conversion to image comment out the next two lines
#w = y.reshape(28,28)
#w = Image.fromarray(w,mode='L')
#w.save('out.jpg')
#a = np.reshape(y, (28,28))
#plt.imshow(a, cmap='gray')
#plt.show()
np.savetxt("foo.csv", y.reshape(1,y.shape[0]),fmt='%i', delimiter=",")