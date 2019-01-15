import numpy as np
from PIL import Image

x = Image.open('main.jpg','r').resize((32,32))
x = x.convert('L') 
y = np.asarray(x.getdata(), dtype=np.float64).reshape((x.size[1],x.size[0]))
#to have just a list of pixed, comment out .reshape

y = np.asarray(y,dtype=np.uint8)
with np.nditer(y, op_flags=['readwrite']) as it:
	for x in it:
		if(x[...] > 128):
			x[...] = 255
			#in reality will be set to 1 to take up less memory
			#only used 255 here so i could visualize it
		else:
			x[...] = 0

#to stop conversion to image comment out the next two lines
w = Image.fromarray(y,mode='L')
w.save('out.jpg')
np.savetxt("foo.csv", y,fmt='%i')#, delimiter=",")