#import mnist_loader
import network2
import numpy as np
import scipy.io
from PIL import Image, ImageDraw

#load in .mat file as python dictionary
mat = scipy.io.loadmat('/tigress/LEIFER/PanNeuronal/20180129/BrainScanner20180129_094932/BehaviorAnalysis/centerline.mat')

num_images_to_load = 2000
for j in range(0, num_images_to_load):
	#load image file
	if(j % 500 == 0):
		print 'loading image ', j, ' of ', num_images_to_load
	frame_number = j
	frames_directory = '/tigress/LEIFER/PanNeuronal/20180129/BrainScanner20180129_094932/LowMagBrain20180129_094932/cam1_frames_30Hz_scaled/'
	frame = frames_directory + 'cam1_frames_' + '%05d'%frame_number + '_scaled.png'
	
	#NOTE: i think PIL makes the origin of the coordinate system the upper-left corner
	#      below I had to switch x and y to get everything to work.
	for i in range(100):
	        #define coords list of two-tuples (x, y) coordinates
	        if(i == 0):
	                coords = ((mat['centerline'][i][1][frame_number], mat['centerline'][i][0][frame_number]),)
		else:
		        coords = coords + ((mat['centerline'][i][1][frame_number], mat['centerline'][i][0][frame_number]),)
	
	#img = Image.open(frame)
	img_original_dim = 1088 #image starts 1088 x 1088 pixels
	img_scaled_dim  = 300 #scale image to 150 x 150 pixels
	img_scaling_factor = float(img_scaled_dim)/float(img_original_dim)
	
	coords_np = np.array(coords) #make numpy array version of coords
	coords_np = coords_np * img_scaling_factor #scale coords
	coords_np = tuple(map(tuple, coords_np)) #make numpy array back into tuple so it works with draw.line()
	
	#load image and convert to array, flatten array, scale to be in range [0, 1]
	img_loaded = Image.open(frame)
	coords_flattened = np.array(coords_np[1::20]).flatten() / 300.
	img_flattened    = np.array(img_loaded).flatten() / 255.

	coords_flattened = np.reshape(coords_flattened, (10, 1)) #reshape to match net input format
	img_flattened    = np.reshape(img_flattened, (90000, 1)) #reshape to match net input format
	
	if(j == 0):
		training_data = ((img_flattened, coords_flattened),)
	else:
                training_data = training_data + ((img_flattened, coords_flattened),)
#	elif(j == num_images_to_load/2):
#		test_data = training_data + ((img_flattened, coords_flattened),)
#	elif(j > num_images_to_load/2):
#		test_data = test_data + ((img_flattened, coords_flattened),)

#training_data1, validation_data1, test_data1 = mnist_loader.load_data_wrapper()

#make training_data into a list
#training_data = list(training_data)

training_data = list(training_data)

#print np.shape(training_data), np.shape(training_data1)

net = network2.Network([300*300, 10, 10, 10], cost=network2.CrossEntropyCost)
net.SGD(training_data, 5, 1, 0.07, monitor_training_cost=True)
#net.SGD(training_data, 10, 50, 1.0, test_data=test_data)

for k in range(5):
	test_image = int(np.random.rand(1) * np.size(training_data))
	print test_image
	
	#retrieve image from array
	img = Image.fromarray(np.reshape(training_data[test_image][0], (300, 300)) * 255)
	draw = ImageDraw.Draw(img)
	
	#retrieve centerline from array
	coords_tuple = tuple(map(tuple, np.reshape(np.array(net.feedforward(training_data[test_image][0])) * 300., (5, 2))))
       #print coords_tuple
	draw.line(coords_tuple, fill=100, width=2)
	img.show()
