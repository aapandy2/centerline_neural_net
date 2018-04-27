import network2
import numpy as np
import scipy.io
from PIL import Image, ImageDraw

#load in .mat file as python dictionary
mat = scipy.io.loadmat('/tigress/LEIFER/PanNeuronal/20180129/BrainScanner20180129_094932/BehaviorAnalysis/centerline.mat')

#define a function which loads images and centerlines into an array;
#the array has shape (image_array, centerline_array), where both
#image_array and centerline_array are flattened (1D) arrays
def load_images(num_images_to_load, first_image_index, step):
	
	for j in range(0, num_images_to_load, step):
		
		#load image files and print progress through loop
		if(j % 500 == 0):
			print 'loading image ', first_image_index + j, ' of ', first_image_index + num_images_to_load
		frame_number = j + first_image_index
		frames_directory = '/tigress/LEIFER/PanNeuronal/20180129/BrainScanner20180129_094932/LowMagBrain20180129_094932/cam1_frames_30Hz_scaled/'
		frame = frames_directory + 'cam1_frames_' + '%05d'%frame_number + '_scaled.png'
		
		#PIL makes the origin of the coordinate system the upper-left 
		#corner, so we switch x and y to get everything to work.
		for i in range(100):
		        #define coords list of two-tuples (x, y) coordinates
		        if(i == 0):
		                coords = ((mat['centerline'][i][1][frame_number], mat['centerline'][i][0][frame_number]),)
			else:
			        coords = coords + ((mat['centerline'][i][1][frame_number], mat['centerline'][i][0][frame_number]),)
		
		#the image starts with dimensions 1088x1088; the 3 lines below
		#scales it to img_scaled_dim and calculates the scaling factor
		img_original_dim = 1088
		img_scaled_dim  = 300
		img_scaling_factor = float(img_scaled_dim)/float(img_original_dim)
	
		#the array of coordinates must be scaled to fit onto the scaled
		#image and made into a tuple of (x, y) tuples
		coords_np = np.array(coords)               
		coords_np = coords_np * img_scaling_factor 
		coords_np = tuple(map(tuple, coords_np))
		
		#load image and convert to array, flatten array, 
		#and scale both image and coords to be in range [0, 1]
		#(all neurons in net have range [0, 1])
		img_loaded = Image.open(frame)
		coords_flattened = np.array(coords_np[1::20]).flatten() / 300.
		img_flattened    = np.array(img_loaded).flatten() / 255.

		#reshape flattened arrays to be 1D columns (net input format)
		coords_flattened = np.reshape(coords_flattened, (10, 1)) 
		img_flattened    = np.reshape(img_flattened, (90000, 1))
		
		#the if-else statement below defines the output image_data
		#array and populates it with (img_flattened, coords_flattened)
		#tuples
		if(j == 0):
			image_data = ((img_flattened, coords_flattened),)
		else:
	                image_data = image_data + ((img_flattened, coords_flattened),)

	#image_data needs to be a list to input into network
	image_data = list(image_data)

	return image_data

#load num_images_to_load, stepping through by step, starting at 
#first_image_index
num_images_to_load = 2000
step = 2
first_image_index = 0
training_data = load_images(num_images_to_load, first_image_index, step)

#define net architecture and cost function
net = network2.Network([300*300, 40, 40, 10], cost=network2.CrossEntropyCost)

#train net using training_data
net.SGD(training_data, 50, 1, 0.065, monitor_training_cost=True)

#generate test data
num_test_images = 2000
test_step = 2
test_data = load_images(num_test_images, 1, test_step)

#take num_test_sample_images from test_data and show "true" centerline along
#with the net's predicted centerline
num_test_sample_images = 5
for k in range(num_test_sample_images):

	#choose random test image from test_data
	test_image = np.random.randint(low=0, high= int(num_test_images/test_step) )
	
	#retrieve image from array and convert it to color to draw colored centerline
	img = Image.fromarray(np.reshape(test_data[test_image][0], (300, 300)) * 255).convert('RGBA')
	draw = ImageDraw.Draw(img)
	
	#generate net centerline and retrieve "true" centerline from array 
	#and convert them to tuples of (x, y) tuples for draw.line() below
	net_coords_tuple = tuple(map(tuple, np.reshape(np.array(net.feedforward(test_data[test_image][0])) * 300., (5, 2))))
	true_coords_tuple = tuple(map(tuple, np.reshape(np.array(test_data[test_image][1])*300., (5, 2))))

	#draw net centerline and "true" centerline on test image and show image
	draw.line(net_coords_tuple, fill=(0, 0, 255), width=2)
	draw.line(true_coords_tuple, fill=(255, 0, 0), width=2)
	img.show()
