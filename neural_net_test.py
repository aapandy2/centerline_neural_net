#import mnist_loader
import network2
import numpy as np
import scipy.io
from PIL import Image, ImageDraw

#load in .mat file as python dictionary
mat = scipy.io.loadmat('/tigress/LEIFER/PanNeuronal/20180129/BrainScanner20180129_094932/BehaviorAnalysis/centerline.mat')


def load_images(num_images_to_load, first_image_index, step):

#	num_images_to_load = 2000
#	step = 1
	for j in range(0, num_images_to_load, step):
		#load image file
		if(j % 500 == 0):
			print 'loading image ', first_image_index + j, ' of ', first_image_index + num_images_to_load
		frame_number = j + first_image_index
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
			image_data = ((img_flattened, coords_flattened),)
		else:
	                image_data = image_data + ((img_flattened, coords_flattened),)
	image_data = list(image_data)
	return image_data

#make training_data into a list
#training_data = list(training_data)
num_images_to_load = 2000
step = 2
first_image_index = 0
training_data = load_images(num_images_to_load, first_image_index, step)

net = network2.Network([300*300, 20, 20, 10], cost=network2.CrossEntropyCost)
net.SGD(training_data, 10, 1, 0.07, monitor_training_cost=True)

#generate test data
num_test_images = 2000
test_step = 2
test_data = load_images(num_test_images, 1, test_step)

for k in range(5):
#	test_image = int(np.random.rand(1) * (np.size(training_data)-1))
	test_image = np.random.randint(low=0, high= int(num_test_images/test_step) )
	print test_image
	
	#retrieve image from array
#	img = Image.fromarray(np.reshape(training_data[test_image][0], (300, 300)) * 255)
	img = Image.fromarray(np.reshape(test_data[test_image][0], (300, 300)) * 255).convert('RGBA')
	draw = ImageDraw.Draw(img)
	
	#retrieve centerline from array
#	coords_tuple = tuple(map(tuple, np.reshape(np.array(net.feedforward(training_data[test_image][0])) * 300., (5, 2))))
	net_coords_tuple = tuple(map(tuple, np.reshape(np.array(net.feedforward(test_data[test_image][0])) * 300., (5, 2))))
	true_coords_tuple = tuple(map(tuple, np.reshape(np.array(test_data[test_image][1])*300., (5, 2))))

	draw.line(net_coords_tuple, fill=(0, 0, 255), width=2)
	draw.line(true_coords_tuple, fill=(255, 0, 0), width=2)
	img.show()
