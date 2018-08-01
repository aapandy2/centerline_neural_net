# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 11:54:21 2018
try the skimage active contour fit on worm data.
@author: monika
"""

"""
====================
Active Contour Model
====================

The active contour model is a method to fit open or closed splines to lines or
edges in an image [1]_. It works by minimising an energy that is in part
defined by the image and part by the spline's shape: length and smoothness. The
minimization is done implicitly in the shape energy and explicitly in the
image energy.

In the following two examples the active contour model is used (1) to segment
the face of a person from the rest of an image by fitting a closed curve
to the edges of the face and (2) to find the darkest curve between two fixed
points while obeying smoothness considerations. Typically it is a good idea to
smooth images a bit before analyzing, as done in the following examples.

We initialize a circle around the astronaut's face and use the default boundary
condition ``bc='periodic'`` to fit a closed curve. The default parameters
``w_line=0, w_edge=1`` will make the curve search towards edges, such as the
boundaries of the face.

.. [1] *Snakes: Active contour models*. Kass, M.; Witkin, A.; Terzopoulos, D.
       International Journal of Computer Vision 1 (4): 321 (1988).
"""

import numpy as np
#import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data, io
from skimage.filters import gaussian, roberts, sobel, scharr, prewitt
from skimage.segmentation import active_contour

from scipy.interpolate import interp1d

img = io.imread('../sample_test_frames/cam1_frames_00000_scaled.png')
img = rgb2gray(img)


import keras
import scipy.io
from keras.layers import Dense, Flatten, GaussianNoise
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, GlobalMaxPooling2D, Dropout
from keras.models import Sequential
from keras.models import load_model
from PIL import Image, ImageDraw
#import numpy as np
import pylab as pl

model_name = 'retrained_loc_model.h5'

#load in .mat file as python dictionary
#frames_directory1 = '/scratch/network/apandya/cam1_frames_resized/'
#frames_directory2 = '/scratch/network/apandya/cam1_frames_resized2/'
##frames_directory3 = '/scratch/network/apandya/cam1_frames_resized3/'
##frames_directory_test = '/scratch/network/apandya/cam1_frames_resized_test/'
#frames_directory_test = '/scratch/network/apandya/cam1_frames_resized3/'
#frames_directory3 = '/scratch/network/apandya/cam1_frames_resized_test/'
#frames_directory4 = '/scratch/network/apandya/cam1_frames_resized4/'
#frames_directory5 = '/scratch/network/apandya/cam1_frames_resized5/'
#frames_directory6 = '/scratch/network/apandya/cam1_frames_resized6/'

frames_directory_test = '../sample_test_frames/'

#define a function which loads images and centerlines into an array;
#the array has shape (image_array, centerline_array), where both
#image_array and centerline_array are flattened (1D) arrays
def load_images(frames_directory, num_images_to_load, first_image_index, step):
        image_data = []
        coords_data = []

        for j in range(0, num_images_to_load, step):

                #load image files and print progress through loop
                if(j % 500 == 0):
                        print 'loading image ', first_image_index + j, ' of ', first_image_index + num_images_to_load
                frame_number = j + first_image_index
                frame = frames_directory + 'cam1_frames_' + '%05d'%frame_number + '_scaled.png'

                #the image starts with dimensions 1088x1088; the 3 lines below
                #scales it to img_scaled_dim and calculates the scaling factor
                img_original_dim = 1088
                img_scaled_dim  = 300
                img_scaling_factor = float(img_scaled_dim)/float(img_original_dim)

                #load image and convert to array, flatten array, 
                #and scale both image and coords to be in range [0, 1]
                #(all neurons in net have range [0, 1])
                img_loaded = Image.open(frame)
                
		#do some elementary image enhancement
                img_loaded = np.array(img_loaded) / np.mean(img_loaded) * 0.25
                img_loaded[img_loaded < 0.1] = 0.
                img_loaded[img_loaded > 1.] = 1.

		#flatten image array
		img_flattened    = img_loaded.flatten()

                image_data.append(img_flattened)


	combined_data = [np.array(image_data).reshape((num_images_to_load/step, 300, 300))]

        return combined_data

model = load_model(model_name)

num_test_images = 1000
test_step       = 10

x_test = load_images(frames_directory_test, num_test_images, 0, test_step)[0]
img_x, img_y = 300, 300
num_x_bins, num_y_bins = 15, 15

x_test = x_test.astype('float32')
x_test  = x_test.reshape(x_test.shape[0], img_x, img_y, 1)

num_test_sample_images = 10

predictions = np.array(model.predict(x_test))

#pulls num_test_sample_images from test_data, draws both predicted line (blue)
#and known centerline (red)
#for k in range(num_test_sample_images):

#choose random test image from test_data
#test_image = np.random.randint(low=0, high= int(num_test_images/test_step) )
test_image = 0

def net_line(test_image):
	#retrieve image from array and convert it to correct format for display
	img_reshaped = np.reshape(x_test[test_image], (300, 300)) * 255
	
	net_ans  = predictions[test_image].reshape(num_x_bins, num_y_bins)
	
	bin_img = Image.fromarray(net_ans)
	bin_img = bin_img.resize((300,300))
	
	tol = 9e-2
	net_ans[net_ans < tol] = 0.
	
	net_coords = net_ans.nonzero()
	net_coords = (np.array(net_coords) + 0.5)
	net_coords = np.reshape(net_coords, (2, np.shape(net_coords)[1]))
	net_coords = net_coords * np.sqrt(15**2. + 15**2.) - 7.5
	
	formatted_net_coords = np.array([net_coords[1], net_coords[0]])

	pl.scatter(formatted_net_coords[0], formatted_net_coords[1], color='red')
	pl.contourf(bin_img, alpha=0.25, cmap='gnuplot')
	pl.colorbar()
	pl.imshow(img_reshaped, cmap='gray')
	pl.show()

	return formatted_net_coords.transpose()

init = net_line(test_image)

print 'init is', init

#clear the data tensorflow saves about old models
import tensorflow as tf
tf.reset_default_graph()
keras.backend.clear_session()

print np.shape(init)[0]


interp_init_fn = interp1d(init[:, 0], init[:, 1], kind='cubic')

print init[:, 0], init[:, 1], np.shape(init[:, 0])

xnew = np.linspace(init[0, 0], init[-1, 0], 50, endpoint=True)

interpolated_init = np.array([xnew, interp_init_fn(xnew)])

interpolated_init = interpolated_init.transpose()

#print interpolated_init, interpolated_init.transpose()

#init = [[ 537.61255411,  565.87229437],\
# [ 522.89393939 , 524.66017316],\
# [ 478.73809524,  524.66017316],\
# [ 419.86363636,  504.05411255],\
# [ 390.42640693,  468.72943723],\
# [ 428.69480519,  403.96753247],\
# [ 464.01948052,  383.36147186],\
# [ 478.73809524,  359.81168831],\
# [ 481.68181818,  321.54329004],\
# [ 481.68181818,  297.99350649]]
#init=np.array(init, dtype=int)
#init[1:-2] +=50
#
#init = np.array(init) * 1.0
#init = init/600.
#
#init = init * 300.

#print init
#print init.shape
#init = interp1d(np.arange(100),init.T, kind='linear', axis=0)
#np.interp()

#img = gaussian(img, 2)
#img = img

img = np.reshape(x_test[test_image], (300, 300))

#print np.max(img)
#print np.max(sobel(img))
#img[img<0.01] = 0
#img[img>0.14] *= 5#0*img[img>0.13]
#img = roberts(img)

#img = img>0.15
#snake = active_contour(img,
#                       init, alpha=10, beta=2, gamma=1, bc='fixed', w_edge=100, w_line=10,convergence = 0.1)#,  max_px_move=50, w_line=1, w_edge=10)

#snake = active_contour(img, init, alpha=30, beta=0.1, w_line=200, bc='fixed-free')
snake = active_contour(img, interpolated_init, alpha=30, beta=0.1, w_line=200, bc='fixed-free')

fig, ax = pl.subplots(figsize=(7, 7))
ax.imshow(img, cmap=pl.cm.gray)
#ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
ax.plot(interpolated_init[:, 0], interpolated_init[:, 1], '--r', lw=3)
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])
pl.show()
