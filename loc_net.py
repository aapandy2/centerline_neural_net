#from __future__ import print_function
import keras
import scipy.io
from keras.layers import Dense, Flatten, GaussianNoise
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, GlobalMaxPooling2D, Dropout
from keras.models import Sequential
from keras.models import load_model
from PIL import Image, ImageDraw
import numpy as np
import pylab as pl

#choose whether to train new model or load existing model
TRAIN_MODEL        = 0
LOAD_MODEL         = 1
TRAIN_LOADED_MODEL = 2
mode = TRAIN_MODEL
model_name = 'retrained_loc_model.h5'

#load in .mat file as python dictionary
frames_directory1 = '/scratch/network/apandya/cam1_frames_resized/'
frames_directory2 = '/scratch/network/apandya/cam1_frames_resized2/'
#frames_directory3 = '/scratch/network/apandya/cam1_frames_resized3/'
#frames_directory_test = '/scratch/network/apandya/cam1_frames_resized_test/'
frames_directory_test = '/scratch/network/apandya/cam1_frames_resized3/'
frames_directory3 = '/scratch/network/apandya/cam1_frames_resized_test/'
frames_directory4 = '/scratch/network/apandya/cam1_frames_resized4/'
frames_directory5 = '/scratch/network/apandya/cam1_frames_resized5/'

#num_classes = 10

num_x_bins = 6
num_y_bins = 6

def generate_bin_array(centerline_array):
        true_coords_array = centerline_array.astype(int)

        box_array = np.zeros((num_x_bins, num_y_bins))
        x_bin = 0
        y_bin = 0
        for point in true_coords_array:
                x_bin = point[0] / (300/num_x_bins)
                y_bin = point[1] / (300/num_y_bins)
                box_array[y_bin, x_bin] += 1

        return box_array

#define a function which loads images and centerlines into an array;
#the array has shape (image_array, centerline_array), where both
#image_array and centerline_array are flattened (1D) arrays
def load_images(frames_directory, num_images_to_load, first_image_index, step):
        image_data = []
        coords_data = []
	mat = scipy.io.loadmat(frames_directory + 'centerline.mat')

        for j in range(0, num_images_to_load, step):

                #load image files and print progress through loop
                if(j % 500 == 0):
                        print 'loading image ', first_image_index + j, ' of ', first_image_index + num_images_to_load
                frame_number = j + first_image_index
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

		bin_array = generate_bin_array(coords_np)

                #load image and convert to array, flatten array, 
                #and scale both image and coords to be in range [0, 1]
                #(all neurons in net have range [0, 1])
                img_loaded = Image.open(frame)
                
		#do some elementary image enhancement
                img_loaded = np.array(img_loaded) / np.mean(img_loaded) * 0.25
                img_loaded[img_loaded < 0.2] = 0.
                img_loaded[img_loaded > 1.] = 1.

		#flatten arrays
#		coords_flattened = np.array(coords_np[1::(100/(num_classes/2))]).flatten() / 300.
		img_flattened    = img_loaded.flatten()

                #reshape flattened coords array to be 1D columns (net input format)
#                coords_flattened = np.reshape(coords_flattened, (1, num_classes))
		bin_array_flattened = np.reshape(bin_array, (1, num_x_bins*num_y_bins)) / 100. #TODO:remove this

                image_data.append(img_flattened)
                coords_data.append(bin_array_flattened[0])

        combined_data = [np.array(image_data).reshape((num_images_to_load/step, 300, 300)), np.array(coords_data)]

        return combined_data

#load num_images_to_load, stepping through by step, starting at 
#first_image_index
step = 100
first_image_index = 0

#training parameters
batch_size = 1
epochs = 20

# input image dimensions
img_x, img_y = 300, 300

def combine_training_data(num_training_sets):

	size_array = [27000, 42000, 24000, 15000, 35000]
	direc_array = [frames_directory1, frames_directory2, frames_directory3, frames_directory4, frames_directory5]

	combined_imgs, combined_centerlines = load_images(direc_array[0], size_array[0], first_image_index, step)
	
	for i in range(1, num_training_sets):
		tdata = load_images(direc_array[i], size_array[i], first_image_index, step)
		combined_imgs = np.append(combined_imgs, tdata[0])
		combined_centerlines = np.append(combined_centerlines, tdata[1])	
		

	total_images = np.sum(size_array[0:num_training_sets]) / step
	print 'total_images', total_images
#
        combined_imgs = np.reshape(combined_imgs, (total_images, img_x, img_y))
        combined_centerlines = np.reshape(combined_centerlines, (total_images, num_x_bins*num_y_bins))

	combined_array = [combined_imgs, combined_centerlines]
	return combined_array

#combined_imgs, combined_centerlines = combine_training_data(1)
#for i in range(10):
#	bin_img = Image.fromarray(combined_centerlines[i].reshape((num_x_bins, num_y_bins)))
#	print combined_centerlines[i].reshape((num_x_bins, num_y_bins))
#	bin_img = bin_img.resize((300,300))
#	
#	pl.contourf(bin_img, alpha=0.2, cmap='jet')
#	pl.imshow(combined_imgs[i], cmap='gray')
#	pl.show()

if(mode == TRAIN_MODEL):
	combined_imgs, combined_centerlines = combine_training_data(3)

	x_train = combined_imgs
	y_train = combined_centerlines
	
	# convert the data to the right type
	x_train = x_train.astype('float32')
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	
	x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
	input_shape = (img_x, img_y, 1)	
#	x_train = x_train.reshape(x_train.shape[0], img_x*img_y, 1)
#	input_shape = (img_x*img_y, 1)

	model = Sequential()
	model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2),
	                 activation='relu',
	                 input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2),
			 activation='relu',))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#	model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2),
#                         activation='relu'))
#        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#	model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2),
#                         activation='relu'))
#        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Flatten())
#	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#	model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#	model.add(MaxPooling2D(pool_size=(2, 2)))
#	model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#        model.add(MaxPooling2D(pool_size=(2, 2)))
#	model.add(Flatten())
	model.add(Dense(num_x_bins*num_y_bins, activation='sigmoid'))
	
#	model.compile(loss=keras.losses.binary_crossentropy,
	model.compile(loss=keras.losses.mean_squared_error,
#	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adam())
#		      optimizer=keras.optimizers.SGD(lr=0.30))
	
	
	class AccuracyHistory(keras.callbacks.Callback):
	    def on_train_begin(self, logs={}):
	        self.acc = []
	
	    def on_epoch_end(self, batch, logs={}):
	        self.acc.append(logs.get('acc'))
	
	history = AccuracyHistory()
	
	model.fit(x_train, y_train,
	          batch_size=batch_size,
	          epochs=epochs,
	          verbose=1,
	          callbacks=[history])
	
	# Creates a HDF5 file 'my_model1.h5'
#	model.save('my_model_retrain.h5')
	model.save(model_name)


if(mode == TRAIN_LOADED_MODEL):
	combined_imgs, combined_centerlines = combine_training_data(5)

	x_train = combined_imgs
        y_train = combined_centerlines

        # convert the data to the right type
        x_train = x_train.astype('float32')
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')

        x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
        input_shape = (img_x, img_y, 1)

        model = load_model(model_name)

        model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=keras.optimizers.Adam())
        #             optimizer=keras.optimizers.SGD(lr=0.30))


        class AccuracyHistory(keras.callbacks.Callback):
            def on_train_begin(self, logs={}):
                self.acc = []

            def on_epoch_end(self, batch, logs={}):
                self.acc.append(logs.get('acc'))

        history = AccuracyHistory()

        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[history])

        # Creates a HDF5 file 'my_model1.h5'
        model.save(model_name)

if(mode == LOAD_MODEL):
	# Returns a compiled model identical to the previous one
	model = load_model(model_name)

	num_test_images = 15000
	test_step       = 100

	x_test, y_test = load_images(frames_directory3, num_test_images, 0, test_step)

	x_test = x_test.astype('float32')
	x_test  = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
#	x_test  = x_test.reshape(x_test.shape[0], img_x*img_y, 1)

#	print y_test[0].reshape((5,5))

	num_test_sample_images = 10
	
	predictions = np.array(model.predict(x_test)) #* 300.
	
	#pulls num_test_sample_images from test_data, draws both predicted line (blue)
	#and known centerline (red)
	for k in range(num_test_sample_images):
	
		#choose random test image from test_data
		test_image = np.random.randint(low=0, high= int(num_test_images/test_step) )
		
		#retrieve image from array and convert it to color to draw colored centerline
#		img = Image.fromarray(np.reshape(x_test[test_image], (300, 300)) * 255).convert('RGBA')
#		draw = ImageDraw.Draw(img)
	        img_reshaped = np.reshape(x_test[test_image], (300, 300)) * 255
	
		
		#generate net centerline and retrieve "true" centerline from array 
		#and convert them to tuples of (x, y) tuples for draw.line() below
#		net_coords_tuple = tuple(map(tuple, np.reshape(predictions[test_image], (num_classes/2, 2))))
#		net_coords_tuple = tuple(map(tuple, np.reshape(np.array(model.predict(x_test))[test_image] * 300., (num_classes/2, 2))))
#		true_coords_tuple = tuple(map(tuple, np.reshape(np.array(y_test[test_image])*300., (200/2, 2))))

		true_ans = y_test[test_image].reshape(num_x_bins, num_y_bins) #TODO: reverse x and y?
		net_ans  = predictions[test_image].reshape(num_x_bins, num_y_bins)

		print 'diff:', (net_ans - true_ans)

		bin_img = Image.fromarray(net_ans)
	        bin_img = bin_img.resize((300,300))

	        pl.contourf(bin_img, alpha=0.2, cmap='Blues')
	        pl.imshow(img_reshaped, cmap='gray')
	        pl.show()

#		print true_ans
#		print net_ans

#		true_coords_array = np.array(true_coords_tuple)

#		print generate_bin_array(true_coords_array)
#		print predictions[test_image]
		
		#draw net centerline and "true" centerline on test image and show image
#		if(num_classes > 2):
#			draw.line(net_coords_tuple, fill=(0, 0, 255), width=2)
#			draw.line(true_coords_tuple, fill=(255, 0, 0), width=2)
#		#	img.show()
#			img.save('output_' + str(k) + '.png')
#		else:
#			print 'right ans:', true_coords_tuple, 'net ans:', net_coords_tuple

#clear the data tensorflow saves about old models
import tensorflow as tf
tf.reset_default_graph()
keras.backend.clear_session()
