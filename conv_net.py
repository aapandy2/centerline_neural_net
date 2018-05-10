#from __future__ import print_function
import keras
import scipy.io
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from PIL import Image, ImageDraw
import numpy as np

#load in .mat file as python dictionary
mat = scipy.io.loadmat('../centerline.mat')

#define a function which loads images and centerlines into an array;
#the array has shape (image_array, centerline_array), where both
#image_array and centerline_array are flattened (1D) arrays
def load_images(num_images_to_load, first_image_index, step):
        image_data = []
        coords_data = []
        for j in range(0, num_images_to_load, step):

                #load image files and print progress through loop
                if(j % 500 == 0):
                        print 'loading image ', first_image_index + j, ' of ', first_image_index + num_images_to_load
                frame_number = j + first_image_index
                frames_directory = '../cam1_frames_30Hz_scaled/'
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

                #load image and convert to array, flatten array, 
                #and scale both image and coords to be in range [0, 1]
                #(all neurons in net have range [0, 1])
                img_loaded = Image.open(frame)
                coords_flattened = np.array(coords_np[1::20]).flatten() / 300.
		img_flattened    = np.array(img_loaded).flatten() / 255.
#                img_flattened    = np.array(img_loaded) / 255.

                #reshape flattened coords array to be 1D columns (net input format)
                coords_flattened = np.reshape(coords_flattened, (1, 10))

                image_data.append(img_flattened)
                coords_data.append(coords_flattened[0])

        combined_data = [np.array(image_data).reshape((num_images_to_load/step, 300, 300)), np.array(coords_data)]

        return combined_data

#load num_images_to_load, stepping through by step, starting at 
#first_image_index
num_images_to_load = 1000
step = 2
first_image_index = 0
training_data = load_images(num_images_to_load, first_image_index, step)
test_data     = load_images(num_images_to_load, 1, 2)

#training parameters
batch_size = 1
num_classes = 10
epochs = 20

# input image dimensions
img_x, img_y = 300, 300

x_train, y_train = training_data
x_test, y_test   = test_data

# convert the data to the right type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test  = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
input_shape = (img_x, img_y, 1)

#my neural net model
#model = Sequential()
#model.add(Dense(20, input_shape=(90000,), activation='sigmoid'))
#model.add(Dense(20, activation='sigmoid'))
#model.add(Dense(10, activation='sigmoid'))

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='sigmoid',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(20, activation='sigmoid'))
model.add(Dense(num_classes, activation='sigmoid'))

#model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.Adam())#,
#             metrics=['accuracy'])

model.compile(loss=keras.losses.binary_crossentropy,
#              optimizer=keras.optimizers.Adam())
	      optimizer=keras.optimizers.SGD(lr=0.30))


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

#print model.predict(x_test)[0]
#print y_test[0]
#model.save_weights('first_try.h5')

num_test_sample_images = 5
num_test_images = num_images_to_load
test_step = 2
for k in range(num_test_sample_images):

	#choose random test image from test_data
	test_image = np.random.randint(low=0, high= int(num_test_images/test_step) )
	
	#retrieve image from array and convert it to color to draw colored centerline
	img = Image.fromarray(np.reshape(x_test[test_image], (300, 300)) * 255).convert('RGBA')
	draw = ImageDraw.Draw(img)
	
	#generate net centerline and retrieve "true" centerline from array 
	#and convert them to tuples of (x, y) tuples for draw.line() below
	net_coords_tuple = tuple(map(tuple, np.reshape(np.array(model.predict(x_test))[test_image] * 300., (5, 2))))
	true_coords_tuple = tuple(map(tuple, np.reshape(np.array(y_test[test_image])*300., (5, 2))))
	
	#draw net centerline and "true" centerline on test image and show image
	draw.line(net_coords_tuple, fill=(0, 0, 255), width=2)
	draw.line(true_coords_tuple, fill=(255, 0, 0), width=2)
	img.show()
