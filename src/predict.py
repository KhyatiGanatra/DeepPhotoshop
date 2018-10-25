import gc
import datetime
import keras
import tensorflow as tf

import pandas as pd
import numpy as np
from copy import deepcopy
from tqdm import tqdm

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras import backend as K

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

import sys
import os
sys.path.append(os.path.join(os.path.join(os.path.dirname(sys.path[0]))))

import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from IPython.display import clear_output

from libs.pconv_model import PConvUnet
from libs.util import random_mask, custom_mask
#import file_sort

import os
import re

plt.ioff()

BATCH_SIZE = 5

TEST_DIR = r'/home/ubuntu/Insight_AI/test_infilling/'

class DataGenerator(ImageDataGenerator):
	def flow_from_directory(self, directory, *args, **kwargs):
		generator = super().flow_from_directory(directory, class_mode=None, *args, **kwargs)
		print (generator)
		while True:
			# Get augmentend image samples
			ori = next(generator)
			#ori = np.random.randInt(512,512)
			# Get masks for images
			mask = np.stack([custom_mask(ori.shape[1], ori.shape[2]) for _ in range(ori.shape[0])], axis=0)

			# Apply masks to all image sample
			masked = deepcopy(ori)
			masked[mask == 0] = 1

			# Yield ([ori, masl],  ori) training batches
			# print(masked.shape, ori.shape)
			gc.collect()
			yield [masked, mask], ori

# Create testing generator
test_datagen = DataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
	TEST_DIR, target_size=(512, 512), batch_size=BATCH_SIZE, seed=1
)

#Pick out an example
test_data = next(test_generator)
(masked, mask), ori = test_data

# Show side by side
for i in range(len(ori)):
	_, axes = plt.subplots(1, 3, figsize=(20, 5))
	axes[0].imshow(masked[i,:,:,:])
	axes[1].imshow(mask[i,:,:,:] * 1.)
	axes[2].imshow(ori[i,:,:,:])
	#plt.show()


def plot_callback(model):
	"""Called at the end of each epoch, displaying our previous test images,
	as well as their masked predictions and saving them to disk"""

	# Get samples & Display them
	pred_img = model.predict([masked, mask])
	pred_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

	# Clear current output and display test images
	for i in range(len(ori)):
		_, axes = plt.subplots(1, 3, figsize=(20, 5))
		axes[0].imshow(masked[i, :, :, :])
		axes[1].imshow(pred_img[i, :, :, :] * 1.)
		axes[2].imshow(ori[i, :, :, :])
		axes[0].set_title('Masked Image')
		axes[1].set_title('Predicted Image')
		axes[2].set_title('Original Image')

		plt.savefig(r'data/custom_test_samples/img_{}_{}.png'.format(i, pred_time))
		plt.close()

# Instantiate the model
model = PConvUnet()
model.load('/home/ubuntu/Insight_AI/DeepPhotoshop/data/logs/382_weights_2018-10-16-23-26-19.h5')
n = 0
for (masked, mask), ori in tqdm(test_generator):
	print(masked, mask, "-----")	

	# Run predictions for this batch of images
	pred_img = model.predict([masked, mask])
	pred_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

	# Clear current output and display test images
	for i in range(len(ori)):
		_, axes = plt.subplots(1, 2, figsize=(10, 5))
		axes[0].imshow(masked[i,:,:,:])
		axes[1].imshow(pred_img[i,:,:,:] * 1.)
		axes[0].set_title('Masked Image')
		axes[1].set_title('Predicted Image')
		axes[0].xaxis.set_major_formatter(NullFormatter())
		axes[0].yaxis.set_major_formatter(NullFormatter())
		axes[1].xaxis.set_major_formatter(NullFormatter())
		axes[1].yaxis.set_major_formatter(NullFormatter())

		plt.savefig(r'../data/custom_results/img_{}_{}.png'.format(i, pred_time))
		plt.close()
		n += 1

	# Only create predictions for about 100 images
	if n > 10:
		break

