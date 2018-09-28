import gc
import datetime

import pandas as pd
import numpy as np
from copy import deepcopy
from tqdm import tqdm

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras import backend as K

import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from IPython.display import clear_output

from libs.pconv_model import PConvUnet
from libs.util import random_mask
import file_sort

import os
import re

def get_latest_weights_file():
	log_dir = "/home/ubuntu/Insight_AI/PConv-Keras/data/logs/"
	files = os.listdir(log_dir)
	absnames = []
	for file in files:
		absnames.append(log_dir + file)
	latest_file = max(absnames, key=os.path.getctime)
	return latest_file
#f = get_latest_weights_file()


plt.ioff()

# SETTINGS
#TRAIN_DIR = r"/home/khyati/Documents/Insight/sports_images/train_1/"
#TEST_DIR = r"/home/khyati/Documents/Insight/sports_images/test_1"
#VAL_DIR = r"/home/khyati/Documents/Insight/sports_images/val_1/"

TRAIN_DIR = r"/home/ubuntu/Insight_AI/sports_images/train_1/"
TEST_DIR = r"/home/ubuntu/Insight_AI/sports_images/test_1/"
VAL_DIR = r"/home/ubuntu/Insight_AI/sports_images/val_1/"

BATCH_SIZE = 4


class DataGenerator(ImageDataGenerator):
	def flow_from_directory(self, directory, *args, **kwargs):
		generator = super().flow_from_directory(directory, class_mode=None, *args, **kwargs)
		while True:
			# Get augmentend image samples
			ori = next(generator)

			# Get masks for each image sample
			mask = np.stack([random_mask(ori.shape[1], ori.shape[2]) for _ in range(ori.shape[0])], axis=0)

			# Apply masks to all image sample
			masked = deepcopy(ori)
			masked[mask == 0] = 1

			# Yield ([ori, masl],  ori) training batches
			# print(masked.shape, ori.shape)
			gc.collect()
			yield [masked, mask], ori


# Create training generator
train_datagen = DataGenerator(
	rotation_range=20,
	width_shift_range=0.2,
	height_shift_range=0.2,
	rescale=1. / 255,
	horizontal_flip=True
)
train_generator = train_datagen.flow_from_directory(
	TRAIN_DIR, target_size=(512, 512), batch_size=BATCH_SIZE
)

# Create validation generator
val_datagen = DataGenerator(rescale=1. / 255)
val_generator = val_datagen.flow_from_directory(
	VAL_DIR, target_size=(512, 512), batch_size=BATCH_SIZE, seed=1
)

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

		plt.savefig(r'data/test_samples/img_{}_{}.png'.format(i, pred_time))
		plt.close()

# Instantiate the model
#model = PConvUnet()
model = PConvUnet(weight_filepath='data/logs/')
#latest_weights = get_latest_weights_file()
#model.load(latest_weights)

# Run training for certain amount of epochs
model.fit(
	train_generator,
	steps_per_epoch=50,
	validation_data=val_generator,
	validation_steps=5,
	epochs=2,
	plot_callback=plot_callback,
	callbacks=[
	TensorBoard(log_dir='../data/logs/initial_training', write_graph=False)
	]
)

# Load weights from previous run
latest_weights = get_latest_weights_file()
model = PConvUnet(weight_filepath='data/logs/')
model.load(
	latest_weights,
	train_bn=False,
	lr=0.00005
)

# Run training for certain amount of epochs
model.fit(
	train_generator, 
	steps_per_epoch=50,
	validation_data=val_generator,
	validation_steps=5,
	epochs=2,	
	workers=3,
	plot_callback=plot_callback,
	callbacks=[
	TensorBoard(log_dir='../data/logs/fine_tuning', write_graph=False)
	]
)

# Load weights from previous run
latest_weights = get_latest_weights_file()
model = PConvUnet(weight_filepath='data/logs/')
model.load(
	latest_weights,
	train_bn=False,
	lr=0.00005
)

n = 0
for (masked, mask), ori in tqdm(test_generator):

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

		plt.savefig(r'data/test_samples/img_{}_{}.png'.format(i, pred_time))
		plt.close()
		n += 1

	# Only create predictions for about 100 images
	if n > 100:
		break

