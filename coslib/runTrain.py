import time
import os
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 


flags = tf.app.flags
FLAGS = flags.FLAGS
# define parameters
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate')
flags.DEFINE_string('path_to_data', 'data/', 'Path to the training and validation dataset')



def run_training():
	'''
	'''




def main():
	run_training()

if __name__ == '__main__':
	tf.app.run()