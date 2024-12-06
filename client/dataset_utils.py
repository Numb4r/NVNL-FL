import tensorflow as tf
import numpy as np
import random
import pickle
import pandas as pd

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
#from sklearn.preprocessing import Normalizer

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

class ManageDatasets():

	def __init__(self, cid):
		self.cid = cid
		#random.seed(self.cid)

	def load_UCIHAR(self):
		with open(f'client/data/UCI-HAR/{self.cid +1}_train.pickle', 'rb') as train_file:
			train = pickle.load(train_file)

		with open(f'client/data/UCI-HAR/{self.cid+1}_test.pickle', 'rb') as test_file:
			test = pickle.load(test_file)

		train['label'] = train['label'].apply(lambda x: x -1)
		y_train        = train['label'].values
		train.drop('label', axis=1, inplace=True)
		x_train = train.values

		test['label'] = test['label'].apply(lambda x: x -1)
		y_test        = test['label'].values
		test.drop('label', axis=1, inplace=True)
		x_test = test.values

		return x_train, y_train, x_test, y_test

	def load_ExtraSensory(self):
		with open(f'client/data/ExtraSensory/x_train_client_{self.cid+1}.pickle', 'rb') as x_train_file:
			x_train = pickle.load(x_train_file)

		with open(f'client/data/ExtraSensory/x_test_client_{self.cid+1}.pickle', 'rb') as x_test_file:
			x_test = pickle.load(x_test_file)
	    
		with open(f'client/data/ExtraSensory/y_train_client_{self.cid+1}.pickle', 'rb') as y_train_file:
			y_train = pickle.load(y_train_file)

		with open(f'client/data/ExtraSensory/y_test_client_{self.cid+1}.pickle', 'rb') as y_test_file:
			y_test = pickle.load(y_test_file)

		y_train = np.array(y_train) + 1
		y_test  = np.array(y_test) + 1

		return x_train, y_train, x_test, y_test


	def load_MotionSense(self):
		with open(f'client/data/motion_sense/{self.cid+1}_train.pickle', 'rb') as train_file:
			train = pickle.load(train_file)
	    
		with open(f'client/data/motion_sense/{self.cid+1}_test.pickle', 'rb') as test_file:
			test = pickle.load(test_file)
	        
		y_train = train['activity'].values
		train.drop('activity', axis=1, inplace=True)
		train.drop('subject', axis=1, inplace=True)
		train.drop('trial', axis=1, inplace=True)
		# train['subject'] /= 24.0
		# train['trial']   /= 16.0
		x_train = train.values

		y_test = test['activity'].values
		test.drop('activity', axis=1, inplace=True)
		# test['subject'] /= 24.0
		# test['trial']   /= 16.0
		test.drop('subject', axis=1, inplace=True)
		test.drop('trial', axis=1, inplace=True)
		x_test = test.values
	    
		return x_train, y_train, x_test, y_test


	def load_MNIST(self, n_clients, non_iid=False):


		if non_iid:

			with open(f'./data/MNIST/{n_clients}/idx_train_{self.cid}.pickle', 'rb') as handle:
				idx_train = pickle.load(handle)

			with open(f'./data/MNIST/{n_clients}/idx_test_{self.cid}.pickle', 'rb') as handle:
				idx_test = pickle.load(handle)


			(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
			x_train, x_test                      = x_train/255.0, x_test/255.0

			x_train = x_train[idx_train]
			x_test  = x_test[idx_test]

			y_train = y_train[idx_train]
			y_test  = y_test[idx_test]
			

		else:

			(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
			x_train, x_test                      = x_train/255.0, x_test/255.0
			x_train, y_train, x_test, y_test     = self.slipt_dataset(x_train, y_train, x_test, y_test, n_clients)

		return x_train, y_train, x_test, y_test



	def slipt_dataset(self, x_train, y_train, x_test, y_test, n_clients):
		p_train = int(len(x_train)/n_clients)
		p_test  = int(len(x_test)/n_clients)


		random.seed(self.cid)
		selected_train = random.sample(range(len(x_train)), p_train)

		random.seed(self.cid)
		selected_test  = random.sample(range(len(x_test)), p_test)
		
		x_train  = x_train[selected_train]
		y_train  = y_train[selected_train]

		x_test   = x_test[selected_test]
		y_test   = y_test[selected_test]


		return x_train, y_train, x_test, y_test


	def select_dataset(self, dataset_name, n_clients, non_iid):

		if dataset_name == 'MNIST':
			return self.load_MNIST(n_clients, non_iid)

		elif dataset_name == 'CIFAR100':
			return self.load_CIFAR100(n_clients, non_iid)

		elif dataset_name == 'CIFAR10':
			return self.load_CIFAR10(n_clients, non_iid)

		elif dataset_name == 'MotionSense':
			return self.load_MotionSense()
		
		elif dataset_name == 'ExtraSensory':
			return self.load_ExtraSensory()

		elif dataset_name == 'UCIHAR':
			return self.load_UCIHAR()

def load_data_flowerdataset(self):
        
	if self.niid:
		partitioner_train = DirichletPartitioner(num_partitions=self.num_clients, partition_by="label",
								alpha=self.dirichlet_alpha, min_partition_size=100,
								self_balancing=False)
	else:
		partitioner_train =  IidPartitioner(num_partitions=self.num_clients)
	
	fds               = FederatedDataset(dataset=self.dataset, partitioners={"train": partitioner_train})
	train             = fds.load_partition(self.cid).with_format("numpy")
	partitioner_test  = IidPartitioner(num_partitions=self.num_clients)
	fds_eval          = FederatedDataset(dataset=self.dataset, partitioners={"test": partitioner_test})
	test              = fds_eval.load_partition(self.cid).with_format("numpy")

	if self.dataset == 'CIFAR10':
		return train['img']/255.0, train['label'], test['img']/255.0, test['label']

	return train['image']/255.0, train['label'], test['image']/255.0, test['label']