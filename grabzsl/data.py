import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing

class Data():
	'''
	This class loads the dataset, generates batches, and provides other useful functions.
	'''
	def __init__(self, dataset_name='awa1', split='', dataroot='./data'):
		'''
		Load the dataset.
		'''
		# read features and labels
		matcontent = sio.loadmat(dataroot + "/" + dataset_name + "/res101.mat")
		feature = matcontent['features'].T
		label = matcontent['labels'].astype(int).squeeze() - 1
		# read attributes and locations data
		matcontent = sio.loadmat(dataroot + "/" + dataset_name + "/att_splits" + split + ".mat")
		self.attributes = torch.from_numpy(matcontent['att'].T).float()
		# normalize, just in case (the datasets used here are already normalized)
		self.attributes /= self.attributes.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attributes.size(0), self.attributes.size(1))
		# numpy array index starts from 0, mat starts from 1
		train_loc = matcontent['trainval_loc'].squeeze() - 1
		test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
		test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
		# load features and labels for training and testing
		scaler = preprocessing.MinMaxScaler()
		self.train_X = torch.from_numpy(scaler.fit_transform(feature[train_loc])).float()
		self.train_Y = torch.from_numpy(label[train_loc]).long()
		self.test_seen_X = torch.from_numpy(scaler.transform(feature[test_seen_loc])).float()
		self.test_seen_Y = torch.from_numpy(label[test_seen_loc]).long()
		self.test_unseen_X = torch.from_numpy(scaler.transform(feature[test_unseen_loc])).float()
		self.test_unseen_Y = torch.from_numpy(label[test_unseen_loc]).long()
		# normalize features
		mx = self.train_X.max()
		self.train_X.mul_(1/mx)
		self.test_seen_X.mul_(1/mx)
		self.test_unseen_X.mul_(1/mx)
		# data size
		self.seen_classes = torch.from_numpy(np.unique(self.train_Y.numpy()))
		self.unseen_classes = torch.from_numpy(np.unique(self.test_unseen_Y.numpy()))
		self.dataset_size = self.train_X.size(0)

	def map_labels(self, label, classes):
		'''
		Map each element in the input label tensor to a corresponding index in the "classes" tensor.
		The resulting "mapped_label" tensor contains indices corresponding to the input "classes" tensor, rather than the original class labels.
		'''
		mapped_label = torch.LongTensor(label.size())
		for i in range(classes.size(0)):
			mapped_label[label == classes[i]] = i
		return mapped_label

	def next_batch(self, batch_size, device='cpu'):
		'''
		Select a batch of data randomly from the training set.
		'''
		idx = torch.randperm(self.dataset_size)[0:batch_size]
		batch_feature = self.train_X[idx].clone()
		batch_label = self.train_Y[idx].clone()
		batch_att = self.attributes[batch_label].clone()
		batch_label = self.map_labels(batch_label, self.seen_classes)
		return batch_feature.to(device), batch_label.to(device), batch_att.to(device)

	def generate_syn_features(self, model_generator, classes, attributes, num_per_class, n_features, n_attributes, latent_size,
							device, model_feedback=None, model_decoder=None, feedback_weight=None):
		'''
		Generate synthetic features for each class.
		'''
		n_classes = classes.size(0)
		# create the tensors for the synthetic dataset
		syn_X = torch.FloatTensor(n_classes * num_per_class, n_features)
		syn_Y = torch.LongTensor(n_classes * num_per_class)
		# create the tensors for the generator input
		syn_attributes = torch.FloatTensor(num_per_class, n_attributes).to(device)
		syn_noise = torch.FloatTensor(num_per_class, latent_size).to(device)
		# create synthetic features for each class
		for i in range(n_classes):
			curr_class = classes[i]
			curr_attributes = attributes[curr_class]
			# generate features conditioned on the current class' signature
			syn_noise.normal_(0, 1)
			syn_attributes.copy_(curr_attributes.repeat(num_per_class, 1))
			fake = model_generator(syn_noise, syn_attributes)
			# go through the feedback module (only for TF-VAEGAN)
			if model_feedback is not None:
				# call the forward function of decoder to get the hidden features
				_ = model_decoder(fake)
				decoder_features = model_decoder.get_hidden_features()
				feedback = model_feedback(decoder_features)
				fake = model_generator(syn_noise, syn_attributes, feedback_weight, feedback=feedback)
			# copy the class' features and labels to the synthetic dataset
			syn_X.narrow(0, i * num_per_class, num_per_class).copy_(fake.data.cpu())
			syn_Y.narrow(0, i * num_per_class, num_per_class).fill_(curr_class)
		return syn_X, syn_Y

	def get_n_classes(self):
		return self.attributes.size(0)

	def get_n_attributes(self):
		return self.attributes.size(1)
