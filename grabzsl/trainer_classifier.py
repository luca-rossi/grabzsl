import torch
import torch.nn as nn
import torch.optim as optim
from grabzsl.models import Classifier

class TrainerClassifier():
	'''
	This class trains a standard log softmax classifier. It is used for training the CLSWGAN pre-classifier and the ZSL and GZSL classifiers.
	'''
	def __init__(self, train_X, train_Y, data_loader, input_dim=None, n_attributes=85, batch_size=64, hidden_size=4096,
				n_epochs=50, n_classes=50, lr=0.001, beta1=0.5, model_decoder=None, is_preclassifier=False, device='cpu', verbose=False):
		'''
		Setup the dataset, model, optimizer, and other parameters.
		'''
		self.device = device
		self.batch_size = batch_size
		self.data_loader = data_loader
		if input_dim is None:
			input_dim = train_X.size(1)
		if not is_preclassifier:
			# set up the ZSL / GZSL dataset.
			self.test_seen_X = data_loader.test_seen_X.clone()
			self.test_seen_Y = data_loader.test_seen_Y
			self.test_unseen_X = data_loader.test_unseen_X.clone()
			self.test_unseen_Y = data_loader.test_unseen_Y
			self.seen_classes = data_loader.seen_classes
			self.unseen_classes = data_loader.unseen_classes
			self.model_decoder = model_decoder
			# used for TF-VAEGAN
			if self.model_decoder:
				# use the decoder to decode the dataset
				self.model_decoder.eval()
				input_dim += n_attributes + hidden_size
				train_X = self.__compute_decoder_output(train_X, input_dim)
				self.test_unseen_X = self.__compute_decoder_output(self.test_unseen_X, input_dim)
				self.test_seen_X = self.__compute_decoder_output(self.test_seen_X, input_dim)
		self.train_X = train_X
		self.train_Y = train_Y
		self.input_dim = input_dim
		self.n_epochs = n_epochs
		self.model = Classifier(self.input_dim, n_classes).to(self.device)
		self.criterion = nn.NLLLoss().to(self.device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(beta1, 0.999))
		self.input = torch.FloatTensor(self.batch_size, self.input_dim).to(self.device)
		self.label = torch.LongTensor(self.batch_size).to(self.device)
		self.dataset_size = self.train_X.size(0)
		self.verbose = verbose
		self.batch_index = 0
		self.n_epochs_completed = 0

	def fit_precls(self):
		'''
		Train the pre-classifier.
		'''
		for epoch in range(self.n_epochs):
			for i in range(0, self.dataset_size, self.batch_size):
				self.__train_batch()

	def fit_zsl(self):
		'''
		Train and evaluate the ZSL classifier.
		'''
		best_acc = 0
		for epoch in range(self.n_epochs):
			for i in range(0, self.dataset_size, self.batch_size):
				self.__train_batch()
			test_Y = self.data_loader.map_labels(self.test_unseen_Y, self.unseen_classes)
			mapped_classes = torch.arange(self.unseen_classes.size(0))
			acc = self.__eval_accuracy(self.test_unseen_X, test_Y, mapped_classes)
			if self.verbose:
				print('ZSL: Unseen: %.4f' % (acc))
			if acc > best_acc:
				best_acc = acc
		return best_acc

	def fit_gzsl(self):
		'''
		Train and evaluate the GZSL classifier.
		'''
		best_seen = 0
		best_unseen = 0
		best_H = 0
		for epoch in range(self.n_epochs):
			for i in range(0, self.dataset_size, self.batch_size):
				self.__train_batch()
			acc_seen = self.__eval_accuracy(self.test_seen_X, self.test_seen_Y, self.seen_classes)
			acc_unseen = self.__eval_accuracy(self.test_unseen_X, self.test_unseen_Y, self.unseen_classes)
			acc_H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
			if self.verbose:
				print('GZSL: Seen: %.4f, Unseen: %.4f, H: %.4f' % (acc_seen, acc_unseen, acc_H))
			if acc_H > best_H:
				best_seen = acc_seen
				best_unseen = acc_unseen
				best_H = acc_H
		return best_seen, best_unseen, best_H

	def __train_batch(self):
		'''
		Train the classifier on one batch.
		'''
		self.model.zero_grad()
		batch_input, batch_label = self.__next_batch(self.batch_size)
		self.input.copy_(batch_input)
		self.label.copy_(batch_label)
		output = self.model(self.input)
		loss = self.criterion(output, self.label)
		loss.backward()
		self.optimizer.step()
		if self.verbose:
			print('Classifier loss: ', loss.data.item())

	def __next_batch(self, batch_size):
		'''
		Select a batch of data sequentially.
		'''
		start = self.batch_index
		# shuffle the data at the first epoch
		if self.n_epochs_completed == 0 and start == 0:
			perm = torch.randperm(self.dataset_size)
			self.train_X = self.train_X[perm]
			self.train_Y = self.train_Y[perm]
		# last batch
		if start + batch_size > self.dataset_size:
			self.n_epochs_completed += 1
			rest_num_examples = self.dataset_size - start
			if rest_num_examples > 0:
				X_rest_part = self.train_X[start:self.dataset_size]
				Y_rest_part = self.train_Y[start:self.dataset_size]
			# shuffle the data
			perm = torch.randperm(self.dataset_size)
			self.train_X = self.train_X[perm]
			self.train_Y = self.train_Y[perm]
			# start next epoch
			start = 0
			self.batch_index = batch_size - rest_num_examples
			end = self.batch_index
			X_new_part = self.train_X[start:end]
			Y_new_part = self.train_Y[start:end]
			if rest_num_examples > 0:
				return torch.cat((X_rest_part, X_new_part), 0) , torch.cat((Y_rest_part, Y_new_part), 0)
			return X_new_part, Y_new_part
		# normal batch
		self.batch_index += batch_size
		end = self.batch_index
		return self.train_X[start:end], self.train_Y[start:end]

	def __compute_decoder_output(self, features, output_dim):
		'''
		Compute the decoder output from the input features.
		'''
		start = 0
		n_features = features.size(0)
		new_features = torch.zeros(n_features, output_dim)
		for i in range(0, n_features, self.batch_size):
			end = min(n_features, start + self.batch_size)
			part_features = features[start:end].to(self.device)
			feat1 = self.model_decoder(part_features)
			feat2 = self.model_decoder.get_hidden_features()
			new_features[start:end] = torch.cat([part_features, feat1, feat2], dim=1).data
			start = end
		return new_features
	
	def __eval_accuracy(self, test_X, test_Y, target_classes):
		'''
		Evaluate the per class accuracy of the classifier.
		'''
		start = 0
		n_features = test_X.size(0)
		predicted_label = torch.LongTensor(test_Y.size())
		# predicted labels
		for i in range(0, n_features, self.batch_size):
			end = min(n_features, start + self.batch_size)
			features = test_X[start:end].to(self.device)
			output = self.model(features)
			_, predicted_label[start:end] = torch.max(output.data, 1)
			start = end
		# compute per class accuracy
		acc_per_class = 0
		for i in target_classes:
			idx = (test_Y == i)
			n_samples = torch.sum(idx)
			if n_samples != 0:
				acc_per_class += torch.sum(test_Y[idx] == predicted_label[idx]) / n_samples
		acc_per_class /= target_classes.size(0)
		return acc_per_class
