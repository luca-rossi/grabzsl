import numpy as np
import scipy.io as sio
import torch

class Splitter():
	'''
	This class generates new splits of the dataset according to some splitting methods.
	'''
	def __init__(self, path, mas_k=40):
		'''
		Load dataset and initialize splitting methods with their parameters (if any).
		'''
		self.split_types = {
			'rnd': self.random_split,
			'gcs': self.greedy_class_split,
			'ccs': self.clustered_class_split,
			'mas': self.minimal_attribute_split,
		}
		self.path = path
		self.mas_k = mas_k
		# load dataset
		matcontent_res101 = sio.loadmat(self.path + '/res101.mat')
		self.matcontent_att_splits = sio.loadmat(self.path + '/att_splits.mat')
		# get data: features, labels, and attributes
		self.features = matcontent_res101['features'].T
		self.labels = matcontent_res101['labels'].astype(int).squeeze() - 1
		self.attributes = self.matcontent_att_splits['att'].T
		# get location (index) data and splits
		train_loc = self.matcontent_att_splits['trainval_loc'].squeeze() - 1
		test_seen_loc = self.matcontent_att_splits['test_seen_loc'].squeeze() - 1
		test_unseen_loc = self.matcontent_att_splits['test_unseen_loc'].squeeze() - 1
		self.test_seen_ratio = test_seen_loc.size / (test_seen_loc.size + train_loc.size)
		self.seen_labels = torch.from_numpy(self.labels[train_loc]).long().numpy()
		self.unseen_labels = torch.from_numpy(self.labels[test_unseen_loc]).long().numpy()
		self.n_seen_classes = torch.from_numpy(np.unique(self.seen_labels)).size(0)
		self.n_unseen_classes = torch.from_numpy(np.unique(self.unseen_labels)).size(0)
		# print data info
		print('Features: ', self.features.shape)
		print('Labels: ', self.labels.shape)
		print('Attributes: ', self.attributes.shape)
		print('Seen classes: ', self.n_seen_classes)
		print('Unseen classes: ', self.n_unseen_classes)

	def generate_split(self, split, inverse=False, save=True):
		'''
		Generate a split or its inverse and save the new dataset.
		Note: the save parameter should always be true, we keep it only for debugging purposes.
		'''
		# generate new splits
		new_seen, new_unseen, new_attributes = self.split_types[split](inverse)
		matcontent_att_splits_new = self.matcontent_att_splits.copy()
		# get new seen_loc and unseen_loc from new splits
		seen_loc = np.where(np.in1d(self.labels, new_seen))[0]
		test_unseen_loc = np.where(np.in1d(self.labels, new_unseen))[0]
		# shuffle the seen locations, this is necessary because the labels are in order,
		# so without shuffling the training set and test set would have different classes
		np.random.shuffle(seen_loc)
		test_seen_loc = seen_loc[:int(self.test_seen_ratio * seen_loc.size)]
		train_loc = seen_loc[int(self.test_seen_ratio * seen_loc.size):]
		matcontent_att_splits_new['test_seen_loc'] = test_seen_loc + 1
		matcontent_att_splits_new['test_unseen_loc'] = test_unseen_loc + 1
		matcontent_att_splits_new['trainval_loc'] = train_loc + 1
		matcontent_att_splits_new['att'] = new_attributes.T
		# save new splits
		if save:
			if split == 'mas':
				split += str(self.mas_k)
			if inverse:
				split += '_inv'
			sio.savemat(self.path + '/att_splits_' + split + '.mat', matcontent_att_splits_new)
			print('Saved')

	def generate_all_splits(self):
		'''
		Generate all the splits and save them.
		'''
		for split in self.split_types:
			self.generate_split(split)
			self.generate_split(split, inverse=True)

	def generate_random_splits(self, seed=1000, n_splits=10, n_seen=0, n_unseen=0, save=True):
		'''
		Generate and save n_splits random splits using the random_split function.
		'''
		np.random.seed(seed)
		for i in range(n_splits):
			# generate new split
			new_seen, new_unseen, new_attributes = self.random_split(n_seen=n_seen, n_unseen=n_unseen)
			matcontent_att_splits_new = self.matcontent_att_splits.copy()
			# get new seen_loc and unseen_loc from new splits
			seen_loc = np.where(np.in1d(self.labels, new_seen))[0]
			test_unseen_loc = np.where(np.in1d(self.labels, new_unseen))[0]
			# shuffle the seen locations, this is necessary because the labels are in order,
			# so without shuffling the training set and test set would have different classes
			np.random.shuffle(seen_loc)
			test_seen_loc = seen_loc[:int(self.test_seen_ratio * seen_loc.size)]
			train_loc = seen_loc[int(self.test_seen_ratio * seen_loc.size):]
			matcontent_att_splits_new['test_seen_loc'] = test_seen_loc + 1
			matcontent_att_splits_new['test_unseen_loc'] = test_unseen_loc + 1
			matcontent_att_splits_new['trainval_loc'] = train_loc + 1
			matcontent_att_splits_new['att'] = new_attributes.T
			# save new splits
			save_str = 'att_splits_rnd'
			if n_splits > 1:
				save_str += str(i)
			if n_seen > 0:
				save_str += '_seen' + str(n_seen)
			if n_unseen > 0:
				save_str += '_unseen' + str(n_unseen)
			if save:
				sio.savemat(self.path + '/' + save_str + '.mat', matcontent_att_splits_new)
				print('Saved')

	def random_split(self, inverse=False, n_seen=0, n_unseen=0):
		'''
		Splitting method: Random Split (RND)
		Usually used as a control split.
		'''
		old_seen = torch.from_numpy(np.unique(self.seen_labels))
		old_unseen = torch.from_numpy(np.unique(self.unseen_labels))
		old_classes = np.concatenate((old_seen, old_unseen))
		np.random.shuffle(old_classes)
		new_seen = old_classes[:int(self.n_seen_classes)]
		new_unseen = old_classes[int(self.n_seen_classes):]
		if n_seen > 0:
			new_seen = old_classes[:int(n_seen)]
			new_unseen = old_classes[int(n_seen):]
		if n_unseen > 0:
			new_unseen = old_classes[int(n_seen):int(n_seen + n_unseen)]
		return new_seen, new_unseen, self.attributes

	def greedy_class_split(self, inverse=False):
		'''
		Splitting method: Greedy Class Split (GCS)
		Tries to avoid the "horse with stripes without stripes images" scenario by keeping as much semantic information as possible among the seen classes.
		In the binary definition of the semantic space, the value 1 indicates the presence of an attribute in an image, while the value 0 indicates its absence.
		This means that ones are more useful than zeros, so we maximize the former in the seen classes split.
		In other words, for each class, we simply sum the values of its signature vector and we sort the classes by these sums in descending order.
		Consequently, we select the first Ns classes as seen classes, and the other Nu as unseen classes.
		'''
		# for each class, sum the values of its signature vector
		sums = np.sum(self.attributes, axis=1)
		# sorted_sums = np.sort(sums)
		sorted_sums = np.argsort(sums)
		new_seen = sorted_sums[:self.n_seen_classes] if inverse else sorted_sums[self.n_unseen_classes:]
		new_unseen = sorted_sums[self.n_seen_classes:] if inverse else sorted_sums[:self.n_unseen_classes]
		return new_seen, new_unseen, self.attributes

	def clustered_class_split(self, inverse=False):
		'''
		Splitting method: Clustered Class Split (CCS)
		Tries to maximize the Class Semantic Distance between seen classes and unseen classes.
		We define the Class Semantic Distance matrix where each element is the euclidean distance between class two class signatures (attribute vectors).
		Seen and unseen classes are defined by sorting the classes by the sum of their row (or column) values in descending order.
		The first Ns classes are those with the lowest distances overall, meaning that they form a cluster in the semantic space. Those classes will be the seen classes.
		The other Nu are far from this cluster in the semantic space, so they will form another cluster
		(although it is not a proper cluster since those classes are probably far away from each other as well), and they will be the unseen classes.
		'''
		distances = []
		for a1 in self.attributes:
			att_distances = []
			for a2 in self.attributes:
				d = np.linalg.norm(a1 - a2)
				att_distances.append(d)
			sum_att_distances = np.sum(att_distances)
			distances.append(sum_att_distances)
		sorted_distances = np.argsort(distances)			# from smaller to largest sum
		new_seen = sorted_distances[:self.n_seen_classes] if inverse else sorted_distances[self.n_unseen_classes:]
		new_unseen = sorted_distances[self.n_seen_classes:] if inverse else sorted_distances[:self.n_unseen_classes]
		return new_seen, new_unseen, self.attributes

	def minimal_attribute_split(self, inverse=False):
		'''
		Splitting method: Minimal Attribute Split (MAS)
		Removes unnecessary (i.e. highly correlated) attributes.
		We measure correlation between attributes i and j in a class as the ratio of co-occurrencies of i and j over i or j. Notice that this is not symmetric.
		'''
		# correlations is a list of size n_attributes and, for each attribute, it contains the sum of correlations
		correlations = []
		# attributes has shape (n_classes, n_attributes), transpose
		attributes_t = self.attributes.T
		for a1 in attributes_t:
			att_correlations = []
			for a2 in attributes_t:
				# for each pair of attributes, check if they are correlated (the more classes they have in common, the higher is the correlation)
				d = np.correlate(a1, a2)
				att_correlations.append(d)
			# for each attribute, sum the correlation values with all the other attributes
			sum_att_correlations = np.sum(att_correlations)
			correlations.append(sum_att_correlations)
		# sorted_correlations contains, for each attribute, its relative ranking among other attributes in sum of correlations (from smallest to largest)
		sorted_correlations = np.argsort(correlations)
		# define new attributes
		new_attributes = []
		for i in range(len(sorted_correlations)):
			# we want to keep only the least (most) correlated K attributes
			if (sorted_correlations[i] < self.mas_k and not inverse) or (sorted_correlations[i] >= (len(self.attributes[0]) - self.mas_k) and inverse):
				new_attributes.append(attributes_t[i])
		new_attributes = np.array(new_attributes)
		new_attributes = new_attributes.T
		print(new_attributes.shape)
		# seen and unseen remain the same, return new attributes
		old_seen = torch.from_numpy(np.unique(self.seen_labels))
		old_unseen = torch.from_numpy(np.unique(self.unseen_labels))
		return old_seen, old_unseen, new_attributes
