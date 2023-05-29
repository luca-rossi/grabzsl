import os
import torch
from grabzsl.losses import loss_grad_penalty_fn
from grabzsl.models import Generator, Critic
from grabzsl.trainer_classifier import TrainerClassifier

class TrainerClswgan():
	'''
	This class implements the training and evaluation of the CLSWGAN model.
	'''
	def __init__(self, data, dataset_name, pre_classifier, n_features=2048, n_attributes=85, latent_size=85, features_per_class=1800,
				batch_size=64, hidden_size=4096, n_epochs=30, n_classes=50, n_critic_iters=5, lr=0.001, lr_cls=0.001, beta1=0.5,
				weight_gp=10, weight_precls=1, save_every=0, device='cpu', verbose=False):
		'''
		Setup models, optimizers, and other parameters.
		'''
		self.data = data
		self.dataset_name = dataset_name
		self.pre_classifier = pre_classifier
		self.n_features = n_features
		self.n_attributes = n_attributes
		self.latent_size = latent_size
		self.features_per_class = features_per_class
		self.batch_size = batch_size
		self.hidden_size = hidden_size
		self.n_epochs = n_epochs
		self.n_classes = n_classes
		self.n_critic_iters = n_critic_iters
		self.lr_cls = lr_cls
		self.weight_gp = weight_gp
		self.weight_precls = weight_precls
		self.save_every = save_every
		self.device = device
		self.verbose = verbose
		# init models
		self.model_generator = Generator(n_features, n_attributes, latent_size, hidden_size).to(device)
		self.model_critic = Critic(n_features, n_attributes, hidden_size).to(device)
		print(self.model_generator)
		print(self.model_critic)
		# init optimizers
		self.opt_generator = torch.optim.Adam(self.model_generator.parameters(), lr=lr, betas=(beta1, 0.999))
		self.opt_critic = torch.optim.Adam(self.model_critic.parameters(), lr=lr, betas=(beta1, 0.999))
		# define pre-classifier loss
		self.loss_classifier_fn = torch.nn.NLLLoss().to(device)
		# init tensors
		self.batch_features = torch.FloatTensor(batch_size, n_features).to(device)
		self.batch_attributes = torch.FloatTensor(batch_size, n_attributes).to(device)
		self.batch_labels = torch.LongTensor(batch_size).to(device)
		self.batch_noise = torch.FloatTensor(batch_size, latent_size).to(device)
		self.one = torch.tensor(1, dtype=torch.float).to(device)
		self.mone = self.one * -1

	def fit(self):
		'''
		Train the model. Both ZSL and GZSL performance are evaluated at each epoch.
		'''
		self.best_gzsl_acc_seen = 0
		self.best_gzsl_acc_unseen = 0
		self.best_gzsl_acc_H = 0
		self.best_zsl_acc = 0
		start_epoch = self.__load_checkpoint()
		for epoch in range(start_epoch, self.n_epochs):
			self.__train_epoch(epoch)
			self.__eval_epoch()
			if self.save_every > 0 and epoch > 0 and (epoch + 1) % self.save_every == 0:
				self.__save_checkpoint(epoch)
		print('Dataset', self.dataset_name)
		print('The best ZSL unseen accuracy is %.4f' % self.best_zsl_acc.item())
		print('The best GZSL seen accuracy is %.4f' % self.best_gzsl_acc_seen.item())
		print('The best GZSL unseen accuracy is %.4f' % self.best_gzsl_acc_unseen.item())
		print('The best GZSL H is %.4f' % self.best_gzsl_acc_H.item())

	def __load_checkpoint(self):
		'''
		Load a checkpoint if it exists.
		'''
		start_epoch = 0
		checkpoints = [f for f in os.listdir('checkpoints') if f.startswith(f'CLSWGAN_{self.dataset_name}')]
		if len(checkpoints) > 0:
			print('Loading checkpoint...')
			checkpoint = torch.load(f'checkpoints/{checkpoints[0]}')
			start_epoch = checkpoint['epoch'] + 1
			self.model_generator.load_state_dict(checkpoint['model_generator'])
			self.model_critic.load_state_dict(checkpoint['model_critic'])
			self.opt_generator.load_state_dict(checkpoint['opt_generator'])
			self.opt_critic.load_state_dict(checkpoint['opt_critic'])
			self.best_gzsl_acc_seen = checkpoint['best_gzsl_acc_seen']
			self.best_gzsl_acc_unseen = checkpoint['best_gzsl_acc_unseen']
			self.best_gzsl_acc_H = checkpoint['best_gzsl_acc_H']
			self.best_zsl_acc = checkpoint['best_zsl_acc']
			torch.set_rng_state(checkpoint['random_state'])
			print('Checkpoint loaded.')
		return start_epoch

	def __save_checkpoint(self, epoch):
		'''
		Save a checkpoint.
		'''
		print('Saving checkpoint...')
		checkpoint = {
			'epoch': epoch,
			'model_generator': self.model_generator.state_dict(),
			'model_critic': self.model_critic.state_dict(),
			'opt_generator': self.opt_generator.state_dict(),
			'opt_critic': self.opt_critic.state_dict(),
			'best_gzsl_acc_seen': self.best_gzsl_acc_seen,
			'best_gzsl_acc_unseen': self.best_gzsl_acc_unseen,
			'best_gzsl_acc_H': self.best_gzsl_acc_H,
			'best_zsl_acc': self.best_zsl_acc,
			'random_state': torch.get_rng_state(),
		}
		torch.save(checkpoint, f'checkpoints/CLSWGAN_{self.dataset_name}.pt')
		print('Checkpoint saved.')
		
	def __train_epoch(self, epoch):
		'''
		Train the models for one epoch: train the critic for n_critic_iters steps, then train the generator for one step.
		'''
		for i in range(0, self.data.dataset_size, self.batch_size):
			# unfreeze the critic parameters for training
			for p in self.model_critic.parameters():
				p.requires_grad = True
			# train the critic for n_critic_iters steps
			for c in range(self.n_critic_iters):
				loss_critic, wasserstein = self.__critic_step()
			# freeze the critic parameters to train the generator
			for p in self.model_critic.parameters():
				p.requires_grad = False
			# train the generator
			loss_generator, loss_classifier = self.__generator_step()
			# show progress
			if self.verbose and i % (self.batch_size * 5) == 0:
				print('%d/%d' % (i, self.data.dataset_size))
		print('[%d/%d] Loss critic: %.4f Loss generator: %.4f, Wasserstein: %.4f, Loss classifier: %.4f'
				% (epoch + 1, self.n_epochs, loss_critic.data.item(), loss_generator.data.item(), wasserstein.data.item(), loss_classifier.data.item()))

	def __critic_step(self):
		'''
		Train the critic for one step on two mini-batches: a real one from the dataset, and a synthetic one from the generator.
		'''
		# sample a mini-batch
		self.batch_features, self.batch_labels, self.batch_attributes = self.data.next_batch(self.batch_size, device=self.device)
		# train with real batch
		self.model_critic.zero_grad()
		critic_real = self.model_critic(self.batch_features, self.batch_attributes).mean()
		critic_real.backward(self.mone)
		# train with synthetic batch
		self.batch_noise.normal_(0, 1)
		fake = self.model_generator(self.batch_noise, self.batch_attributes)
		critic_fake = self.model_critic(fake.detach(), self.batch_attributes).mean()
		critic_fake.backward(self.one)
		# gradient penalty
		gradient_penalty = loss_grad_penalty_fn(self.model_critic, self.batch_features, fake.data, self.batch_attributes, self.batch_size, self.weight_gp, self.device)
		gradient_penalty.backward()
		# loss
		wasserstein = critic_real - critic_fake
		loss_critic = -wasserstein + gradient_penalty
		self.opt_critic.step()
		return loss_critic, wasserstein
	
	def __generator_step(self):
		'''
		Train the generator for one step. Include the classification loss from the pre-classifier.
		'''
		# train the generator
		self.model_generator.zero_grad()
		self.batch_noise.normal_(0, 1)
		fake = self.model_generator(self.batch_noise, self.batch_attributes)
		critic_fake = self.model_critic(fake, self.batch_attributes).mean()
		loss_generator = -critic_fake
		# loss
		loss_classifier = self.loss_classifier_fn(self.pre_classifier.model(fake), self.batch_labels)
		loss_generator_tot = loss_generator + self.weight_precls * loss_classifier
		loss_generator_tot.backward()
		self.opt_generator.step()
		return loss_generator, loss_classifier

	def __eval_epoch(self):
		'''
		Evaluate the model at the end of each epoch, both ZSL and GZSL.
		The generator is used to generate unseen features, then a classifier is trained and evaluated on the (partially) synthetic dataset.
		'''
		# evaluation mode
		self.model_generator.eval()
		# generate synthetic features
		syn_X, syn_Y = self.data.generate_syn_features(self.model_generator, self.data.unseen_classes, self.data.attributes,
								self.features_per_class, self.n_features, self.n_attributes, self.latent_size, self.device)
		# GZSL evaluation: concatenate real seen features with synthesized unseen features, then train and evaluate a classifier
		train_X = torch.cat((self.data.train_X, syn_X), 0)
		train_Y = torch.cat((self.data.train_Y, syn_Y), 0)
		cls = TrainerClassifier(train_X, train_Y, self.data, batch_size=self.features_per_class, hidden_size=self.hidden_size,
				n_epochs=25, n_classes=self.n_classes, lr=self.lr_cls, beta1=0.5, device=self.device)
		acc_seen, acc_unseen, acc_H = cls.fit_gzsl()
		if self.best_gzsl_acc_H < acc_H:
			self.best_gzsl_acc_seen, self.best_gzsl_acc_unseen, self.best_gzsl_acc_H = acc_seen, acc_unseen, acc_H
		print('GZSL: Seen: %.4f, Unseen: %.4f, H: %.4f' % (acc_seen, acc_unseen, acc_H))
		# ZSL evaluation: use only synthesized unseen features, then train and evaluate a classifier
		train_X = syn_X
		train_Y = self.data.map_labels(syn_Y, self.data.unseen_classes)
		cls = TrainerClassifier(train_X, train_Y, self.data, batch_size=self.features_per_class, hidden_size=self.hidden_size,
				n_epochs=25, n_classes=self.data.unseen_classes.size(0), lr=self.lr_cls, beta1=0.5, device=self.device)
		acc = cls.fit_zsl()
		if self.best_zsl_acc < acc:
			self.best_zsl_acc = acc
		print('ZSL: Unseen: %.4f' % (acc))
		# training mode
		self.model_generator.train()
