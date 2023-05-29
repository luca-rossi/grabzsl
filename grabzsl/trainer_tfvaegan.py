import os
import torch
from grabzsl.losses import loss_grad_penalty_fn, loss_vae_fn, loss_reconstruction_fn
from grabzsl.models import Encoder, Generator, Critic, Feedback, Decoder
from grabzsl.trainer_classifier import TrainerClassifier

class TrainerTfvaegan():
	'''
	This class implements the training and evaluation of the TFVAEGAN model.
	'''
	def __init__(self, data, dataset_name, n_features=2048, n_attributes=85, latent_size=85, features_per_class=1800,
				batch_size=64, hidden_size=4096, n_epochs=30, n_classes=50, n_critic_iters=5, n_loops=2,
				lr=0.001, lr_feedback=0.0001, lr_decoder=0.0001, lr_cls=0.001, beta1=0.5, freeze_dec=False,
				weight_gp=10, weight_critic=0.1, weight_generator=0.1, weight_feed_train=0.1, weight_feed_eval=0.1, weight_recons=1.0,
				save_every=0, device='cpu', verbose=False):
		'''
		Setup models, optimizers, and other parameters.
		'''
		self.data = data
		self.dataset_name = dataset_name
		self.n_features = n_features
		self.n_attributes = n_attributes
		self.latent_size = latent_size
		self.features_per_class = features_per_class
		self.batch_size = batch_size
		self.hidden_size = hidden_size
		self.n_epochs = n_epochs
		self.n_classes = n_classes
		self.n_critic_iters = n_critic_iters
		self.n_loops = n_loops
		self.lr_cls = lr_cls
		self.freeze_dec = freeze_dec
		self.weight_gp = weight_gp
		self.weight_critic = weight_critic
		self.weight_generator = weight_generator
		self.weight_feed_train = weight_feed_train
		self.weight_feed_eval = weight_feed_eval
		self.weight_recons = weight_recons
		self.save_every = save_every
		self.device = device
		self.verbose = verbose
		# init models
		self.model_encoder = Encoder(n_features, n_attributes, latent_size, hidden_size).to(device)
		self.model_generator = Generator(n_features, n_attributes, latent_size, hidden_size, use_sigmoid=True).to(device)
		self.model_critic = Critic(n_features, n_attributes, hidden_size).to(device)
		self.model_feedback = Feedback(hidden_size).to(device)
		self.model_decoder = Decoder(n_features, n_attributes, hidden_size).to(device)
		print(self.model_encoder)
		print(self.model_generator)
		print(self.model_critic)
		print(self.model_feedback)
		print(self.model_decoder)
		# init optimizers
		self.opt_encoder = torch.optim.Adam(self.model_encoder.parameters(), lr=lr)
		self.opt_generator = torch.optim.Adam(self.model_generator.parameters(), lr=lr, betas=(beta1, 0.999))
		self.opt_critic = torch.optim.Adam(self.model_critic.parameters(), lr=lr, betas=(beta1, 0.999))
		self.opt_feedback = torch.optim.Adam(self.model_feedback.parameters(), lr=lr_feedback, betas=(beta1, 0.999))
		self.opt_decoder = torch.optim.Adam(self.model_decoder.parameters(), lr=lr_decoder, betas=(beta1, 0.999))
		# init tensors
		self.batch_features = torch.FloatTensor(batch_size, n_features).to(device)
		self.batch_attributes = torch.FloatTensor(batch_size, n_attributes).to(device)
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
		checkpoints = [f for f in os.listdir('checkpoints') if f.startswith(f'TFVAEGAN_{self.dataset_name}')]
		if len(checkpoints) > 0:
			print('Loading checkpoint...')
			checkpoint = torch.load(f'checkpoints/{checkpoints[0]}')
			start_epoch = checkpoint['epoch'] + 1
			self.model_encoder.load_state_dict(checkpoint['model_encoder'])
			self.model_generator.load_state_dict(checkpoint['model_generator'])
			self.model_critic.load_state_dict(checkpoint['model_critic'])
			self.model_feedback.load_state_dict(checkpoint['model_feedback'])
			self.model_decoder.load_state_dict(checkpoint['model_decoder'])
			self.opt_encoder.load_state_dict(checkpoint['opt_encoder'])
			self.opt_generator.load_state_dict(checkpoint['opt_generator'])
			self.opt_critic.load_state_dict(checkpoint['opt_critic'])
			self.opt_feedback.load_state_dict(checkpoint['opt_feedback'])
			self.opt_decoder.load_state_dict(checkpoint['opt_decoder'])
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
			'model_encoder': self.model_encoder.state_dict(),
			'model_generator': self.model_generator.state_dict(),
			'model_critic': self.model_critic.state_dict(),
			'model_feedback': self.model_feedback.state_dict(),
			'model_decoder': self.model_decoder.state_dict(),
			'opt_encoder': self.opt_encoder.state_dict(),
			'opt_generator': self.opt_generator.state_dict(),
			'opt_critic': self.opt_critic.state_dict(),
			'opt_feedback': self.opt_feedback.state_dict(),
			'opt_decoder': self.opt_decoder.state_dict(),
			'best_gzsl_acc_seen': self.best_gzsl_acc_seen,
			'best_gzsl_acc_unseen': self.best_gzsl_acc_unseen,
			'best_gzsl_acc_H': self.best_gzsl_acc_H,
			'best_zsl_acc': self.best_zsl_acc,
			'random_state': torch.get_rng_state(),
		}
		torch.save(checkpoint, f'checkpoints/TFVAEGAN_{self.dataset_name}.pt')
		print('Checkpoint saved.')

	def __train_epoch(self, epoch):
		'''
		Train the models for one epoch. Each epoch consists of n_loops feedback loops.
		During each feedback loop, the critic and decoder are trained for n_critic_iters steps,
		then the generator is trained for one step with the encoder, the feedback module, and optionally the decoder.
		'''
		for loop in range(0, self.n_loops):
			for i in range(0, self.data.dataset_size, self.batch_size):
				# unfreeze the critic and decoder parameters for training
				for p in self.model_critic.parameters():
					p.requires_grad = True
				for p in self.model_decoder.parameters():
					p.requires_grad = True
				# train the decoder and critic for n_critic_iters steps, dynamically adjust weight_gp
				gp_sum = 0
				for _ in range(self.n_critic_iters):
					loss_critic, wasserstein, gp_incr = self.__decoder_critic_step(loop)
					gp_sum += gp_incr
				gp_sum /= (self.weight_critic * self.weight_gp * self.n_critic_iters)
				self.__adjust_weight_gp(gp_sum)
				# freeze the critic and decoder parameters to train the generator
				for p in self.model_critic.parameters():
					p.requires_grad = False
				if self.weight_recons > 0 and self.freeze_dec:
					for p in self.model_decoder.parameters():
						p.requires_grad = False
				# train the generator, the encoder, the feedback module, and optionally the decoder
				loss_generator, loss_vae = self.__encoder_generator_step(loop)
				# show progress
				if self.verbose and i % (self.batch_size * 5) == 0:
					print('%d/%d - %d/%d' % (loop + 1, self.n_loops, i, self.data.dataset_size))
		print('[%d/%d] Loss critic: %.4f, Loss generator: %.4f, Wasserstein: %.4f, Loss VAE: %.4f'
				% (epoch + 1, self.n_epochs, loss_critic.data.item(), loss_generator.data.item(), wasserstein.data.item(), loss_vae.data.item()))

	def __decoder_critic_step(self, loop):
		'''
		Train the decoder and the critic for one step. The decoder is trained to reconstruct the attributes of the real batch.
		The critic is trained on two mini-batches: a real one from the dataset, and a synthetic one from the generator.
		'''
		# sample a mini-batch
		self.batch_features, _, self.batch_attributes = self.data.next_batch(self.batch_size, device=self.device)
		# train decoder to reconstruct the attributes of the real batch with a reconstruction loss
		self.model_decoder.zero_grad()
		recons = self.model_decoder(self.batch_features)
		# reconstruction loss
		loss_recons = self.weight_recons * loss_reconstruction_fn(recons, self.batch_attributes)
		loss_recons.backward()
		self.opt_decoder.step()
		# train critic with real batch
		self.model_critic.zero_grad()
		critic_real = self.model_critic(self.batch_features, self.batch_attributes)
		critic_real = self.weight_critic * critic_real.mean()
		critic_real.backward(self.mone)
		# train critic with fake batch
		fake, _, _ = self.__generate_from_features(loop)
		critic_fake = self.model_critic(fake.detach(), self.batch_attributes)
		critic_fake = self.weight_critic * critic_fake.mean()
		critic_fake.backward(self.one)
		# gradient penalty
		gradient_penalty = self.weight_critic * loss_grad_penalty_fn(self.model_critic, self.batch_features, fake.data, self.batch_attributes, self.batch_size, self.weight_gp, self.device)
		gradient_penalty.backward()
		# loss
		wasserstein = critic_real - critic_fake
		loss_critic = -wasserstein + gradient_penalty
		self.opt_critic.step()
		return loss_critic, wasserstein, gradient_penalty.data

	def __encoder_generator_step(self, loop):
		'''
		Train the encoder, the generator, the feedback module, and optionally the decoder for one step.
		The generator is trained to fool the critic, and the encoder learns a distribution over a latent space for the generator.
		'''
		self.model_encoder.zero_grad()
		self.model_generator.zero_grad()
		self.model_feedback.zero_grad()
		# generate a fake batch from a latent distribution, learned from real features
		recon_x, means, log_var = self.__generate_from_features(loop)
		# VAE loss
		loss_vae = loss_vae_fn(recon_x, self.batch_features, means, log_var)
		# generator loss from the critic's evaluation
		critic_fake = self.model_critic(recon_x,self.batch_attributes).mean()
		loss_generator = -critic_fake
		# decoder and reconstruction loss
		self.model_decoder.zero_grad()
		recons_fake = self.model_decoder(recon_x)
		loss_recons = loss_reconstruction_fn(recons_fake, self.batch_attributes)
		# total loss
		loss_generator_tot = loss_vae - self.weight_generator * critic_fake + self.weight_recons * loss_recons
		loss_generator_tot.backward()
		self.opt_encoder.step()
		self.opt_generator.step()
		if loop >= 1:
			self.opt_feedback.step()
		if self.weight_recons > 0 and not self.freeze_dec:
			self.opt_decoder.step()
		return loss_generator, loss_vae

	def __generate_from_features(self, loop):
		'''
		Use the generator to synthesize a batch from a latent distribution, which is learned from real features by the decoder.
		Improve the generated features with the feedback module from the second feedback loop onward.
		'''
		# use real features to generate a latent distribution with the encoder
		means, log_var = self.model_encoder(self.batch_features, self.batch_attributes)
		std = torch.exp(0.5 * log_var)
		eps = torch.randn([self.batch_size, self.latent_size]).to(self.device)
		noise = eps * std + means
		# generate a fake batch with the generator from the latent distribution
		fake = self.model_generator(noise, self.batch_attributes)
		# from the second feedback loop onward, improve the generated features with the feedback module
		if loop >= 1:
			# call the forward function of decoder to get the hidden features
			_ = self.model_decoder(fake)
			decoder_features = self.model_decoder.get_hidden_features()
			feedback = self.model_feedback(decoder_features)
			fake = self.model_generator(noise, self.batch_attributes, feedback_weight=self.weight_feed_train, feedback=feedback)
		return fake, means, log_var

	def __adjust_weight_gp(self, gp_sum):
		'''
		Dynamically adjust the weight of the gradient penalty to keep it in a reasonable range.
		'''
		if (gp_sum > 1.05).sum() > 0:
			self.weight_gp *= 1.1
		elif (gp_sum < 1.001).sum() > 0:
			self.weight_gp /= 1.1

	def __eval_epoch(self):
		'''
		Evaluate the model at the end of each epoch, both ZSL and GZSL.
		The generator is used to generate unseen features, then a classifier is trained and evaluated on the (partially) synthetic dataset.
		'''
		# evaluation mode
		self.model_generator.eval()
		self.model_decoder.eval()
		self.model_feedback.eval()
		# generate synthetic features
		syn_X, syn_Y = self.data.generate_syn_features(self.model_generator, self.data.unseen_classes, self.data.attributes,
								self.features_per_class, self.n_features, self.n_attributes, self.latent_size, self.device,
								self.model_feedback, self.model_decoder, feedback_weight=self.weight_feed_eval)
		# GZSL evaluation: concatenate real seen features with synthesized unseen features, then train and evaluate a classifier
		train_X = torch.cat((self.data.train_X, syn_X), 0)
		train_Y = torch.cat((self.data.train_Y, syn_Y), 0)
		cls = TrainerClassifier(train_X, train_Y, self.data, n_attributes=self.n_attributes, batch_size=self.features_per_class,
				hidden_size=self.hidden_size, n_epochs=25, n_classes=self.n_classes, lr=self.lr_cls, beta1=0.5,
				model_decoder=self.model_decoder, device=self.device)
		acc_seen, acc_unseen, acc_H = cls.fit_gzsl()
		if self.best_gzsl_acc_H < acc_H:
			self.best_gzsl_acc_seen, self.best_gzsl_acc_unseen, self.best_gzsl_acc_H = acc_seen, acc_unseen, acc_H
		print('GZSL: Seen: %.4f, Unseen: %.4f, H: %.4f' % (acc_seen, acc_unseen, acc_H))
		# ZSL evaluation: use only synthesized unseen features, then train and evaluate a classifier
		train_X = syn_X
		train_Y = self.data.map_labels(syn_Y, self.data.unseen_classes)
		cls = TrainerClassifier(train_X, train_Y, self.data, n_attributes=self.n_attributes, batch_size=self.features_per_class,
				hidden_size=self.hidden_size, n_epochs=25, n_classes=self.data.unseen_classes.size(0), lr=self.lr_cls, beta1=0.5,
				model_decoder=self.model_decoder, device=self.device)
		acc = cls.fit_zsl()
		if self.best_zsl_acc < acc:
			self.best_zsl_acc = acc
		print('ZSL: Unseen: %.4f' % (acc))
		# training mode
		self.model_generator.train()
		self.model_decoder.train()
		self.model_feedback.train()
