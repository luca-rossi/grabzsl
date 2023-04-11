import torch
import torch.nn as nn

def init_weights(m):
	'''
	Initialize weights.
	'''
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		m.weight.data.normal_(0.0, 0.02)
		m.bias.data.fill_(0)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

class Classifier(nn.Module):
	'''
	Standard log softmax classifier.
	'''
	def __init__(self, input_dim, n_classes):
		super(Classifier, self).__init__()
		self.fc = nn.Linear(input_dim, n_classes)
		self.log = nn.LogSoftmax(dim=1)
		self.apply(init_weights)

	def forward(self, x):
		h = self.fc(x)
		h = self.log(h)
		return h

class Critic(nn.Module):
	'''
	Critic network conditioned on attributes: takes in a feature vector and an attribute vector and outputs a "realness" score.
	'''
	def __init__(self, n_features, n_attributes, hidden_size=4096):
		super(Critic, self).__init__()
		self.fc1 = nn.Linear(n_features + n_attributes, hidden_size)
		self.fc2 = nn.Linear(hidden_size, 1)
		self.lrelu = nn.LeakyReLU(0.2, True)
		self.apply(init_weights)

	def forward(self, x, att):
		h = torch.cat((x, att), dim=1)
		h = self.lrelu(self.fc1(h))
		h = self.fc2(h)
		return h

class Generator(nn.Module):
	'''
	Generator network conditioned on attributes: takes in a noise vector and an attribute vector and outputs a feature vector.
	Its hidden layer can optionally take in a feedback vector to improve the quality of the generated features.
	'''
	def __init__(self, n_features, n_attributes, latent_size, hidden_size=4096, use_sigmoid=False):
		super(Generator, self).__init__()
		self.fc1 = nn.Linear(n_attributes + latent_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, n_features)
		self.lrelu = nn.LeakyReLU(0.2, True)
		self.activation = nn.Sigmoid() if use_sigmoid else nn.ReLU(True)
		self.apply(init_weights)

	def forward(self, noise, att, feedback_weight=None, feedback=None):
		h = torch.cat((noise, att), dim=1)
		h = self.lrelu(self.fc1(h))
		if feedback is not None:
			h = h + feedback_weight * feedback
		h = self.activation(self.fc2(h))
		return h

class Decoder(nn.Module):
	'''
	Semantic embedding decoder network: takes in a feature vector and outputs an attribute vector.
	It learns to reconstruct the attribute vector using a cycle consistency reconstruction loss.
	Its hidden representation is passed to the feedback module.
	'''
	def __init__(self, n_features, n_attributes, hidden_size=4096):
		super(Decoder, self).__init__()
		self.fc1 = nn.Linear(n_features, hidden_size)
		self.fc2 = nn.Linear(hidden_size, n_attributes)
		self.lrelu = nn.LeakyReLU(0.2, True)
		# define the hidden layer to detach for the feedback module
		self.hidden_features = None
		self.apply(init_weights)

	def forward(self, x):
		self.hidden_features = self.lrelu(self.fc1(x))
		h = self.fc2(self.hidden_features)
		h = h / h.pow(2).sum(1).sqrt().unsqueeze(1).expand(h.size(0), h.size(1))
		return h

	def get_hidden_features(self):
		return self.hidden_features.detach()

class FRDecoder(nn.Module):
	'''
	Decoder network used in the FREE model. It produces a large vector h of dimension n_attributes * 2,
	where the first half learns to generate centroids (as the SAMC loss is applied to it).
	So these two halves encode means and standard deviations, and are used to reconstruct the attribute vector.
	'''
	def __init__(self, n_features, n_attributes, hidden_size=4096):
		super(FRDecoder, self).__init__()
		self.n_attributes = n_attributes
		self.fc1 = nn.Linear(n_features, hidden_size)
		self.fc2 = nn.Linear(hidden_size, n_attributes * 2)
		self.lrelu = nn.LeakyReLU(0.2, True)
		self.sigmoid = nn.Sigmoid()
		# define the hidden layer to detach for the feedback module
		self.hidden_features = None
		self.apply(init_weights)

	def forward(self, x):
		self.hidden_features = self.lrelu(self.fc1(x))
		h = self.fc2(self.hidden_features)
		means, stds = h[:, :self.n_attributes], h[:, self.n_attributes:]
		stds = self.sigmoid(stds)
		h = torch.randn_like(means) * stds + means
		h = self.sigmoid(h)
		return means, h

	def get_hidden_features(self):
		return self.hidden_features.detach()

class Encoder(nn.Module):
	'''
	VAE encoder network: takes in a feature vector and an attribute vector and outputs a distribution over the latent space.
	This distribution is used to sample the latent vector for the generator.
	'''
	def __init__(self, n_features, n_attributes, latent_size, hidden_size=4096):
		super(Encoder,self).__init__()
		self.fc1 = nn.Linear(n_features + n_attributes, hidden_size)
		self.fc2 = nn.Linear(hidden_size, latent_size * 2)
		self.lrelu = nn.LeakyReLU(0.2, True)
		self.linear_mean = nn.Linear(latent_size * 2, latent_size)
		self.linear_log_var = nn.Linear(latent_size * 2, latent_size)
		self.apply(init_weights)

	def forward(self, x, att):
		h = torch.cat((x, att), dim=1)
		h = self.lrelu(self.fc1(h))
		h = self.lrelu(self.fc2(h))
		mean = self.linear_mean(h)
		log_var = self.linear_log_var(h)
		return mean, log_var

class Feedback(nn.Module):
	'''
	Feedback module: takes in a hidden representation from the decoder and outputs a feedback vector for the generator.
	'''
	def __init__(self, hidden_size=4096):
		super(Feedback, self).__init__()
		self.fc1 = nn.Linear(hidden_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.lrelu = nn.LeakyReLU(0.2, True)
		self.apply(init_weights)

	def forward(self, x):
		h = self.lrelu(self.fc1(x))
		h = self.lrelu(self.fc2(h))
		return h
