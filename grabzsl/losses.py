import torch
import torch.nn as nn
import torch.autograd as autograd

def loss_grad_penalty_fn(model, batch_real, batch_fake, batch_attributes, batch_size, weight_gp, device='cpu'):
	'''
	Gradient penalty loss.
	'''
	alpha = torch.rand(batch_size, 1)
	alpha = alpha.expand(batch_real.size()).to(device)
	interpolated = (alpha * batch_real + ((1 - alpha) * batch_fake)).requires_grad_(True)
	pred_interpolated = model(interpolated, batch_attributes)
	ones = torch.ones(pred_interpolated.size()).to(device)
	gradients = autograd.grad(outputs=pred_interpolated, inputs=interpolated, grad_outputs=ones,
			   create_graph=True, retain_graph=True, only_inputs=True)[0]
	gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * weight_gp
	return gradient_penalty

def loss_vae_fn(recon_x, x, mean, log_var):
	'''
	VAE loss.
	'''
	bce = nn.functional.binary_cross_entropy(recon_x + 1e-12, x.detach(), reduction='sum')
	bce = bce.sum() / x.size(0)
	kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0)
	return (bce + kld)

def loss_reconstruction_fn(pred, gt):
	'''
	Weighted reconstruction l1 loss.
	'''
	wt = (pred - gt).pow(2)
	wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0), wt.size(1))
	loss = wt * (pred - gt).abs()
	return loss.sum() / loss.size(0)

class LossMarginCenter(nn.Module):
	"""
	Self-Adaptive Margin Center loss.
	It aims minimize the distances between samples of the same class and maximize the distances between samples of different classes,
	and does so by learning a set of label centers.
	"""
	def __init__(self, n_classes=10, n_attributes=312, min_margin=False, device='cpu'):
		super(LossMarginCenter, self).__init__()
		self.n_classes = n_classes
		self.n_attributes = n_attributes
		self.min_margin = min_margin
		self.device = device
		self.centers = nn.Parameter(torch.randn(self.n_classes, self.n_attributes).to(self.device))

	def forward(self, x, labels, margin, weight_center):
		batch_size = x.size(0)
		# compute the distance matrix between the input samples and the centers
		all_distances = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.n_classes) + \
					torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.n_classes, batch_size).t()
		all_distances.addmm_(x, self.centers.t(), beta=1, alpha=-2)
		classes = torch.arange(self.n_classes).long().to(self.device)
		mask_labels = labels.unsqueeze(1).expand(batch_size, self.n_classes)
		mask = mask_labels.eq(classes.expand(batch_size, self.n_classes))
		# distance between the features and the class center of their label
		distances = all_distances[mask]
		if not self.min_margin:
			# if not using the minimum margin, compute the "other" labels to be used for computing the margin
			index = torch.randint(self.n_classes, (labels.shape[0],)).to(labels.device)
			other_labels = labels + index
			other_labels[other_labels >= self.n_classes] = other_labels[other_labels >= self.n_classes] - self.n_classes
			other_labels = other_labels.unsqueeze(1).expand(batch_size, self.n_classes)
			mask_other = other_labels.eq(classes.expand(batch_size, self.n_classes))
			# distance between the features and the class center of another (random) label
			other_distances = all_distances[mask_other]
		else:
			# if using the minimum margin, compute the "other" distances for each sample and take the minimum
			other = torch.FloatTensor(batch_size, self.n_classes - 1).cuda()
			for i in range(batch_size):
				other[i] = (all_distances[i, mask[i, :] == 0])
			# distance between the features and the class center of the minimum distance label
			other_distances, _ = other.min(dim=1)
		# compute the loss
		loss = torch.max(margin + weight_center * distances - (1 - weight_center) * other_distances, torch.tensor(0.0).cuda()).sum() / batch_size
		return loss
