import torch
import torch.autograd as autograd

def loss_grad_penalty_fn(model_critic, batch_real, batch_fake, batch_attributes, batch_size, weight_gp, device='cpu'):
	'''
	Gradient penalty loss.
	'''
	alpha = torch.rand(batch_size, 1)
	alpha = alpha.expand(batch_real.size()).to(device)
	interpolated = (alpha * batch_real + ((1 - alpha) * batch_fake)).requires_grad_(True)
	pred_interpolated = model_critic(interpolated, batch_attributes)
	ones = torch.ones(pred_interpolated.size()).to(device)
	gradients = autograd.grad(outputs=pred_interpolated, inputs=interpolated, grad_outputs=ones,
			   create_graph=True, retain_graph=True, only_inputs=True)[0]
	gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * weight_gp
	return gradient_penalty

def loss_vae_fn(recon_x, x, mean, log_var):
	'''
	VAE loss.
	'''
	bce = torch.nn.functional.binary_cross_entropy(recon_x + 1e-12, x.detach(), reduction='none')
	bce = bce.sum() / x.size(0)
	kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0)
	return (bce + kld)

def loss_reconstruction_fn(pred, gt):
	'''
	Weighted reconstruction l1 loss
	'''
	wt = (pred - gt).pow(2)
	wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0), wt.size(1))
	loss = wt * (pred - gt).abs()
	return loss.sum() / loss.size(0)
