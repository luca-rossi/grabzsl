import random
import torch
import torch.backends.cudnn as cudnn
from args import args
from grabzsl.data import Data
from grabzsl.trainer_tfvaegan import TrainerTfvaegan

# init seed and cuda
if args.seed is None:
	args.seed = random.randint(1, 10000)
print('Split: ', ('none' if args.split == '' else args.split))
print('Random Seed: ', args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
	torch.cuda.manual_seed_all(args.seed)
cudnn.benchmark = True
# load data
data = Data(dataset_name=args.dataset, split=args.split, dataroot=args.dataroot)
print('Training samples: ', data.dataset_size)
# train the TF-VAEGAN
tfvaegan = TrainerTfvaegan(data, args.dataset, n_features=args.n_features, n_attributes=args.n_attributes,
			 			latent_size=args.latent_size, features_per_class=args.features_per_class, batch_size=args.batch_size,
						hidden_size=args.hidden_size, n_epochs=args.n_epochs, n_classes=args.n_classes,
						n_critic_loops=args.n_critic_loops, n_feedback_loops=args.n_feedback_loops,
						lr=args.lr, lr_cls=args.lr_cls, beta1=args.beta1, freeze_dec=args.freeze_dec,
						weight_gp=args.weight_gp, weight_critic=args.weight_critic, weight_generator=args.weight_generator,
						weight_feed_train=args.weight_feed_train, weight_feed_eval=args.weight_feed_eval,
						weight_recons=args.weight_recons, device=device)
tfvaegan.fit()
