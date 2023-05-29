import random
import torch
import torch.backends.cudnn as cudnn
from args import parse_args
from grabzsl.data import Data
from grabzsl.trainer_free import TrainerFree

# parse arguments
args = parse_args('FREE')
# init seed and cuda
if args.seed is None:
	args.seed = random.randint(1, 10000)
print('Split:', ('none' if args.split == '' else args.split))
print('Random Seed:', args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
	torch.cuda.manual_seed_all(args.seed)
cudnn.benchmark = True
cudnn.deterministic = True
# load data
data = Data(dataset_name=args.dataset, split=args.split, dataroot=args.dataroot)
# define center loss type depending on the type of dataset
min_margin = args.dataset == 'AWA2'
# train the FREE model
free = TrainerFree(data, args.dataset, n_features=args.n_features, n_attributes=data.get_n_attributes(),
				latent_size=data.get_n_attributes(), features_per_class=args.features_per_class, batch_size=args.batch_size,
				hidden_size=args.hidden_size, n_epochs=args.n_epochs, n_classes=data.get_n_classes(),
				n_critic_iters=args.n_critic_iters, n_loops=args.n_loops,
				lr=args.lr, lr_cls=args.lr_cls, beta1=args.beta1, freeze_dec=args.freeze_dec,
				weight_gp=args.weight_gp, weight_critic=args.weight_critic, weight_generator=args.weight_generator,
				center_margin=args.center_margin, weight_margin=args.weight_margin, weight_center=args.weight_center,
				weight_recons=args.weight_recons, min_margin=min_margin, save_every=args.save_every, device=device)
free.fit()
