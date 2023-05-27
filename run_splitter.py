import argparse
import os
from grabzsl.splitter import Splitter

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', default='FLO', help='dataset name (folder containing the res101.mat and att_splits.mat files)')
parser.add_argument('--dataroot', '-p', default='../data', help='path to dataset')
parser.add_argument('--split', '-s', type=str, default=None, help='type of split to use for the new dataset generation (\'gcs\', \'ccs\', \'mas\')')
parser.add_argument('--inverse', '-i', action='store_true', help='use inverse split')
parser.add_argument('--mas_k', '-m', type=int, default=40, help='number of attributes to keep for MAS split')
parser.add_argument('--pas_k', '-k', type=int, default=40, help='number of attributes to keep for PAS split')
parser.add_argument('--all', '-a', action='store_true', help='if set, all splits will be generated')
parser.add_argument('--n_rnd', '-r', type=int, default=5, help='if higher than 0, n_rnd random splits will be generated')
parser.add_argument('--n_seen', type=int, default=0, help='number of seen classes for random split (set to 0 to keep the original number of seen classes)')
parser.add_argument('--n_unseen', type=int, default=0, help='number of unseen classes for random split (set to 0 to keep the original number of unseen classes)')
args = parser.parse_args()

path = os.path.join(args.dataroot, args.dataset)
splitter = Splitter(path, mas_k=args.mas_k, pas_k=args.pas_k)
if args.all:
	splitter.generate_all_splits()
if args.n_rnd > 0:
	splitter.generate_random_splits(n_splits=args.n_rnd, n_seen=args.n_seen, n_unseen=args.n_unseen)
if args.split is not None:
	splitter.generate_split(args.split, inverse=args.inverse)
