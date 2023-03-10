import argparse
from grabzsl.splitter import Splitter

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', type=str, default='../data/FLO', help='path to the dataset folder')
parser.add_argument('--split', '-s', type=str, default=None, help='type of split to use for the new dataset generation (\'rnd\', \'gcs\', \'ccs\', \'mas\', \'mcs\', \'pca\')')
parser.add_argument('--inverse', '-i', action='store_true', help='use inverse split')
parser.add_argument('--mas_k', '-m', type=int, default=40, help='number of attributes to keep for MAS split')
parser.add_argument('--all', '-a', action='store_true', help='if set, all splits will be generated')
parser.add_argument('--rnd', '-r', action='store_true', help='if set, reproducible random splits will be generated')
args = parser.parse_args()

splitter = Splitter(args.path, mas_k=args.mas_k)
if args.all:
	splitter.generate_all_splits()
if args.rnd:
	splitter.generate_random_splits(n_splits=5)#, n_seen=700)#, n_unseen=25)
if args.split is not None:
	splitter.generate_split(args.split, inverse=args.inverse)
