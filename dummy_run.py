import argparse
from elliot.run import run_experiment

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='Name of the dataset')
args = parser.parse_args()

dataset = args.dataset if args.dataset is not None else 'yelp2018'

run_experiment(f"config_files/test_fpsr_{dataset}.yml")
