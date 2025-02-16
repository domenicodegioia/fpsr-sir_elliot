import argparse
from elliot.run import run_experiment

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='Name of the dataset')
parser.add_argument('--model', type=str, help='Name of the model')

args = parser.parse_args()

dataset = args.dataset if args.dataset is not None else 'gowalla'
model = args.model if args.model is not None else 'fpsr'

run_experiment(f"config_files/test_{model}_{dataset}.yml")
