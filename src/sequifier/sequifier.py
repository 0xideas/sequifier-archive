import numpy as np

from argparse import ArgumentParser
from sequifier.preprocess import preprocess
from sequifier.train import train
from sequifier.infer import infer


def build_args_config(args):
    args_config = {"project_path": args.project_path}

    if args.randomize:
        seed = np.random.choice(np.arange(int(1e9)))
        args_config["seed"] = seed
    else:
        args_config["seed"] = 1010

    return(args_config)


def main():
    parser = ArgumentParser()

    parser.add_argument("--project_path", type=str, help="file path to folder that will contain the output paths")
    parser.add_argument("--config_path", type=str, help="file path to config for current processing step")
    parser.add_argument('-p', '--preprocess', action='store_true') 
    parser.add_argument('-t', '--train', action='store_true') 
    parser.add_argument('-op', '--on-preprocessed', action='store_true') 
    parser.add_argument('-i', '--infer', action='store_true') 
    parser.add_argument('-r', '--randomize', action='store_true') 

    args = parser.parse_args()

    args_config = build_args_config(args)

    if args.preprocess:
        preprocess(args, args_config)
    if args.train:
        train(args, args_config)
    if args.infer:
        infer(args, args_config)


if __name__ == "__main__":
    main()