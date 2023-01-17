from argparse import ArgumentParser
from preprocess import preprocess
from train import train
from infer import infer


def main(args):
    if args.preprocess:
        preprocess(args)
    if args.train:
        train(args)
    if args.infer:
        infer(args)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--project_path", type=str, help="file path to folder that will contain the output paths")
    parser.add_argument("--config_path", type=str, help="file path to config for current processing step")
    parser.add_argument('-p', '--preprocess', action='store_true') 
    parser.add_argument('-t', '--train', action='store_true') 
    parser.add_argument('-op', '--on-preprocessed', action='store_true') 
    parser.add_argument('-i', '--infer', action='store_true') 

    args = parser.parse_args()
    main(args)