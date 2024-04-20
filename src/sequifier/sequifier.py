from argparse import ArgumentParser

import numpy as np

from sequifier.infer import infer
from sequifier.preprocess import preprocess
from sequifier.train import train


def build_args_config(args):
    args_config = {
        k: v for k, v in vars(args).items() if v is not None and k != "randomize"
    }
    if args.randomize:
        seed = np.random.choice(np.arange(int(1e9)))
        args_config["seed"] = seed
    else:
        args_config["seed"] = 1010

    if "selected_columns" in args_config:
        if args_config["selected_columns"] == "None":
            args_config["selected_columns"] = None
        else:
            args_config["selected_columns"] = (
                args_config["selected_columns"].replace(" ", "").split(",")
            )

    return args_config


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    parser_preprocess = subparsers.add_parser("preprocess", help="Run the process")
    parser_train = subparsers.add_parser("train", help="Run the process")
    parser_infer = subparsers.add_parser("infer", help="Run the process")

    for subparser in [parser_preprocess, parser_train, parser_infer]:
        subparser.add_argument(
            "--config-path",
            type=str,
            help="file path to config for current processing step",
        )
        subparser.add_argument("-r", "--randomize", action="store_true")
        subparser.add_argument("-sc", "--selected-columns", type=str)
        subparser.add_argument("-dp", "--data-path", type=str)

    for subparser in [parser_train, parser_infer]:
        subparser.add_argument("-ddcp", "--ddconfig-path", type=str)
        subparser.add_argument("-op", "--on-unprocessed", action="store_true")

    parser_train.add_argument("-mn", "--model-name", type=str)
    parser_infer.add_argument("-imp", "--model-path", type=str)

    args = parser.parse_args()

    args_config = build_args_config(args)

    if args.command == "preprocess":
        preprocess(args, args_config)
    if args.command == "train":
        train(args, args_config)
    if args.command == "infer":
        infer(args, args_config)


if __name__ == "__main__":
    main()
