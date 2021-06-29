import argparse
import os

HOME = os.environ['HOME']


def config_USC():
    parser = argparse.ArgumentParser(prog="CMFD")

    parser.add_argument("--dataset", type=str, default="usc")

    parser.add_argument("--size", type=str, default="320x320",
                        help="image shape (h x w)")

    parser.add_argument("--model", type=str, default="dlab", help="model name")
    # network config
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", "-b", type=int, default=20)
    parser.add_argument("--max-epoch", type=int, default=50)
    parser.add_argument("--resume", type=int, default=1,
                        help="resume from epoch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--suffix", type=str, default="",
                        help="model name suffix")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="pretrained model path")
    parser.add_argument("--test", action='store_true', help="test only mode")
    parser.add_argument("--thres", type=float, default=0.5,
                        help="threshold for detection")
    # path config
    parser.add_argument("--lmdb-dir", type=str,
                        default=HOME+"/dataset/CMFD/USCISI-CMFD")
    parser.add_argument("--train-key", type=str, default="train.keys")
    parser.add_argument("--test-key", type=str, default="test.keys")
    parser.add_argument("--valid-key", type=str, default="valid.keys")
    parser.add_argument("--out-channel", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.001)
    parser.add_argument("--gamma2", type=float, default=0.01)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--bw", action="store_true", help="whether to add boundary loss"
    )

    args = parser.parse_args()
    print(args)
    return args


def config_COMO():
    parser = argparse.ArgumentParser(prog="CMFD")

    parser.add_argument("--dataset", type=str, default="como")

    parser.add_argument("--size", type=str, default="320x320",
                        help="image shape (h x w)")

    parser.add_argument("--model", type=str, default="dlab", help="model name")
    # network config
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", "-b", type=int, default=20)
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument("--resume", type=int, default=0,
                        help="resume from epoch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--suffix", type=str, default="",
                        help="model name suffix")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="pretrained model path")
    parser.add_argument("--test", action='store_true', help="test only mode")
    parser.add_argument("--thres", type=float, default=0.5,
                        help="threshold for detection")
    # path config
    parser.add_argument("--root", type=str,
                        default=HOME+"/dataset/CMFD/COMO")
    parser.add_argument("--data-path", type=str,
                        default=(HOME+"/dataset/CMFD/COMO"
                                 "/CoMoFoD-CMFD.hd5"))
    parser.add_argument("--split", type=int, default=0, help="data split")
    parser.add_argument("--out-channel", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()
    print(args)
    return args


def config_CASIA():
    parser = argparse.ArgumentParser(prog="CMFD")

    parser.add_argument("--dataset", type=str, default="casia")

    parser.add_argument("--size", type=str, default="320x320",
                        help="image shape (h x w)")

    parser.add_argument("--model", type=str, default="dlab", help="model name")
    # network config
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", "-b", type=int, default=15)
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument("--resume", type=int, default=0,
                        help="resume from epoch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--suffix", type=str, default="",
                        help="model name suffix")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="pretrained model path")
    parser.add_argument("--test", action='store_true', help="test only mode")
    parser.add_argument("--thres", type=float, default=0.5,
                        help="threshold for detection")
    # path config
    parser.add_argument("--root", type=str,
                        default=HOME+"/dataset/CMFD/CASIA")
    parser.add_argument("--data-path", type=str,
                        default=(HOME+"/dataset/CMFD/CASIA"
                                 "/CASIA-CMFD-Pos.hd5"))
    parser.add_argument("--split", type=int, default=0, help="data split")
    parser.add_argument("--out-channel", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()
    print(args)
    return args
