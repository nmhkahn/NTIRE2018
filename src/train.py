import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import json
import argparse
import importlib
from solver import Solver

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--ckpt_name", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument('--data_names', nargs='+', type=str, required=True)
    parser.add_argument('--scales', nargs='+', type=int, required=True)

    parser.add_argument("--ckpt_dir", type=str,
                        default="checkpoint/")
    parser.add_argument("--sample_dir", type=str,
                        default="sample/")
    parser.add_argument("--print_every", type=int, default=10000)
    
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--shave", type=int, default=20)

    parser.add_argument("--verbose", action="store_true", default="store_true")

    parser.add_argument("--loss", type=str, default="l1")
    parser.add_argument("--patch_size", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--decay", type=int, default=150000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--clip", type=float, default=10.0)

    parser.add_argument("--test_dirname", type=str)
    parser.add_argument("--test_data_from", type=str)
    parser.add_argument("--test_data_to", type=str)

    return parser.parse_args()

def main(cfg):
    # dynamic import using --model argument
    net = importlib.import_module("model.{}".format(cfg.model)).Net
    print(json.dumps(vars(cfg), indent=4, sort_keys=True))
    
    solver = Solver(net, cfg)
    solver.fit()

if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
