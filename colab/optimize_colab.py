import argparse
import os
import random
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from arguments import ModelParams, OptimizationParams, PipelineParams
from script.optimize import training
from utils.general_utils import safe_state


def recursive_merge_into_args(args, config_node):
    for key in config_node.keys():
        value = config_node[key]
        if isinstance(value, DictConfig):
            recursive_merge_into_args(args, value)
        else:
            if not hasattr(args, key):
                raise AttributeError(f"Config key does not match an argparse field: {key}")
            setattr(args, key, value)


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Colab-safe Instant4D optimization wrapper")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument("--num_pts", type=int, default=100_000)
    parser.add_argument("--num_pts_ratio", type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--exhaust_test", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    cli_source_path = args.source_path
    cli_model_path = args.model_path

    cfg = OmegaConf.load(args.config)
    recursive_merge_into_args(args, cfg)

    if cli_source_path:
        args.source_path = cli_source_path
    if cli_model_path:
        args.model_path = cli_model_path

    args.source_path = os.path.abspath(args.source_path)
    args.model_path = os.path.abspath(args.model_path)
    os.makedirs(args.model_path, exist_ok=True)

    setup_seed(args.seed)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    lp_ = lp.extract(args)
    op_ = op.extract(args)
    pp_ = pp.extract(args)
    lp_.source_path = args.source_path
    lp_.model_path = args.model_path

    print(f"Optimizing source={lp_.source_path}")
    print(f"Writing model={lp_.model_path}")

    training(
        lp_,
        op_,
        pp_,
        args.test_iterations,
        args.save_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.gaussian_dim,
        args.time_duration,
        args.num_pts,
        args.num_pts_ratio,
        args.rot_4d,
        args.force_sh_3d,
        args.batch_size,
    )

    print("Training complete.")


if __name__ == "__main__":
    main()

