"""
Common utilities for qtzxchat.
"""

import os
import torch
import torch.distributed as dist


def print0(s="", **kwargs):
    """Print only on rank 0 (or if not using DDP)."""
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)


def get_dist_info():
    """
    Get distributed training info from environment variables.
    Returns: (is_ddp, rank, local_rank, world_size)
    """
    if all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE']):
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1


def autodetect_device():
    """Detect best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
