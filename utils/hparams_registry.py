# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
from utils import misc

def _hparams(network, random_seed):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions.
    _hparam('lr', 0.001, lambda r: 10**r.uniform(-5, -2)) # 
    _hparam('weight_decay', 0, lambda r: 10**r.uniform(-6, -2)) 
    _hparam('batch_size', 16, lambda r: int(r.choice([8,12,16]))) 
    _hparam('epoch', 100, lambda r: int(r.choice([60,90,120,150]))) 
    _hparam('transform_aug', False, lambda r: bool(r.choice([True,False])))
    _hparam('lr_schedule', 1, lambda r: int(r.choice([0,1,2,3]))) 
    
    if network == 'PoseResNet':
        _hparam('num_layers', 50, lambda r: int(r.choice([50])))  #[18,34,50,101,152] 
        _hparam('pretrained', False, lambda r: bool(r.choice([False]))) #True,

    return hparams


def default_hparams(network):
    return {a: b for a, (b, c) in _hparams(network, 0).items()}


def random_hparams(network, seed):
    return {a: c for a, (b, c) in _hparams(network, seed).items()}
