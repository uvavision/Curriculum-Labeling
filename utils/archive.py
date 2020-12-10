"""
Some of this code was adapted from https://github.com/kakaobrain/fast-autoaugment to apply random augmentations only
"""
from collections import defaultdict

from .augmentations import get_augment
import numpy as np
import random

PARAMETER_MAX = 10

def autoaug2arsaug(f):
    """
    Maps valid values for each PIL operation
    """
    def autoaug():
        mapper = defaultdict(lambda: lambda x: x)
        mapper.update({
            'ShearX': lambda x: float_parameter(x, 0.3),
            'ShearY': lambda x: float_parameter(x, 0.3),
            'TranslateX': lambda x: int_parameter(x, 10),
            'TranslateY': lambda x: int_parameter(x, 10),
            'Rotate': lambda x: int_parameter(x, 30),
            'Solarize': lambda x: 256 - int_parameter(x, 256),
            'Posterize2': lambda x: 4 - int_parameter(x, 4),
            'Contrast': lambda x: float_parameter(x, 1.8) + .1,
            'Color': lambda x: float_parameter(x, 1.8) + .1,
            'Brightness': lambda x: float_parameter(x, 1.8) + .1,
            'Sharpness': lambda x: float_parameter(x, 1.8) + .1,
            'CutoutAbs': lambda x: int_parameter(x, 20)
        })

        def low_high(name, prev_value):
            _, low, high = get_augment(name)
            return float(prev_value - low) / (high - low)

        policies = f()
        new_policies = []
        for policy in policies:
            new_policies.append([(name, pr, low_high(name, mapper[name](level))) for name, pr, level in policy])
        return new_policies

    return autoaug

def float_parameter(level, maxval):
    return float(level) * maxval / PARAMETER_MAX


def int_parameter(level, maxval):
    return int(float_parameter(level, maxval))

@autoaug2arsaug
def random_augment():
    """
    Pairs of random augmentations with random values to apply as heavy augmentation policies
    E.g., [('Invert', np.random.uniform(0,1), np.random.randint(10)), ('Contrast', np.random.uniform(0,1), np.random.randint(10))],
        [('Rotate', np.random.uniform(0,1), np.random.randint(10)), ('TranslateXAbs', np.random.uniform(0,1), np.random.randint(10))],
        [('Sharpness', np.random.uniform(0,1), np.random.randint(10)), ('Sharpness', np.random.uniform(0,1), np.random.randint(10))],
        ...
    """
    all_augmentations = ['Invert', 'Contrast', 'Rotate', 'TranslateXAbs', 'TranslateYAbs', 'Sharpness', 'ShearY', 'AutoContrast', 'Equalize', 'Posterize2', 'Color', 'Brightness', 'Solarize']
    return [
        [(random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10)), (random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10))],
        [(random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10)), (random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10))],
        [(random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10)), (random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10))],
        [(random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10)), (random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10))],
        [(random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10)), (random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10))],
        [(random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10)), (random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10))],
        [(random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10)), (random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10))],
        [(random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10)), (random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10))],
        [(random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10)), (random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10))],
        [(random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10)), (random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10))],
        [(random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10)), (random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10))],
        [(random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10)), (random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10))],
        [(random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10)), (random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10))],
        [(random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10)), (random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10))],
        [(random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10)), (random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10))],
        [(random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10)), (random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10))],
        [(random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10)), (random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10))],
        [(random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10)), (random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10))],
        [(random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10)), (random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10))],
        [(random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10)), (random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10))],
        [(random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10)), (random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10))],
        [(random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10)), (random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10))],
        [(random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10)), (random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10))],
        [(random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10)), (random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10))],
        [(random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10)), (random.choice(all_augmentations), np.random.uniform(0,1), np.random.randint(10))],
    ]