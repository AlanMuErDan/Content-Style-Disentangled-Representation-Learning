# utils/lr_scheduler.py

import math
import torch
from functools import partial

# step scheduler
def fn_LinearWarmup(warmup_steps, step):
    if step < warmup_steps:  # linear warmup
        return float(step) / float(max(1, warmup_steps))
    else:
        return 1.0

def Scheduler_LinearWarmup(warmup_steps):
    return partial(fn_LinearWarmup, warmup_steps)


def fn_LinearWarmup_CosineDecay(warmup_steps, max_steps, multipler_min, step):
    if max_steps <= warmup_steps: 
        return 1.0
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    else:
        progress = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
        progress = min(progress, 1.0)
        multipler = 0.5 * (math.cos(math.pi * progress) + 1)
        return max(multipler, multipler_min)

def Scheduler_LinearWarmup_CosineDecay(warmup_steps, max_steps, multipler_min):
    return partial(fn_LinearWarmup_CosineDecay, warmup_steps, max_steps, multipler_min)