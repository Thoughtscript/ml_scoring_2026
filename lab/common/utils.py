"""Shared helpers: paths, logging, data loading, and splitting."""
import functools
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
SEED = 42

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("lab")

np.random.seed(SEED)


def load_data(name):
    return pd.read_csv(DATA_DIR / f"{name}.csv")


def split_xy(df, target, stratify=False):
    """One deterministic 75/25 split, shared by training (techniques/) and scoring (tests/)."""
    X, y = df.drop(columns=[target]).to_numpy(), df[target].to_numpy()
    return train_test_split(X, y, test_size=0.25, random_state=SEED,
                            stratify=y if stratify else None)


TOTAL_STAGES = 4


def stage(number, title):
    """Decorator that logs a banner and elapsed time around a stage's main(), so each
    container's output is easy to follow (e.g. [STAGE 2/4])."""
    def wrap(fn):
        @functools.wraps(fn)
        def run(*args, **kwargs):
            tag = f"STAGE {number}/{TOTAL_STAGES}"
            log.info("=" * 70)
            log.info("[%s] START: %s", tag, title)
            log.info("=" * 70)
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
            except Exception:
                log.exception("[%s] FAILED: %s", tag, title)
                raise
            log.info("[%s] DONE: %s (%.1fs)", tag, title, time.perf_counter() - start)
            return result
        return run
    return wrap
