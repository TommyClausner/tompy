"""
TomPy
=====

Provides a bunch of useful functions for a variety of tasks.

Example:
--------
A function that returns indices that would sort array a like array b:
:py:func:`tp.argsort_like <tompy.core.argsort_like>`:

>>> import numpy as np
>>> import tompy as tp
>>> a = np.random.randint(0, 100, 42)
>>> b = a[np.random.permutation(len(a))]
>>> tp.argsort_like(a, b)
"""

from .core import *
from .stats import *
from .plotting import *
