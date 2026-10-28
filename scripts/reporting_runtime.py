#!/usr/bin/env python3
"""Fail before generating reports in an unvalidated numerical runtime."""
import json
import sys

import numpy as np


def require_runtime():
    if sys.version_info[:2] != (3, 12) or np.__version__ != '2.5.3':
        raise RuntimeError('Unvalidated reporting runtime: require Python 3.12 and NumPy 2.5.3; '
                           f'got Python {sys.version.split()[0]}, NumPy {np.__version__}. '
                           'Use the supplied pinned review environment; no output generated.')
    x = np.array([-3., 1.])
    square, absolute = np.empty_like(x), np.empty_like(x)
    np.multiply(x, x, out=square)
    np.absolute(x, out=absolute)
    if x.tolist() != [-3., 1.] or absolute.mean() != 2. or square.mean() != 5.:
        raise RuntimeError('Numerical reduction self-check failed')
    return {'python': sys.version.split()[0], 'numpy': np.__version__, 'reduction_self_check': True}


if __name__ == '__main__':
    print(json.dumps(require_runtime()))
