from distutils.core import setup
import numpy as np
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("stats.pyx"),
    include_dirs=[np.get_include()]
)