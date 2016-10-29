from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import numpy
import sys
reload(sys)
sys.setdefaultencoding('utf8') 

extensions = [
  Extension('im2col_cython', ['im2col_cython.pyx'],
            include_dirs = [numpy.get_include()]
  ),
]

setup(
    ext_modules = cythonize(extensions),
)
