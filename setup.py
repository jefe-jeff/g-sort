# from setuptools import find_packages, setup

# setup(
#     name='src',
#     packages=find_packages(),
# )


from setuptools import find_packages#, setup
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize



ext_modules = [
    Extension("src.utilities.cython_extensions.bin2py_cythonext", 
                ["src/utilities/cython_extensions/bin2py_cythonext.pyx",]),
    Extension("src.utilities.cython_extensions.visionfile_cext",
                    ["src/utilities/cython_extensions/visionfile_cext.pyx", ]),
    Extension("src.utilities.cython_extensions.visionwrite_cext",
                ["src/utilities/cython_extensions/visionwrite_cext.pyx",]),
]


setup(
    name='src',
    packages=find_packages(),
    ext_modules=cythonize(ext_modules),
#     cmd_class={'build_ext' : build_ext}
)



# distutils.core.setup(

#         )
