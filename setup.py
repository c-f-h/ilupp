from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import setuptools

# setup.py adapted from the pybind11 python_example by Sylvain Corlay

def readme():
    with open(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'README.md')) as fp:
        return fp.read()

def get_version(fname):
    with open(fname) as f:
        for line in f:
            if line.startswith("__version__ = '"):
                return line.split("'")[1]
    raise RuntimeError('could not parse version string')

__version__ = get_version('ilupp/__init__.py')

# env variable INT64_INDICES determines whether to use large indices
INT64_INDICES = os.environ.get('INT64_INDICES')
INT64_INDICES = True if (INT64_INDICES and int(INT64_INDICES)) else False

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


ext_modules = [
    Extension(
        'ilupp._ilupp',
        ['src/binding.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++'
    ),
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.

    The newer version is prefered over c++11 (when it is available).
    """
    flags = ['-std=c++17', '-std=c++14', '-std=c++11']

    for flag in flags:
        if has_flag(compiler, flag): return flag

    raise RuntimeError('Unsupported compiler -- at least C++11 support is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
            if INT64_INDICES:
                opts.append('-DINT64_INDICES=1')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
            if INT64_INDICES:
                opts.append('/DINT64_INDICES=1')
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)

setup(
    name = 'ilupp',
    version = __version__,
    author = 'Clemens Hofreither',
    author_email = 'chofreither@gmail.com',
    url = 'https://github.com/c-f-h/ilupp',
    description = "Incomplete LU algorithms for C++ and Python",
    long_description = readme(),
    long_description_content_type = 'text/markdown',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
    ],
    ext_modules = ext_modules,
    packages = ['ilupp'],
    install_requires = [ 'numpy>=1.11', 'scipy' ],
    setup_requires = ['pybind11>=2.4'],
    cmdclass = {'build_ext': BuildExt},
    zip_safe = False,

    tests_require = ['nose'],
    test_suite = 'nose.collector'
)
