
from setuptools import setup, find_packages
from setuptools.command.install import install as _install
import os
import sys
import ctypes
import platform

cmdclass = dict()

# Force platform specific wheel.
# https://stackoverflow.com/a/45150383/1255535
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):

        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False

        # https://github.com/Yelp/dumb-init/blob/57f7eebef694d780c1013acd410f2f0d3c79f6c6/setup.py#L25
        def get_tag(self):
            python, abi, plat = _bdist_wheel.get_tag(self)
            if plat.startswith("linux_"):
                libc = ctypes.CDLL('libc.so.6')
                libc.gnu_get_libc_version.restype = ctypes.c_char_p
                GLIBC_VER = libc.gnu_get_libc_version().decode('utf8').split('.')
                plat = f'manylinux_{GLIBC_VER[0]}_{GLIBC_VER[1]}_{platform.machine()}'
            return python, abi, plat

    cmdclass['bdist_wheel'] = bdist_wheel

except ImportError:
    print(
        'Warning: cannot import "wheel" package to build platform-specific wheel'
    )
    print('Install the "wheel" package to fix this warning')


# Force use of "platlib" dir for auditwheel to recognize this is a non-pure
# build
# http://lxr.yanyahua.com/source/llvmlite/setup.py
class install(_install):

    def finalize_options(self):
        _install.finalize_options(self)
        self.install_libbase = self.install_platlib
        self.install_lib = self.install_platlib


cmdclass['install'] = install

# Read requirements.
with open('requirements.txt', 'r') as f:
    install_requires = [line.strip() for line in f.readlines() if line]

setup_args = dict(
    name="@PYPI_PACKAGE_NAME@",
    version="@PROJECT_VERSION@",
    python_requires='>=3.6',
    include_package_data=True,
    install_requires=install_requires,
    packages=find_packages(),
    cmdclass=cmdclass,
    zip_safe=False,
    description='python package for pydock',
)


setup(**setup_args)
