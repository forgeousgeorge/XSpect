from setuptools import setup, find_packages

from XSpect_EW_package import __version__

setup(
    name='XSpect_EW_package',
    version=__version__,

    url='https://github.com/forgeousgeorge/XSpect',
    author='George Vejar',
    author_email='forgeousgeorge@gmail.com',

    packages=find_packages(),
)