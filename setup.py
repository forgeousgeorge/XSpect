from setuptools import setup, find_packages

from XSpect_EW import __version__

setup(
    name='XSpect_EW',
    version=__version__,

    url='https://github.com/forgeousgeorge/XSpect',
    author='George Vejar',
    author_email='forgeousgeorge@gmail.com',

    packages=find_packages(),
)