"""Setup script for opfsdr"""

from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='opfsdr',
    version='0.2.0',
    description='Semidefinite Relaxation of AC Optimal Power Flow',
    long_description=long_description,
    url='https://github.com/martinandersen/opfsdr',
    author='Martin S. Andersen',
    author_email='martin.skovgaard.andersen@gmail.com',
    license='GPL-3',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='optimization power flow',
    py_modules=['opfsdr'],
    zip_safe = True,
    install_requires=['cvxopt>=1.1.9','chompack>=2.3.2','numpy','mosek','requests'],
)
