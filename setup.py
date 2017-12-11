#!/usr/bin/env python

import os
import sys
from setuptools.command.test import test as TestCommand
from setuptools import find_packages

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

requires = [] #during runtime
tests_require=['pytest>=2.3'] #for testing

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))

setup(
    name='deepsource9',
    version='0.1.0',
    description='Point Source Detection by Deep Learinng',
    author='Alireza',
    url='https://github.com/vafaeiar/deepsource9',
    packages=find_packages(PACKAGE_PATH, "deepsource9"),
    package_dir={'deepsource9': 'deepsource9'},
    include_package_data=True,
    install_requires=requires,
    license='GPLv3',
    zip_safe=False,
    keywords='deepsource9',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        "Intended Audience :: Science/Research",
        'Intended Audience :: Developers',
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
    ]
)
