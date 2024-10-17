#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re

try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup

setup(
    name="kith",
    version='0.1',
    author="Ben Johnson",
    author_email="benjamin.johnson@cfa.harvard.edu",
    packages=["kith"],
    url="",
    #license="LICENSE",
    description="Heirarchical CMD modeling",
    long_description=open("README.md").read() + "\n\n",
    #install_requires=["numpy", "scipy >= 0.9", "astropy", "matplotlib", "scikit-learn"],
)
