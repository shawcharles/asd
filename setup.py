# Copyright 2025 Charles Shaw
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Install script for Adaptive Supergeo Design (ASD) package."""

import os
from setuptools import setup, find_packages
import sys

__version__ = '1.0.0'

PROJECT_NAME = 'asd'

REQUIRED_PACKAGES = [
    'numpy>=1.8.0rc1', 'pandas>=1.1.5', 'matplotlib', 'scipy',
    'seaborn', 'networkx>=2.5', 'jinja2'
]

setup(
    name=PROJECT_NAME,
    version=__version__,
    description='Adaptive Supergeo Design (ASD) for geographic experiment design',
    author='Charles Shaw',
    author_email='charles@fixedpoint.io',
    # Contained modules and scripts.
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=REQUIRED_PACKAGES,
    # PyPI package information.
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Programming Language :: Python :: 3.7',
    ],
    license='Apache 2.0',
    keywords='trimmed match estimator supergeo adaptive supergeo geographic experiment design',
)
