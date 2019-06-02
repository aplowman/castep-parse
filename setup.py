"""Pip installation script for `castep-parse`."""

import os
import re
from setuptools import find_packages, setup


def get_version():

    ver_file = '{}/_version.py'.format('castep_parse')
    with open(ver_file) as handle:
        ver_str_line = handle.read()

    ver_pattern = r'^__version__ = [\'"]([^\'"]*)[\'"]'
    match = re.search(ver_pattern, ver_str_line, re.M)
    if match:
        ver_str = match.group(1)
    else:
        msg = 'Unable to find version string in "{}"'.format(ver_file)
        raise RuntimeError(msg)

    return ver_str


setup(
    name='castep-parse',
    version=get_version(),
    description=('Input file writers and output file parsers for the density '
                 'functional theory code CASTEP.'),
    author='Adam J. Plowman',
    author_email='adam.plowman@manchester.ac.uk',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'mendeleev',
    ],
    project_urls={
        'Github': 'https://github.com/aplowman/castep-parse',
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Operating System :: OS Independent',
    ],
)
