#!/usr/bin/env python
import os
import sys
from setuptools import setup

# Prepare and send a new release to PyPI
if "release" in sys.argv[-1]:
    os.system("python setup.py sdist")
    os.system("python setup.py bdist_wheel")
    os.system("twine upload dist/*")
    os.system("rm -rf dist/hufflescuff*")
    sys.exit()

# Load the __version__ variable without importing the package already
exec(open('hufflescuff/version.py').read())

# DEPENDENCIES
# 1. What are the required dependencies?
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(name='hufflescuff',
      version=__version__,
      description="A widget tool to calculate the Hough Transform of some data, and let you explore the data with a bokeh widget.",
      long_description=open('README.md').read(),
      author='Christina Hedges',
      author_email='christina.l.hedges@nasa.gov',
      license='MIT',
      package_dir={
            'hufflescuff': 'hufflescuff'},
      packages=['hufflescuff'],
      install_requires=install_requires,
      include_package_data=True,
      classifiers=[
          "Development Status :: 5 - Production/Stable",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering :: Astronomy",
          ],
      )
