# setup.py
from setuptools import setup, find_packages
import os

# Read the version from _version.py
with open(os.path.join('rfsynth_ad', '_metadata.py')) as f:
    exec(f.read()) # This sets __version__

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='rfsynth-ad',
    version=__version__,
    author=__author__,
    author_email=__email__,
    description="Anomaly detection using LGBM trained on real vs. uniform synthetic data",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/danielgaskins/rfsynth-ad',
    packages=find_packages(),
    install_requires=[
        'pandas>=1.0',
        'numpy>=1.18',
        'lightgbm>=3.0',
        'scikit-learn>=0.23',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    python_requires='>=3.7',
)