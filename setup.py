#!/usr/bin/env python3
"""
Setup script for FinBERT AAPL Price Prediction Demo
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="finbert-aapl-prediction",
    version="1.0.0",
    author="Trinh Khuong",
    author_email="trinhkhuong1997@gmail.com",
    description="AAPL Price Prediction Demo with FinBERT Sentiment Analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/TrinhKhuong1997DHKienTruc/Vietnam-LLM",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "finbert-aapl-demo=aapl_price_prediction_demo:main",
            "finbert-test=test_finbert:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
