"""
Setup script for AAPL Prediction using Chronos-Bolt
"""

from setuptools import setup, find_packages

setup(
    name="aapl-chronos-prediction",
    version="1.0.0",
    description="AAPL Hourly Price Prediction using Chronos-Bolt Base Model",
    author="Trinh Khuong",
    author_email="trinhkhuong1997@gmail.com",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "yfinance>=0.2.0",
        "chronos-forecasting>=0.0.1",
        "scikit-learn>=1.3.0",
        "plotly>=5.15.0",
        "jupyter>=1.0.0",
        "ipykernel>=6.25.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
