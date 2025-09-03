#!/usr/bin/env python3
"""
Setup script for FinRobot - AI Agent Platform for Financial Analysis
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README_FINROBOT.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "FinRobot - AI Agent Platform for Financial Analysis"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="finrobot",
    version="0.1.5",
    author="Trinh Khuong",
    author_email="trinhkhuong@example.com",
    description="AI Agent Platform for Financial Analysis using Large Language Models",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/TrinhKhuong1997DHKienTruc/Vietnam-LLM",
    project_urls={
        "Bug Reports": "https://github.com/TrinhKhuong1997DHKienTruc/Vietnam-LLM/issues",
        "Source": "https://github.com/TrinhKhuong1997DHKienTruc/Vietnam-LLM",
        "Documentation": "https://github.com/TrinhKhuong1997DHKienTruc/Vietnam-LLM/blob/master/FinRobot/README_FINROBOT.md",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Office/Business :: Financial",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "notebook": [
            "jupyter>=1.0",
            "ipywidgets>=7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "finrobot-demo=finrobot.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "finrobot": ["*.json", "*.yaml", "*.yml"],
    },
    keywords=[
        "finance",
        "ai",
        "agent",
        "llm",
        "trading",
        "investment",
        "analysis",
        "forecasting",
        "machine-learning",
        "artificial-intelligence",
    ],
)
