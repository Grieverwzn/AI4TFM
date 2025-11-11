"""
Setup script for AI4TFM package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai4tfm",
    version="0.1.0",
    author="AI4TFM Contributors",
    description="Advanced Traffic Flow Models for MPO Planning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Grieverwzn/AI4TFM",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    keywords="traffic, congestion, MPO, planning, transportation, analysis",
    project_urls={
        "Bug Reports": "https://github.com/Grieverwzn/AI4TFM/issues",
        "Source": "https://github.com/Grieverwzn/AI4TFM",
    },
)
