"""Setup configuration for the DDR Biomarker Pipeline package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="ddr-biomarker-pipeline",
    version="1.0.0",
    author="DDR Biomarker Pipeline Team",
    author_email="pipeline@example.com",
    description=(
        "ML pipeline for identifying genomic biomarkers predictive of "
        "DDR-targeting therapy sensitivity using GDSC2 and DepMap data"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/ddr-biomarker-pipeline",
    packages=find_packages(exclude=["tests*", "scripts*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.2.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "ddr-train=scripts.train:main",
            "ddr-evaluate=scripts.evaluate:main",
            "ddr-analyze=scripts.analyze_biomarkers:main",
        ],
    },
)
