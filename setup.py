from setuptools import find_packages, setup

setup(
    name="denoizer",
    version='v0.0.1',
    author="GUrumov",
    description="A network to remove noise from images",
    packages=find_packages(exclude=["test_denoizer"]),
    install_requires=[
        "numpy>=1.17.4",
        "tensorflow==2.6.4",
        "PyYAML==5.3.1",
        "Pillow==7.1.1",
        "setuptools==40.8.0",
        "tqdm>=4.41.1"
    ],
    extras_require={  # allow to pip install object_region_proposal_3d[dev] to get the full environment including test stuff
        "dev": [
            "pytest==5.3.1",
            "pytest-cov==2.8.1",
        ]
    },
    python_requires=">=3.7",
)
