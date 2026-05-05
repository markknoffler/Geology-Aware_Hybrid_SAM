from setuptools import find_packages, setup

setup(
    name="terrascope",
    version="0.1.0",
    description="Terrascope: dual-stream SAM-inspired landslide segmentation (RGB + DEM)",
    packages=find_packages(include=["terrascope*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "h5py",
        "Pillow",
        "tqdm",
        "scikit-learn",
    ],
)
