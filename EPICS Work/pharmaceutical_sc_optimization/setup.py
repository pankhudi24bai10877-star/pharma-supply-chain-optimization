from setuptools import setup, find_packages

setup(
    name="pharmaceutical_sc_optimization",
    version="0.1.0",
    description="Hybrid CVaR-Augmented SP and DRL for Pharmaceutical Supply Chain",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "gymnasium>=0.29.0",
        "stable-baselines3>=2.1.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
    ],
)
