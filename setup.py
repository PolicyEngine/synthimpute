from setuptools import setup

setup(
    name="synthimpute",
    version="0.1",
    description="Python package for data synthesis and imputation using parametric and nonparametric methods, and evaluation of these methods.",
    url="http://github.com/PSLmodels/synthimpute",
    author="Max Ghenis",
    author_email="mghenis@gmail.com",
    license="MIT",
    packages=["synthimpute"],
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "statsmodels",
        "scipy",
    ],
    zip_safe=False,
)
