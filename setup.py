from setuptools import setup

setup(
    name="synthimpute",
    version="0.3.0",
    description="Python package for data synthesis and imputation using parametric and nonparametric methods, and evaluation of these methods.",
    url="http://github.com/PolicyEngine/synthimpute",
    author="Max Ghenis",
    author_email="max@policyengine.org",
    license="MIT",
    packages=["synthimpute"],
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "statsmodels",
        "scipy",
        "tqdm",
    ],
    extras_require={
        "dev": {
            "black",
            "flake8",
            "pytest",
            "wheel",
            "coverage",
        }
    },
    zip_safe=False,
)
