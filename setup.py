from setuptools import setup, find_packages

setup(
    name="RAMPAGE",          # The name of your package
    version="1.0",                     # The version of your package
    packages=find_packages(),          # Automatically finds sub-packages
    install_requires=[                 # Dependencies required for your package
    ],
    author="Tomas Pelayo Benedet",                # The author of the package
    author_email="tpelayo@unizar.es", # Author's email
    description="A short description of your package",
    long_description=open('README.md').read(),  # A longer description (usually from README.md)
    long_description_content_type="text/markdown", # Content type of the long description
    url="https://github.com/reverseame/RAMPAGE", # URL to the project repository
    classifiers=[                      # Categories and metadata for PyPI
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',           # Minimum Python version required
)
