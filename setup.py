import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hetgen",
    version="0.0.0",
    author="Mahdi Qezlou",
    author_email="mahdi.qezlou@email.ucr.edu",
    description="Emission Line Generator for HETDEX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qezlou/private-HETDEX-cosmo",
    project_urls={
        "Bug Tracker": "https://github.com/qezlou/private-HETDEX-cosmo",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(where="."),
    package_dir={"": "."},  # Explicitly tell setuptools that packages are inside 'src/'
    python_requires=">=3.8",
    install_requires=[
    "numpy",
    "h5py",
    ],
)
