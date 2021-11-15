import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch_models_imp",
    version="0.0.1",
    author="Evstifeev Stepan",
    author_email="minemile69@gmail.com",
    description="Several implementations of deep learning algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/minemile/pytorch-models",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # package_dir={"": "."},
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)