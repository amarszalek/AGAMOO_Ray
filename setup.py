from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = []
with open("requirements.txt", "r") as reqs_file:
    for line in reqs_file:
        requirements.append(line.strip())

setup(
    name="agamoo",
    version="0.0.1",
    author="Adam Marszałek & Paweł Jarosz",
    author_email="amarszalek@pk.edu.pl",
    description="Asynchronous GAme theory based framework for MultiObjective Optimization in Python",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/amarszalek/AGAMOO",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
    ],
    include_package_data=True,
    package_data={'': ['_cutils.so']},
)
