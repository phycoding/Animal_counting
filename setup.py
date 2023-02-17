from setuptools import setup, find_packages

setup(
    name='animalcounter',
    version='0.1.0',
    description='A package for counting animals in images and videos',
    author='Puspak Meher',
    author_email='puspakmeher3@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'detectron2',
        'opencv-python',
        'joblib'
    ],
    include_package_data=True
)
