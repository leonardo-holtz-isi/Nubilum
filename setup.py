from setuptools import setup, find_packages

setup(
    name='Nubilum',
    version='0.1',
    description='An explainability library for instance segmentation in \
        point clouds',
    author='Leonardo Holtz de Oliveira',
    author_email='leo.holtz98@gmail.com',
    url='https://github.com/LeonardoHoltz/Nubilum',
    packages=find_packages(include=['nubilum', 'nubilum.*']),
    install_requires=[
        'torch',
        'numpy',
        'pandas',
        'plotly',
        'k3d',
        'captum'
    ]
)
