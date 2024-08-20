from distutils.core import setup

setup(
    name='euclidean_hausdorff',
    version='1.0.3',
    author='Vladyslav Oles, Blake Cecil',
    author_email='vlad.oles@proton.me',
    packages=['euclidean_hausdorff'],
    url='http://pypi.python.org/pypi/euclidean-hausdorff/',
    description="quick approximation of the Gromov–Hausdorff distance using Euclidean isometries",
    long_description=open('README.md').read(),
    install_requires=[
        "scipy >= 1.12.0",
        "sortedcontainers >= 2.4.0",
        "tqdm >= 4.66.4",
        "numpy >= 1.26.4",
    ],
)