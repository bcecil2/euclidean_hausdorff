from distutils.core import setup

setup(
    name='euclidean_hausdorff',
    version='1.3.2',
    author='Vladyslav Oles, Blake Cecil',
    author_email='vlad.oles@proton.me',
    packages=['euclidean_hausdorff'],
    url='http://pypi.python.org/pypi/euclidean-hausdorff/',
    description="quick approximation of the Gromov–Hausdorff distance restricted to Euclidean isometries",
    long_description=open('README.md').read(),
    install_requires=[
        "torch==2.9.1+cpu",
        "scipy >= 1.15.3",
        "sortedcontainers >= 2.4.0",
        "numpy >= 1.26.4",
        "pymanopt >= 2.2.1"
    ],
)