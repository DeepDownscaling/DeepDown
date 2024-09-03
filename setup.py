from setuptools import setup, Command
from setuptools.command.install import install as _install
from setuptools.command.develop import develop as _develop
import subprocess
import eigenpip
import os

with open('README.md') as f:
    readme = f.read()


class InstallSBCKWithEigen(Command):
    description = 'Install SBCK package with custom Eigen path'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        print('Installing SBCK with custom Eigen path')

        # Set the environment variable for Eigen path
        os.environ['EIGEN_PATH'] = eigenpip.get_include()

        # Install SBCK using the environment variable
        subprocess.check_call(['pip', 'install', 'git+https://github.com/pascalhorton/SBCK-python.git'])


class CustomInstall(_install):
    def run(self):
        _install.run(self)
        self.run_command('install_sbck_with_eigen')


class CustomDevelop(_develop):
    def run(self):
        _develop.run(self)
        self.run_command('install_sbck_with_eigen')


setup(
    name="DeepDown",
    version="0.0.1",
    description="Deep downscaling of climate variables in Switzerland",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=['deepdown',
              'deepdown.models',
              'deepdown.utils'],
    package_dir={'deepdown': 'deepdown',
                 'deepdown.models': 'deepdown/models',
                 'deepdown.utils': 'deepdown/utils',
                 },
    zip_safe=False,
    extras_require={"test": ["pytest>=6.0"]},
    python_requires=">=3.8",
    install_requires=[
        'xarray',
        'pyproj',
        'numpy',
        'dask',
        'torch',
        'matplotlib',
        'scipy',
        'omegaconf',
        'rich',
        'eigenpip'
    ],
    cmdclass={
        'install_sbck_with_eigen': InstallSBCKWithEigen,
        'install': CustomInstall,
        'develop': CustomDevelop,
    },
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    project_urls={
        "Source Code": "https://github.com/DeepDownscaling/DeepDown",
        "Bug Tracker": "https://github.com/DeepDownscaling/DeepDown/issues",
    },
    license="MIT",
)
