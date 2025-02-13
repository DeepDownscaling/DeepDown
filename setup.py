from setuptools import setup, find_packages, Command
import subprocess
import eigenpip
import os


class InstallSBCKWithEigen(Command):
    description = 'Install SBCK package with custom Eigen path'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        print('Installing SBCK with custom Eigen path')
        os.environ['EIGEN_PATH'] = eigenpip.get_include()
        subprocess.check_call(['pip', 'install', 'git+https://github.com/pascalhorton/SBCK-python.git'])


setup(
    packages=find_packages(include=["deepdown", "deepdown.*"]),
    cmdclass={"install_sbck_with_eigen": InstallSBCKWithEigen},
)
