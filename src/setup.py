from os import system
from setuptools import setup, find_packages, Command

# visit: https://setuptools.readthedocs.io/en/latest/setuptools.html


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')


setup(
    name='duu',
    version='0.3.0',

    author="Kennedy Putra Kusumo, "
           "Lucian Gomoescu, "
           "Radoslav Paulen",
    author_email="kennedy.kusumo16@imperial.ac.uk, "
                 "gomoescu.lucian@gmail.com, "
                 "radoslav.paulen@stuba.sk",

    packages=find_packages(),

    cmdclass={'clean': CleanCommand}
)


