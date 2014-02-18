__author__ = 'xinchou'

from setuptools import setup
import SNFtools

setup(
     name = "SNFtools",
     version= SNFtools.__version__,
     url="http://www.github.com/xinchoubiology/SNFtools",
     license="MIT License",
     author="xinchoubiology",
     author_email='xinchoubiology@gmail.com',
     description=('A similarity network fusion tool'),
     packages=['SNFtools'],
     install_requires=['numpy', 'matplotlib']
)
