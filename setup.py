from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='motivBCS2015',
      version='2',
      description='Code used to generate Figure.3 of our article',
      url='http://github.com/rcaze/motivBCS2015',
      author='Romain Caze',
      author_email='r.caze@iimperial.ac.uk',
      license='GNU',
      packages=['motivBCS2015'],
      install_requires=['numpy', 'matplotlib', 'scipy'],
      zip_safe=False)
