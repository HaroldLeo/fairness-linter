from setuptools import setup

setup(name='fairness_linter',
      version='1.0',
      description='Fairness Linter for ML',
      url='',
      author='Hongji Liu',
      author_email='liuhj@uchicago.edu',
      license='MIT',
      packages=['fairness_linter'],
      install_requires=[
          'pandas',
          'statistics',
          'sklearn',
          'matplotlib',
          'numpy'
      ],
      zip_safe=False)