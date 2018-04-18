from setuptools import setup, find_packages

setup(name='trainer',
      version='0.1',
      packages=find_packages(),
      description="Playing around with TensorFlow",
      author="Jon Deaotn",
      author_email="jonpdeaton@gmail.com",
      license='MIT',
      install_requires=[
          "tensorflow",
          "numpy",
          ""
      ],
      zip_safe=False)