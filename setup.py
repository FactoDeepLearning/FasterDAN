from setuptools import setup, find_namespace_packages

setup(name='FasterDAN',
      packages=find_namespace_packages(include=["faster_dan", "faster_dan.*"]),
      version='1.0.0',
      install_requires=[
            "torch==1.12.1",
            "torchvision==0.13.1",
            "tensorboard",
            "scikit-learn",
            "opencv-python",
            "tqdm",
            "pillow",
            "networkx",
            "editdistance",
            "pyunpack",
            "fonttools"
            ]
      )