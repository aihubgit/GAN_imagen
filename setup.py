from setuptools import setup, find_packages
exec(open('imagen_pytorch/version.py').read())

setup(
  name = 'GAN_imagen',
  packages = find_packages(exclude=[]),
  include_package_data = True,
  entry_points={
    'console_scripts': [
      'GAN_imagen = imagen_pytorch.cli:main',
      'imagen = imagen_pytorch.cli:imagen'
    ],
  },
  version = __version__,
  license='MIT',
  description = 'Imagen - unprecedented photorealism × deep level of language understanding',
  author = 'smartcoop',
  author_email = 'aihubgit@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/aihubgit/GAN_imagen',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'text-to-image',
    'denoising-diffusion'
  ],
  install_requires=[
    'accelerate',
    'click',
    'datasets',
    'einops>=0.6',
    'einops-exts',
    'ema-pytorch>=0.0.3',
    'fsspec',
    'kornia',
    'numpy',
    'packaging',
    'pillow',
    'pydantic',
    'pytorch-lightning',
    'pytorch-warmup',
    'sentencepiece',
    'torch>=1.6',
    'torchvision',
    'transformers',
    'tqdm'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.10',
  ],
)
