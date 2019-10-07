from setuptools import setup, find_packages
import re, ast

_version_re = re.compile(r'__version__\s+=\s+(.*)')

with open('re_map/__init__.py', 'rb') as f:
	version = str(ast.literal_eval(_version_re.search(
		f.read().decode('utf-8')).group(1)))

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = []
with open('requirements.txt', 'r') as f:
  for line in f:
    if line.strip():
      install_requires.append(line.strip())

setup (
  name='jigsaw-toxic-comment-classification-challenge',
  author="Aleksas Pielikis",
  author_email="ant.kampo@gmail.com",
  version=version,
  long_description=long_description,
  url="https://github.com/aleksas/jigsaw-toxic-comment-classification-challenge",
  long_description_content_type="text/markdown",
  zip_safe=False,
  packages=find_packages(),
  install_requires=install_requires
)