# from distutils.core import setup
try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements

install_reqs = parse_requirements('requirements.txt', session='hack')
reqs = [str(ir.req) for ir in install_reqs]


from setuptools import setup

# Package meta-Data.
NAME = 'mcdoi'
DESCRIPTION = 'mcdoi package for Python 3'
URL = 'http://github.com/kajdanowicz/mc-doi'
EMAIL = 'falkiewicz.maciej@gmail.com'
AUTHOR = 'Maciej Falkiewicz'
REQUIRES_PYTHON = '>=3.5.0'
VERSION = '0.0.1'


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name=NAME,
    version=VERSION,
    description = DESCRIPTION,
    long_description=readme,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    license=license,
    py_modules=['mc-doi'],
    include_package_data=True,
    install_requires=reqs
)

