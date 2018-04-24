from distutils.core import setup
try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements

install_reqs = parse_requirements('requirements.txt', session='hack')
reqs = [str(ir.req) for ir in install_reqs]

setup(
    name='mcdoi',
    version='0.0.1',
    packages=[''],
    url='http://github.com/kajdanowicz/mc-doi',
    license='Apache 2.0',
    author='Maciej Falkiewicz',
    author_email='falkiewicz.maciej@gmail.com',
    description='',
#    install_requires=reqs
)

