project('experimentalHMC', 'cpp',
        default_options : ['cpp_std=c++14'])

incdir = include_directories('include')
py = import('python').find_installation(pure: false)

subdir('experimentalhmc')

py.install_sources(
    'experimentalhmc/__init__.py',
    'experimentalhmc/convergence.py',
    'experimentalhmc/dualaverage.py',
    'experimentalhmc/leapfrog.py',
    'experimentalhmc/metric_adapter.py',
    'experimentalhmc/normal.py',
    'experimentalhmc/onlinemad.py',
    'experimentalhmc/onlinemean.py',
    'experimentalhmc/onlinemeanvar.py',
    'experimentalhmc/onlinequantile.py',
    'experimentalhmc/rng.py',
    'experimentalhmc/stan.py',
    'experimentalhmc/windowedaptation.py',
    'experimentalhmc/ehmc_cpp.py',
    subdir: 'experimentalhmc',
)