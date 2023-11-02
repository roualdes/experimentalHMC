project('experimentalHMC', 'cpp')

incdir = include_directories('include')
py = import('python').find_installation(pure: false)

subdir('experimentalHMC')

py.install_sources(
    'experimentalhmc/__init__.py',
    'experimentalhmc/convergence.py',
    'experimentalhmc/dualaverage.py',
    'experimentalhmc/leapfrog.py',
    'experimentalhmc/metric_adapter.py',
    'experimentalhmc/onlinemad.py',
    'experimentalhmc/onlinemean.py',
    'experimentalhmc/onlinemeanvar.py',
    'experimentalhmc/onlinequantile.py',
    'experimentalhmc/stan.py',
    'experimentalhmc/windowedaptation.py',
    'experimentalhmc/f.py',
    subdir: 'experimentalhmc',
)