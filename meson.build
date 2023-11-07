project('experimentalHMC', 'cpp',
        default_options : ['cpp_std=c++17'])

incdir = include_directories('include')
py = import('python').find_installation(pure: false)

subdir('experimentalhmc')

py.install_sources(
    'experimentalhmc/__init__.py',
    'experimentalhmc/convergence.py',
    'experimentalhmc/dual_average.py',
    'experimentalhmc/ehmc_cpp.py',
    'experimentalhmc/initialize_draws.py',
    'experimentalhmc/leapfrog.py',
    'experimentalhmc/metric_adapter.py',
    'experimentalhmc/normal.py',
    'experimentalhmc/onlinemad.py',
    'experimentalhmc/onlinemeanvar.py',
    'experimentalhmc/onlinequantile.py',
    'experimentalhmc/rng.py',
    'experimentalhmc/stan.py',
    'experimentalhmc/step_size_adapter.py',
    'experimentalhmc/step_size_initializer.py',
    'experimentalhmc/windowedaptation.py',
    subdir: 'experimentalhmc',
)