ehmc_dir = py.get_install_dir() / 'experimentalhmc'

shared_library('main',
               'main.cpp',
               include_directories: incdir,
               install: true,
               install_dir: ehmc_dir,
               name_suffix: 'so')