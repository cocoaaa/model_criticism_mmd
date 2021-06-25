import subprocess
import pathlib
import os

def install_softDTW():
    if not pathlib.Path('./').absolute().joinpath('opt').exists():
        pathlib.Path('./').absolute().joinpath('opt').mkdir()
    # end if
    os.chdir('opt')
    cmd = ['git', 'clone', 'https://github.com/mblondel/soft-dtw.git']
    subprocess.run(cmd)
    os.chdir(pathlib.Path('./').joinpath('soft-dtw'))
    subprocess.run(['make', 'cython'])
    current_python = subprocess.run(['which', 'python'], stdout=subprocess.PIPE).stdout.decode("utf-8").strip()
    subprocess.run([current_python, 'setup.py', 'build'])
    subprocess.run([current_python, 'setup.py', 'install'])
