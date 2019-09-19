#!/usr/bin/env python
"""
Usage:
    download_dataset.py DESTINATION_DIR

Options:
    -h --help   Show this screen.
"""

import os
from subprocess import call

from docopt import docopt


if __name__ == '__main__':
    args = docopt(__doc__)

    destination_dir = os.path.abspath(args['DESTINATION_DIR'])
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    os.chdir(destination_dir)

    for language in ('python', 'javascript', 'java', 'ruby', 'php', 'go'):
        call(['wget', 'https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{}.zip'.format(language), '-P', destination_dir, '-O', '{}.zip'.format(language)])
        call(['unzip', '{}.zip'.format(language)])
        call(['rm', '{}.zip'.format(language)])
