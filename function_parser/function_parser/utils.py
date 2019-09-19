import itertools
import os
import re
import subprocess
import tempfile
from typing import List, Tuple

import requests


def flatten(l):
    """Flatten list of lists.
    Args:
        l: A list of lists
    Returns: A flattened iterable
    """
    return itertools.chain.from_iterable(l)


def chunks(l: List, n: int):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def remap_nwo(nwo: str) -> Tuple[str, str]:
    r = requests.get('https://github.com/{}'.format(nwo))
    if r.status_code not in (404, 451, 502): # DMCA
        if 'migrated' not in r.text:
            if r.history:
                return (nwo, '/'.join(re.findall(r'"https://github.com/.+"', r.history[0].text)[0].strip('"').split('/')[-2:]))
            return (nwo, nwo)
    return (nwo, None)


def get_sha(tmp_dir: tempfile.TemporaryDirectory, nwo: str):
    os.chdir(os.path.join(tmp_dir.name, nwo))
    # git rev-parse HEAD
    cmd = ['git', 'rev-parse', 'HEAD']
    sha = subprocess.check_output(cmd).strip().decode('utf-8')
    os.chdir('/tmp')
    return sha


def download(nwo: str):
    os.environ['GIT_TERMINAL_PROMPT'] = '0'
    tmp_dir = tempfile.TemporaryDirectory()
    cmd = ['git', 'clone', '--depth=1', 'https://github.com/{}.git'.format(nwo), '{}/{}'.format(tmp_dir.name, nwo)]
    subprocess.run(cmd, stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    return tmp_dir


def walk(tmp_dir: tempfile.TemporaryDirectory, ext: str):
    results = []
    for root, _, files in os.walk(tmp_dir.name):
        for f in files:
            if f.endswith('.' + ext):
                results.append(os.path.join(root, f))
    return results
