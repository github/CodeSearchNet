import glob
from itertools import chain
import os
import pickle
import re

from dask.distributed import Client
import dask.distributed
from tqdm import tqdm

from language_data import LANGUAGE_METADATA
from utils import download

# Gets notices
LEGAL_FILES_REGEX ='(AUTHORS|NOTICE|LEGAL)(?:\..*)?\Z'

PREFERRED_EXT_REGEX = '\.[md|markdown|txt|html]\Z'

# Regex to match any extension except .spdx or .header
OTHER_EXT_REGEX = '\.(?!spdx|header|gemspec)[^./]+\Z'

# Regex to match, LICENSE, LICENCE, unlicense, etc.
LICENSE_REGEX = '(un)?licen[sc]e'

# Regex to match COPYING, COPYRIGHT, etc.
COPYING_REGEX = 'copy(ing|right)'

# Regex to match OFL.
OFL_REGEX = 'ofl'

# BSD + PATENTS patent file
PATENTS_REGEX = 'patents'


def match_license_file(filename):
    for regex in [LEGAL_FILES_REGEX, 
                  LICENSE_REGEX + '\Z',
                  LICENSE_REGEX + PREFERRED_EXT_REGEX,
                  COPYING_REGEX + '\Z',
                  COPYING_REGEX + PREFERRED_EXT_REGEX,
                  LICENSE_REGEX + OTHER_EXT_REGEX,
                  COPYING_REGEX + OTHER_EXT_REGEX,
                  LICENSE_REGEX + '[-_]',
                  COPYING_REGEX + '[-_]',
                  '[-_]' + LICENSE_REGEX,
                  '[-_]' + COPYING_REGEX,
                  OFL_REGEX + PREFERRED_EXT_REGEX,
                  OFL_REGEX + OTHER_EXT_REGEX,
                  OFL_REGEX + '\Z',
                  PATENTS_REGEX + '\Z',
                  PATENTS_REGEX + OTHER_EXT_REGEX]:
        if re.match(regex, filename.lower()):
            return filename
    return None

def flattenlist(listoflists):
    return list(chain.from_iterable(listoflists))

def fetch_license(nwo):
    licenses = []
    tmp_dir = download(nwo)
    for f in sorted(glob.glob(tmp_dir.name + '/**/*', recursive=True), key=lambda x: len(x)):
        if not os.path.isdir(f):
            if match_license_file(f.split('/')[-1]):
                licenses.append((nwo, f.replace(tmp_dir.name + '/', ''), open(f, errors='surrogateescape').read()))
    
    return licenses


client = Client()

for language in LANGUAGE_METADATA.keys():
    definitions = pickle.load(open('../data/{}_dedupe_definitions_v2.pkl'.format(language), 'rb'))
    nwos = list(set([d['nwo'] for d in definitions]))
    
    futures = client.map(fetch_license, nwos)
    results = []
    for r in tqdm(futures):
        try:
            results.append(r.result(2))
        except dask.distributed.TimeoutError:
            continue
    
    flat_results = flattenlist(results)
    licenses = dict()
    for nwo, path, content in flat_results:
        if content:
            licenses[nwo] = licenses.get(nwo, []) + [(path, content)]
    pickle.dump(licenses, open('../data/{}_licenses.pkl'.format(language), 'wb'))
