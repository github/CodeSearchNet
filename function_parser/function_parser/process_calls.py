"""
Usage:
    process_calls.py [options] INPUT_DIR DEFINITION_FILE OUTPUT_DIR

Options:
    -h --help
    --language LANGUAGE             Language
    --processes PROCESSES           # of processes to use [default: 16]
    --tree-sitter-build FILE        [default: /src/build/py-tree-sitter-languages.so]
"""
from collections import Counter, defaultdict
import functools
import gzip
from multiprocessing import Pool
import pandas as pd
import pickle

from docopt import docopt
from tree_sitter import Language

from language_data import LANGUAGE_METADATA
from process import DataProcessor


if __name__ == '__main__':
    args = docopt(__doc__)

    repository_dependencies = pd.read_csv(args['INPUT_DIR'] + 'repository_dependencies-1.4.0-2018-12-22.csv', index_col=False)
    projects = pd.read_csv(args['INPUT_DIR'] + 'projects_with_repository_fields-1.4.0-2018-12-22.csv', index_col=False)

    repository_dependencies['Manifest Platform'] = repository_dependencies['Manifest Platform'].apply(lambda x: x.lower())
    id_to_nwo = {project['ID']: project['Repository Name with Owner'] for project in projects[['ID', 'Repository Name with Owner']].dropna().to_dict(orient='records')}
    nwo_to_name = {project['Repository Name with Owner']: project['Name'] for project in projects[['Repository Name with Owner', 'Name']].dropna().to_dict(orient='records')}

    filtered = repository_dependencies[(repository_dependencies['Host Type'] == 'GitHub') & (repository_dependencies['Manifest Platform'] == LANGUAGE_METADATA[args['--language']]['platform'])][['Repository Name with Owner', 'Dependency Project ID']].dropna().to_dict(orient='records')

    dependency_pairs = [(rd['Repository Name with Owner'], id_to_nwo[int(rd['Dependency Project ID'])])
                        for rd in filtered if int(rd['Dependency Project ID']) in id_to_nwo]

    dependency_pairs = list(set(dependency_pairs))

    dents, dees = zip(*dependency_pairs)
    dents = list(set(dents))

    definitions = defaultdict(list)
    with open(args['DEFINITION_FILE'], 'rb') as f:
        for d in pickle.load(f)
            definitions[d['nwo']].append(d)
    definitions = dict(definitions)

    # Fill candidates from most depended libraries
    c = Counter(dees)
    library_candidates = {}
    for nwo, _ in c.most_common(len(c)):
        if nwo.split('/')[-1] not in library_candidates and nwo in definitions:
            # Approximate library name with the repository name from nwo
            library_candidates[nwo.split('/')[-1]] = definitions[nwo]

    DataProcessor.PARSER.set_language(Language(args['--tree-sitter-build'], args['--language']))
    processor = DataProcessor(language=args['--language'],
                              language_parser=LANGUAGE_METADATA[args['--language']]['language_parser'])

    with Pool(processes=int(args['--processes'])) as pool:
        output = pool.imap_unordered(functools.partial(processor.process_dent,
                                                       ext=LANGUAGE_METADATA[args['--language']]['ext']),
                                     dents)

    dent_definitions, edges = map(list, map(flatten, zip(*output)))

    with gzip.GzipFile(args['OUTPUT_DIR'] + '{}_dent_definitions.pkl.gz'.format(args['--language']), 'wb') as outfile:
        pickle.dump(dent_definitions, outfile)
    with gzip.GzipFile(args['OUTPUT_DIR'] + '{}_edges.pkl.gz'.format(args['--language']), 'wb') as outfile:
        pickle.dump(edges, outfile)
