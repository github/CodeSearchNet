#!/usr/bin/env python
"""
Usage:
    embeddingvis.py [options] plot-tsne (--code | --query) MODEL_PATH
    embeddingvis.py [options] print-nns (--code | --query) MODEL_PATH DISTANCE_THRESHOLD

Options:
    --azure-info=<path>        Azure authentication information file (JSON). Used to load data from Azure storage.
    --distance-metric METRIC   The distance metric to use [default: cosine]
    --num-nns NUM              The number of nearest neighbors to show when print-nns. [default: 2]
    --lim-items NUM            Maximum number of items to use. Useful when memory is limited. [default: -1]
    -h --help                  Show this screen.
    --hypers-override HYPERS   JSON dictionary overriding hyperparameter values.
    --language LANG            The code language to use. Only when --code option is given. [default: python]
    --debug                    Enable debug routines. [default: False]
"""
from docopt import docopt
from dpu_utils.utils import RichPath, run_and_debug
from sklearn.manifold import TSNE
import numpy as np
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt


import model_restore_helper
from utils.visutils import square_to_condensed


def run(arguments) -> None:
    azure_info_path = arguments.get('--azure-info', None)

    model_path = RichPath.create(arguments['MODEL_PATH'], azure_info_path=azure_info_path)

    model = model_restore_helper.restore(
        path=model_path,
        is_train=False)

    if arguments['--query']:
        embeddings, elements = model.get_query_token_embeddings()
    else:
        embeddings, elements = model.get_code_token_embeddings(arguments['--language'])

    max_num_elements = int(arguments['--lim-items'])
    if max_num_elements > 0:
        embeddings, elements = embeddings[:max_num_elements], elements[:max_num_elements]

    print(f'Collected {len(elements)} elements to visualize.')

    embeddings = model.sess.run(fetches=embeddings)

    if arguments['plot-tsne']:
        emb_2d = TSNE(n_components=2, verbose=1, metric=arguments['--distance-metric']).fit_transform(embeddings)

        plt.scatter(emb_2d[:, 0], emb_2d[:, 1])
        for i in range(len(elements)):
            plt.annotate(elements[i], xy=(emb_2d[i,0], emb_2d[i,1]))

        plt.show()
    elif arguments['print-nns']:
        flat_distances = pdist(embeddings, arguments['--distance-metric'])
        num_nns = int(arguments['--num-nns'])

        for i, element in enumerate(elements):
            distance_from_i = np.fromiter(
                (flat_distances[square_to_condensed(i, j, len(elements))] if i != j else float('inf') for j in
                 range(len(elements))), dtype=np.float)

            nns = [int(k) for k in np.argsort(distance_from_i)[:num_nns]]  # The first two NNs

            if distance_from_i[nns[0]] > float(arguments['DISTANCE_THRESHOLD']):
                continue
            try:
                print(f'{element} --> ' + ', '.join(f'{elements[n]} ({distance_from_i[n]:.2f})' for n in nns))
            except:
                print('Error printing token for nearest neighbors pair.')




if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get('--debug', False))