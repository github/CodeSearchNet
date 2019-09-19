from collections import Counter
import numpy as np
from typing import Dict, Any, List, Iterable, Optional, Tuple
import random
import re

from utils.bpevocabulary import BpeVocabulary
from utils.tfutils import convert_and_pad_token_sequence

import tensorflow as tf
from dpu_utils.codeutils import split_identifier_into_parts
from dpu_utils.mlutils import Vocabulary

from .encoder import Encoder, QueryType


class SeqEncoder(Encoder):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        encoder_hypers = { 'token_vocab_size': 10000,
                           'token_vocab_count_threshold': 10,
                           'token_embedding_size': 128,

                           'use_subtokens': False,
                           'mark_subtoken_end': False,

                           'max_num_tokens': 200,

                           'use_bpe': True,
                           'pct_bpe': 0.5
                         }
        hypers = super().get_default_hyperparameters()
        hypers.update(encoder_hypers)
        return hypers

    IDENTIFIER_TOKEN_REGEX = re.compile('[_a-zA-Z][_a-zA-Z0-9]*')

    def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
        super().__init__(label, hyperparameters, metadata)
        if hyperparameters['%s_use_bpe' % label]:
            assert not hyperparameters['%s_use_subtokens' % label], 'Subtokens cannot be used along with BPE.'
        elif hyperparameters['%s_use_subtokens' % label]:
            assert not hyperparameters['%s_use_bpe' % label], 'Subtokens cannot be used along with BPE.'

    def _make_placeholders(self):
        """
        Creates placeholders "tokens" for sequence encoders.
        """
        super()._make_placeholders()
        self.placeholders['tokens'] = \
            tf.placeholder(tf.int32,
                           shape=[None, self.get_hyper('max_num_tokens')],
                           name='tokens')

    def embedding_layer(self, token_inp: tf.Tensor) -> tf.Tensor:
        """
        Creates embedding layer that is in common between many encoders.

        Args:
            token_inp:  2D tensor that is of shape (batch size, sequence length)

        Returns:
            3D tensor of shape (batch size, sequence length, embedding dimension)
        """

        token_embeddings = tf.get_variable(name='token_embeddings',
                                           initializer=tf.glorot_uniform_initializer(),
                                           shape=[len(self.metadata['token_vocab']),
                                                  self.get_hyper('token_embedding_size')],
                                           )
        self.__embeddings = token_embeddings

        token_embeddings = tf.nn.dropout(token_embeddings,
                                         keep_prob=self.placeholders['dropout_keep_rate'])

        return tf.nn.embedding_lookup(params=token_embeddings, ids=token_inp)

    @classmethod
    def init_metadata(cls) -> Dict[str, Any]:
        raw_metadata = super().init_metadata()
        raw_metadata['token_counter'] = Counter()
        return raw_metadata

    @classmethod
    def _to_subtoken_stream(cls, input_stream: Iterable[str], mark_subtoken_end: bool) -> Iterable[str]:
        for token in input_stream:
            if SeqEncoder.IDENTIFIER_TOKEN_REGEX.match(token):
                yield from split_identifier_into_parts(token)
                if mark_subtoken_end:
                    yield '</id>'
            else:
                yield token

    @classmethod
    def load_metadata_from_sample(cls, data_to_load: Iterable[str], raw_metadata: Dict[str, Any],
                                  use_subtokens: bool=False, mark_subtoken_end: bool=False) -> None:
        if use_subtokens:
            data_to_load = cls._to_subtoken_stream(data_to_load, mark_subtoken_end=mark_subtoken_end)
        raw_metadata['token_counter'].update(data_to_load)

    @classmethod
    def finalise_metadata(cls, encoder_label: str, hyperparameters: Dict[str, Any], raw_metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        final_metadata = super().finalise_metadata(encoder_label, hyperparameters, raw_metadata_list)
        merged_token_counter = Counter()
        for raw_metadata in raw_metadata_list:
            merged_token_counter += raw_metadata['token_counter']

        if hyperparameters['%s_use_bpe' % encoder_label]:
            token_vocabulary = BpeVocabulary(vocab_size=hyperparameters['%s_token_vocab_size' % encoder_label],
                                             pct_bpe=hyperparameters['%s_pct_bpe' % encoder_label]
                                             )
            token_vocabulary.fit(merged_token_counter)
        else:
            token_vocabulary = Vocabulary.create_vocabulary(tokens=merged_token_counter,
                                                            max_size=hyperparameters['%s_token_vocab_size' % encoder_label],
                                                            count_threshold=hyperparameters['%s_token_vocab_count_threshold' % encoder_label])

        final_metadata['token_vocab'] = token_vocabulary
        # Save the most common tokens for use in data augmentation:
        final_metadata['common_tokens'] = merged_token_counter.most_common(50)
        return final_metadata

    @classmethod
    def load_data_from_sample(cls,
                              encoder_label: str,
                              hyperparameters: Dict[str, Any],
                              metadata: Dict[str, Any],
                              data_to_load: Any,
                              function_name: Optional[str],
                              result_holder: Dict[str, Any],
                              is_test: bool = True) -> bool:
        """
        Saves two versions of both the code and the query: one using the docstring as the query and the other using the
        function-name as the query, and replacing the function name in the code with an out-of-vocab token.
        Sub-tokenizes, converts, and pads both versions, and rejects empty samples.
        """
        # Save the two versions of the code and query:
        data_holder = {QueryType.DOCSTRING.value: data_to_load, QueryType.FUNCTION_NAME.value: None}
        # Skip samples where the function name is very short, because it probably has too little information
        # to be a good search query.
        if not is_test and hyperparameters['fraction_using_func_name'] > 0. and function_name and \
                len(function_name) >= hyperparameters['min_len_func_name_for_query']:
            if encoder_label == 'query':
                # Set the query tokens to the function name, broken up into its sub-tokens:
                data_holder[QueryType.FUNCTION_NAME.value] = split_identifier_into_parts(function_name)
            elif encoder_label == 'code':
                # In the code, replace the function name with the out-of-vocab token everywhere it appears:
                data_holder[QueryType.FUNCTION_NAME.value] = [Vocabulary.get_unk() if token == function_name else token
                                                              for token in data_to_load]

        # Sub-tokenize, convert, and pad both versions:
        for key, data in data_holder.items():
            if not data:
                result_holder[f'{encoder_label}_tokens_{key}'] = None
                result_holder[f'{encoder_label}_tokens_mask_{key}'] = None
                result_holder[f'{encoder_label}_tokens_length_{key}'] = None
                continue
            if hyperparameters[f'{encoder_label}_use_subtokens']:
                data = cls._to_subtoken_stream(data,
                                               mark_subtoken_end=hyperparameters[
                                                   f'{encoder_label}_mark_subtoken_end'])
            tokens, tokens_mask = \
                convert_and_pad_token_sequence(metadata['token_vocab'], list(data),
                                               hyperparameters[f'{encoder_label}_max_num_tokens'])
            # Note that we share the result_holder with different encoders, and so we need to make our identifiers
            # unique-ish
            result_holder[f'{encoder_label}_tokens_{key}'] = tokens
            result_holder[f'{encoder_label}_tokens_mask_{key}'] = tokens_mask
            result_holder[f'{encoder_label}_tokens_length_{key}'] = int(np.sum(tokens_mask))

        if result_holder[f'{encoder_label}_tokens_mask_{QueryType.DOCSTRING.value}'] is None or \
                int(np.sum(result_holder[f'{encoder_label}_tokens_mask_{QueryType.DOCSTRING.value}'])) == 0:
            return False

        return True

    def extend_minibatch_by_sample(self, batch_data: Dict[str, Any], sample: Dict[str, Any], is_train: bool=False,
                                   query_type: QueryType = QueryType.DOCSTRING.value) -> bool:
        """
        Implements various forms of data augmentation.
        """
        current_sample = dict()

        # Train with some fraction of samples having their query set to the function name instead of the docstring, and
        # their function name replaced with out-of-vocab in the code:
        current_sample['tokens'] = sample[f'{self.label}_tokens_{query_type}']
        current_sample['tokens_mask'] = sample[f'{self.label}_tokens_mask_{query_type}']
        current_sample['tokens_lengths'] = sample[f'{self.label}_tokens_length_{query_type}']

        # In the query, randomly add high-frequency tokens:
        # TODO: Add tokens with frequency proportional to their frequency in the vocabulary
        if is_train and self.label == 'query' and self.hyperparameters['query_random_token_frequency'] > 0.:
            total_length = len(current_sample['tokens'])
            length_without_padding = current_sample['tokens_lengths']
            # Generate a list of places in which to insert tokens:
            insert_indices = np.array([random.uniform(0., 1.) for _ in range(length_without_padding)])  # don't allow insertions in the padding
            insert_indices = insert_indices < self.hyperparameters['query_random_token_frequency']  # insert at the correct frequency
            insert_indices = np.flatnonzero(insert_indices)
            if len(insert_indices) > 0:
                # Generate the random tokens to add:
                tokens_to_add = [random.randrange(0, len(self.metadata['common_tokens']))
                                 for _ in range(len(insert_indices))]  # select one of the most common tokens for each location
                tokens_to_add = [self.metadata['common_tokens'][token][0] for token in tokens_to_add]  # get the word corresponding to the token we're adding
                tokens_to_add = [self.metadata['token_vocab'].get_id_or_unk(token) for token in tokens_to_add]  # get the index within the vocab of the token we're adding
                # Efficiently insert the added tokens, leaving the total length the same:
                to_insert = 0
                output_query = np.zeros(total_length, dtype=int)
                for idx in range(min(length_without_padding, total_length - len(insert_indices))):  # iterate only through the beginning of the array where changes are being made
                    if to_insert < len(insert_indices) and idx == insert_indices[to_insert]:
                        output_query[idx + to_insert] = tokens_to_add[to_insert]
                        to_insert += 1
                    output_query[idx + to_insert] = current_sample['tokens'][idx]
                current_sample['tokens'] = output_query
                # Add the needed number of non-padding values to the mask:
                current_sample['tokens_mask'][length_without_padding:length_without_padding + len(tokens_to_add)] = 1.
                current_sample['tokens_lengths'] += len(tokens_to_add)

        # Add the current sample to the minibatch:
        [batch_data[key].append(current_sample[key]) for key in current_sample.keys() if key in batch_data.keys()]

        return False

    def get_token_embeddings(self) -> Tuple[tf.Tensor, List[str]]:
        return (self.__embeddings,
                list(self.metadata['token_vocab'].id_to_token))
