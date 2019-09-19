from typing import Dict, Any

import tensorflow as tf

from .masked_seq_encoder import MaskedSeqEncoder
from utils.tfutils import pool_sequence_embedding


class NBoWEncoder(MaskedSeqEncoder):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        encoder_hypers = { 'nbow_pool_mode': 'weighted_mean',
                         }
        hypers = super().get_default_hyperparameters()
        hypers.update(encoder_hypers)
        return hypers

    def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
        super().__init__(label, hyperparameters, metadata)

    @property
    def output_representation_size(self):
        return self.get_hyper('token_embedding_size')

    def make_model(self, is_train: bool=False) -> tf.Tensor:
        with tf.variable_scope("nbow_encoder"):
            self._make_placeholders()

            seq_tokens_embeddings = self.embedding_layer(self.placeholders['tokens'])
            seq_token_mask = self.placeholders['tokens_mask']
            seq_token_lengths = tf.reduce_sum(seq_token_mask, axis=1)  # B
            return pool_sequence_embedding(self.get_hyper('nbow_pool_mode').lower(),
                                           sequence_token_embeddings=seq_tokens_embeddings,
                                           sequence_lengths=seq_token_lengths,
                                           sequence_token_masks=seq_token_mask)
