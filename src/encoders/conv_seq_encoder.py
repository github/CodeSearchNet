from typing import Dict, Any

import tensorflow as tf

from .masked_seq_encoder import MaskedSeqEncoder
from utils.tfutils import get_activation, pool_sequence_embedding


class ConvolutionSeqEncoder(MaskedSeqEncoder):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        encoder_hypers = {'1dcnn_position_encoding': 'learned',  # One of {'none', 'learned'}
                          '1dcnn_layer_list': [128, 128, 128],
                          '1dcnn_kernel_width': [16, 16, 16],  # Has to have same length as 1dcnn_layer_list
                          '1dcnn_add_residual_connections': True,
                          '1dcnn_activation': 'tanh',
                          '1dcnn_pool_mode': 'weighted_mean',
                         }
        hypers = super().get_default_hyperparameters()
        hypers.update(encoder_hypers)
        return hypers

    def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
        super().__init__(label, hyperparameters, metadata)

    @property
    def output_representation_size(self):
        return self.get_hyper('1dcnn_layer_list')[-1]

    def make_model(self, is_train: bool=False) -> tf.Tensor:
        with tf.variable_scope("1dcnn_encoder"):
            self._make_placeholders()

            seq_tokens_embeddings = self.embedding_layer(self.placeholders['tokens'])
            seq_tokens_embeddings = self.__add_position_encoding(seq_tokens_embeddings)

            activation_fun = get_activation(self.get_hyper('1dcnn_activation'))
            current_embeddings = seq_tokens_embeddings
            num_filters_and_width = zip(self.get_hyper('1dcnn_layer_list'), self.get_hyper('1dcnn_kernel_width'))
            for (layer_idx, (num_filters, kernel_width)) in enumerate(num_filters_and_width):
                next_embeddings = tf.layers.conv1d(
                    inputs=current_embeddings,
                    filters=num_filters,
                    kernel_size=kernel_width,
                    padding="same")

                # Add residual connections past the first layer.
                if self.get_hyper('1dcnn_add_residual_connections') and layer_idx > 0:
                    next_embeddings += current_embeddings

                current_embeddings = activation_fun(next_embeddings)

                current_embeddings = tf.nn.dropout(current_embeddings,
                                                   keep_prob=self.placeholders['dropout_keep_rate'])

            seq_token_mask = self.placeholders['tokens_mask']
            seq_token_lengths = tf.reduce_sum(seq_token_mask, axis=1)  # B
            return pool_sequence_embedding(self.get_hyper('1dcnn_pool_mode').lower(),
                                           sequence_token_embeddings=current_embeddings,
                                           sequence_lengths=seq_token_lengths,
                                           sequence_token_masks=seq_token_mask)

    def __add_position_encoding(self, seq_inputs: tf.Tensor) -> tf.Tensor:
        position_encoding = self.get_hyper('1dcnn_position_encoding').lower()
        if position_encoding == 'none':
            return seq_inputs
        elif position_encoding == 'learned':
            position_embeddings = \
                tf.get_variable(name='position_embeddings',
                                initializer=tf.truncated_normal_initializer(stddev=0.02),
                                shape=[self.get_hyper('max_num_tokens'),
                                       self.get_hyper('token_embedding_size')],
                                )
            # Add batch dimension to position embeddings to make broadcasting work, then add:
            return seq_inputs + tf.expand_dims(position_embeddings, axis=0)
        else:
            raise ValueError("Unknown position encoding '%s'!" % position_encoding)
