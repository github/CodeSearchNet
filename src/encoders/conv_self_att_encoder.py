from typing import Dict, Any

import tensorflow as tf

from .utils.bert_self_attention import BertConfig, BertModel
from .masked_seq_encoder import MaskedSeqEncoder
from utils.tfutils import get_activation, pool_sequence_embedding


class ConvSelfAttentionEncoder(MaskedSeqEncoder):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        encoder_hypers = {'1dcnn_position_encoding': 'none',  # One of {'none', 'learned'}
                          '1dcnn_layer_list': [128, 128],
                          '1dcnn_kernel_width': [8, 8],  # Has to have same length as 1dcnn_layer_list
                          '1dcnn_add_residual_connections': True,
                          '1dcnn_activation': 'tanh',

                          'self_attention_activation': 'gelu',
                          'self_attention_hidden_size': 128,
                          'self_attention_intermediate_size': 512,
                          'self_attention_num_layers': 2,
                          'self_attention_num_heads': 8,
                          'self_attention_pool_mode': 'weighted_mean',
                          }
        hypers = super().get_default_hyperparameters()
        hypers.update(encoder_hypers)
        return hypers

    def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
        super().__init__(label, hyperparameters, metadata)

    @property
    def output_representation_size(self):
        return self.get_hyper('self_attention_hidden_size')

    def make_model(self, is_train: bool = False) -> tf.Tensor:
        with tf.variable_scope("self_attention_encoder"):
            self._make_placeholders()

            seq_tokens_embeddings = self.embedding_layer(self.placeholders['tokens'])

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

            config = BertConfig(vocab_size=self.get_hyper('token_vocab_size'),
                                hidden_size=self.get_hyper('self_attention_hidden_size'),
                                num_hidden_layers=self.get_hyper('self_attention_num_layers'),
                                num_attention_heads=self.get_hyper('self_attention_num_heads'),
                                intermediate_size=self.get_hyper('self_attention_intermediate_size'))

            model = BertModel(config=config,
                              is_training=is_train,
                              input_ids=self.placeholders['tokens'],
                              input_mask=self.placeholders['tokens_mask'],
                              use_one_hot_embeddings=False,
                              embedded_input=current_embeddings)

            output_pool_mode = self.get_hyper('self_attention_pool_mode').lower()
            if output_pool_mode == 'bert':
                return model.get_pooled_output()
            else:
                seq_token_embeddings = model.get_sequence_output()
                seq_token_masks = self.placeholders['tokens_mask']
                seq_token_lengths = tf.reduce_sum(seq_token_masks, axis=1)  # B
                return pool_sequence_embedding(output_pool_mode,
                                               sequence_token_embeddings=seq_token_embeddings,
                                               sequence_lengths=seq_token_lengths,
                                               sequence_token_masks=seq_token_masks)
