from typing import List, Tuple, Dict, Any, Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.init_ops import Initializer

from dpu_utils.mlutils import Vocabulary

from utils.bpevocabulary import BpeVocabulary

BIG_NUMBER = 1e7


def convert_and_pad_token_sequence(token_vocab: Union[Vocabulary, BpeVocabulary],
                                   token_sequence: List[str],
                                   output_tensor_size: int,
                                   pad_from_left: bool = False) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Tensorise token sequence with padding; returning a mask for used elements as well.

    Args:
        token_vocab: Vocabulary or BPE encoder to use. We assume that token_vocab[0] is the padding symbol.
        token_sequence: List of tokens in string form
        output_tensor_size: Size of the resulting tensor (i.e., length up which we pad / down to which we truncate.
        pad_from_left: Indicate if we are padding/truncating on the left side of string. [Default: False]

    Returns:
        Pair of numpy arrays. First is the actual tensorised token sequence, the second is a masking tensor
        that is 1.0 for those token indices that are actually used.
    """
    if isinstance(token_vocab, BpeVocabulary):
        token_ids = np.array(list(token_vocab.transform([token_sequence], fixed_length=output_tensor_size))[0])
        token_mask = np.array([1 if token_ids[i] > 0 else 0 for i in range(len(token_ids))])
        return token_ids, token_mask

    if pad_from_left:
        token_sequence = token_sequence[-output_tensor_size:]
    else:
        token_sequence = token_sequence[:output_tensor_size]

    sequence_length = len(token_sequence)
    if pad_from_left:
        start_idx = output_tensor_size - sequence_length
    else:
        start_idx = 0

    token_ids = np.zeros(output_tensor_size, dtype=np.int32)
    token_mask = np.zeros(output_tensor_size, dtype=np.float32)
    for i, token in enumerate(token_sequence, start=start_idx):
        token_ids[i] = token_vocab.get_id_or_unk(token)
        token_mask[i] = True

    return token_ids, token_mask


def write_to_feed_dict(feed_dict: Dict[tf.Tensor, Any], placeholder, val) -> None:
    if len(val) == 0:
        ph_shape = [dim if dim is not None else 0 for dim in placeholder.shape.as_list()]
        feed_dict[placeholder] = np.empty(ph_shape)
    else:
        feed_dict[placeholder] = val


class NoisyIdentityInitializer(Initializer):
    def __init__(self, noise: float=1e-1):
        self.__noise = noise
        self.__identity_initializer = tf.initializers.identity()
        self.__noise_initializer = tf.initializers.random_uniform(minval=-self.__noise, maxval=self.__noise)

    def set_config(self):
        return {
            "noise": self.__noise,
        }

    def __call__(self, shape, dtype=None, partition_info=None):
        identity = self.__identity_initializer(shape=shape, dtype=dtype, partition_info=partition_info)
        noise = self.__noise_initializer(shape=shape, dtype=dtype, partition_info=partition_info)
        return identity + noise


def get_activation(activation_fun: Optional[str]):
    if activation_fun is None:
        return None
    activation_fun = activation_fun.lower()
    if activation_fun == 'linear':
        return None
    if activation_fun == 'tanh':
        return tf.tanh
    if activation_fun == 'relu':
        return tf.nn.relu
    if activation_fun == 'leaky_relu':
        return tf.nn.leaky_relu
    if activation_fun == 'elu':
        return tf.nn.elu
    if activation_fun == 'selu':
        return tf.nn.selu
    if activation_fun == 'gelu':
        def gelu(input_tensor):
            cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
            return input_tensor * cdf
        return gelu
    else:
        raise ValueError("Unknown activation function '%s'!" % activation_fun)


def pool_sequence_embedding(pool_mode: str,
                            sequence_token_embeddings: tf.Tensor,
                            sequence_lengths: tf.Tensor,
                            sequence_token_masks: tf.Tensor) -> tf.Tensor:
    """
    Takes a batch of sequences of token embeddings and applies a pooling function,
    returning one representation for each sequence.

    Args:
        pool_mode: The pooling mode, one of "mean", "max", "weighted_mean". For
         the latter, a weight network is introduced that computes a score (from [0,1])
         for each token, and embeddings are weighted by that score when computing
         the mean.
        sequence_token_embeddings: A float32 tensor of shape [B, T, D], where B is the
         batch dimension, T is the maximal number of tokens per sequence, and D is
         the embedding size.
        sequence_lengths: An int32 tensor of shape [B].
        sequence_token_masks: A float32 tensor of shape [B, T] with 0/1 values used
         for masking out unused entries in sequence_embeddings.
    Returns:
        A tensor of shape [B, D], containing the pooled representation for each
        sequence.
    """
    if pool_mode == 'mean':
        seq_token_embeddings_masked = \
            sequence_token_embeddings * tf.expand_dims(sequence_token_masks, axis=-1)  # B x T x D
        seq_token_embeddings_sum = tf.reduce_sum(seq_token_embeddings_masked, axis=1)  # B x D
        sequence_lengths = tf.expand_dims(tf.cast(sequence_lengths, dtype=tf.float32), axis=-1)  # B x 1
        return seq_token_embeddings_sum / sequence_lengths
    elif pool_mode == 'max':
        sequence_token_masks = -BIG_NUMBER * (1 - sequence_token_masks)  # B x T
        sequence_token_masks = tf.expand_dims(sequence_token_masks, axis=-1)  # B x T x 1
        return tf.reduce_max(sequence_token_embeddings + sequence_token_masks, axis=1)
    elif pool_mode == 'weighted_mean':
        token_weights = tf.layers.dense(sequence_token_embeddings,
                                        units=1,
                                        activation=tf.sigmoid,
                                        use_bias=False)  # B x T x 1
        token_weights *= tf.expand_dims(sequence_token_masks, axis=-1)  # B x T x 1
        seq_embedding_weighted_sum = tf.reduce_sum(sequence_token_embeddings * token_weights, axis=1)  # B x D
        return seq_embedding_weighted_sum / (tf.reduce_sum(token_weights, axis=1) + 1e-8)  # B x D
    else:
        raise ValueError("Unknown sequence pool mode '%s'!" % pool_mode)
