from typing import Dict, Any, Iterable, Optional

import numpy as np
import tensorflow as tf

from .seq_encoder import SeqEncoder
from utils.tfutils import write_to_feed_dict, pool_sequence_embedding


class MaskedSeqEncoder(SeqEncoder):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        encoder_hypers = {
                         }
        hypers = super().get_default_hyperparameters()
        hypers.update(encoder_hypers)
        return hypers

    def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
        super().__init__(label, hyperparameters, metadata)

    def _make_placeholders(self):
        """
        Creates placeholders "tokens" and "tokens_mask" for masked sequence encoders.
        """
        super()._make_placeholders()
        self.placeholders['tokens_mask'] = \
            tf.placeholder(tf.float32,
                           shape=[None, self.get_hyper('max_num_tokens')],
                           name='tokens_mask')

    def init_minibatch(self, batch_data: Dict[str, Any]) -> None:
        super().init_minibatch(batch_data)
        batch_data['tokens'] = []
        batch_data['tokens_mask'] = []

    def minibatch_to_feed_dict(self, batch_data: Dict[str, Any], feed_dict: Dict[tf.Tensor, Any], is_train: bool) -> None:
        super().minibatch_to_feed_dict(batch_data, feed_dict, is_train)
        write_to_feed_dict(feed_dict, self.placeholders['tokens'], batch_data['tokens'])
        write_to_feed_dict(feed_dict, self.placeholders['tokens_mask'], batch_data['tokens_mask'])
