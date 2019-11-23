from abc import ABC, abstractmethod
from typing import *

import tensorflow as tf

__all__ = ("Loss", "SoftMax", "Cosine", "MaxMargin", "Triplet")


def get_logits(query_encoded, code_encoded):
    return tf.matmul(
        query_encoded,
        code_encoded,
        transpose_a=False,
        transpose_b=True,
        name='code_query_cooccurrence_logits',
    )  # B x B


class Loss(ABC):
    @abstractmethod
    def make_loss(self, query_encoded, code_encoded):
        pass


class _SoftMax(Loss):
    def make_loss(self, query_encoded, code_encoded) -> Tuple[Any, Any]:
        logits = get_logits(query_encoded, code_encoded)
        per_sample_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.range(tf.shape(code_encoded)[0]),  # [0, 1, 2, 3, ..., n]
            logits=logits,
        )
        return logits, per_sample_loss


class Cosine(Loss):
    def __init__(self, margin):
        self.margin = margin

    def make_loss(self, query_encoded, code_encoded):
        query_norms = tf.norm(query_encoded, axis=-1, keep_dims=True) + 1e-10
        code_norms = tf.norm(code_encoded, axis=-1, keep_dims=True) + 1e-10
        cosine_similarities = get_logits(query_encoded / query_norms, code_encoded / code_norms)
        # A max-margin-like loss, but do not penalize negative cosine similarities.
        neg_matrix = tf.diag(tf.fill(dims=[tf.shape(cosine_similarities)[0]], value=float('-inf')))
        per_sample_loss = tf.maximum(
            0.,
            self.margin - tf.diag_part(cosine_similarities)
            + tf.reduce_max(tf.nn.relu(cosine_similarities + neg_matrix), axis=-1),
        )
        return cosine_similarities, per_sample_loss


class MaxMargin(Loss):
    def __init__(self, margin):
        self.margin = margin

    def make_loss(self, query_encoded, code_encoded):
        logits = get_logits(query_encoded, code_encoded)
        logprobs = tf.nn.log_softmax(logits)
        min_inf_matrix = tf.diag(tf.fill(dims=[tf.shape(logprobs)[0]], value=float('-inf')))
        per_sample_loss = tf.maximum(
            0.,
            self.margin - tf.diag_part(logprobs) + tf.reduce_max(logprobs + min_inf_matrix, axis=-1),
        )
        return logits, per_sample_loss


class Triplet(Loss):
    def __init__(self, margin):
        self.margin = margin

    def make_loss(self, query_encoded, code_encoded):
        query_reps = tf.broadcast_to(
            query_encoded,
            shape=[tf.shape(query_encoded)[0], tf.shape(query_encoded)[0], tf.shape(query_encoded)[1]],
        )  # B*xBxD
        code_reps = tf.broadcast_to(
            code_encoded,
            shape=[tf.shape(code_encoded)[0], tf.shape(code_encoded)[0], tf.shape(code_encoded)[1]],
        )  # B*xBxD
        code_reps = tf.transpose(code_reps, perm=(1, 0, 2))  # BxB*xD

        all_pair_distances = tf.norm(query_reps - code_reps, axis=-1)  # BxB
        correct_distances = tf.expand_dims(tf.diag_part(all_pair_distances), axis=-1)  # Bx1

        pointwise_loss = tf.nn.relu(correct_distances - all_pair_distances + self.margin)  # BxB
        pointwise_loss *= (1 - tf.eye(tf.shape(pointwise_loss)[0]))

        per_sample_loss = tf.reduce_sum(pointwise_loss, axis=-1) / (
                    tf.reduce_sum(tf.cast(tf.greater(pointwise_loss, 0), dtype=tf.float32), axis=-1) + 1e-10)  # B

        return -all_pair_distances, per_sample_loss


# singleton for the unparametrized softmax loss
SoftMax = _SoftMax()
