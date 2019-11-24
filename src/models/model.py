import os
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
from enum import Enum, auto
from typing import *

import numpy as np
import tensorflow as tf
import wandb
from dpu_utils.utils import RichPath

from encoders import Encoder, QueryType

LoadedSamples = Dict[str, List[Dict[str, Any]]]
SampleId = Tuple[str, int]


class RepresentationType(Enum):
    CODE = auto()
    QUERY = auto()


class Model(ABC):
    query_encoder_type: Type[Encoder]

    @classmethod
    @abstractmethod
    def code_encoder_type(cls, language: str) -> Type[Encoder]:
        raise NotImplementedError


    @classmethod
    @abstractmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        return {
                'batch_size': 200,

                'optimizer': 'Adam',
                'seed': 0,
                'dropout_keep_rate': 0.9,
                'learning_rate': 0.01,
                'learning_rate_code_scale_factor': 1.,
                'learning_rate_query_scale_factor': 1.,
                'learning_rate_decay': 0.98,
                'momentum': 0.85,
                'gradient_clip': 1,
                'loss': 'softmax',  # One of softmax, cosine, max-margin
                'margin': 1,
                'max_epochs': 500,
                'patience': 5,

                # Fraction of samples for which the query should be the function name instead of the docstring:
                'fraction_using_func_name': 0.1,
                # Only functions with a name at least this long will be candidates for training with the function name
                # as the query instead of the docstring:
                'min_len_func_name_for_query': 12,
                # Frequency at which random, common tokens are added into the query:
                'query_random_token_frequency': 0.,

                # Maximal number of tokens considered to compute a representation for code/query:
                'code_max_num_tokens': 200,
                'query_max_num_tokens': 30,
               }

    def __init__(self,
                 hyperparameters: Dict[str, Any],
                 run_name: Optional[str] = None,
                 model_save_dir: Optional[str] = None,
                 log_save_dir: Optional[str] = None) -> None:
        self.__code_encoders: OrderedDict[str, Any] = OrderedDict()  # OrderedDict as we are using the order of languages a few times...

        self.__query_encoder: Any = None

        # start with default hyper-params and then override them
        self.hyperparameters = self.get_default_hyperparameters()
        self.hyperparameters.update(hyperparameters)

        self.__query_metadata: Dict[str, Any] = {}
        self.__per_code_language_metadata: Dict[str, Dict[str, Any]] = {}
        self.__placeholders: Dict[str, Union[tf.placeholder, tf.placeholder_with_default]] = {}
        self.__ops: Dict[str, Any] = {}
        if run_name is None:
            run_name = type(self).__name__
        self.__run_name = run_name

        if model_save_dir is None:
            self.__model_save_dir = os.environ.get('PHILLY_MODEL_DIRECTORY', default='.')  # type: str
        else:
            self.__model_save_dir = model_save_dir  # type: str

        if log_save_dir is None:
            self.__log_save_dir = os.environ.get('PHILLY_LOG_DIRECTORY', default='.')  # type: str
        else:
            self.__log_save_dir = log_save_dir  # type: str

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if "gpu_device_id" in self.hyperparameters:
            config.gpu_options.visible_device_list = str(self.hyperparameters["gpu_device_id"])

        graph = tf.Graph()
        self.__sess = tf.Session(graph=graph, config=config)

        # save directory as tensorboard.
        self.__tensorboard_dir = log_save_dir

    @property
    def query_metadata(self):
        return self.__query_metadata

    @property
    def per_code_language_metadata(self):
        return self.__per_code_language_metadata

    @property
    def placeholders(self):
        return self.__placeholders

    @property
    def ops(self):
        return self.__ops

    @property
    def sess(self):
        return self.__sess

    @property
    def run_name(self):
        return self.__run_name

    @property
    def representation_size(self) -> int:
        return self.__query_encoder.output_representation_size

    def _log_tensorboard_scalar(self, tag:str, value:float, step:int) -> None:
        """Log scalar values that are not ops to tensorboard."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        self.__summary_writer.add_summary(summary, step)
        self.__summary_writer.flush()

    def save(self, path: RichPath) -> None:
        variables_to_save = list(set(self.__sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
        weights_to_save = self.__sess.run(variables_to_save)
        weights_to_save = {var.name: value
                           for (var, value) in zip(variables_to_save, weights_to_save)}

        data_to_save = {
                         "model_type": type(self).__name__,
                         "hyperparameters": self.hyperparameters,
                         "query_metadata": self.__query_metadata,
                         "per_code_language_metadata": self.__per_code_language_metadata,
                         "weights": weights_to_save,
                         "run_name": self.__run_name,
                       }

        path.save_as_compressed_file(data_to_save)

    def train_log(self, msg) -> None:
        log_path = os.path.join(self.__log_save_dir,
                                f'{self.run_name}.train_log')
        with open(log_path, mode='a', encoding='utf-8') as f:
            f.write(msg + "\n")
        print(msg.encode('ascii', errors='replace').decode())

    def test_log(self, msg) -> None:
        log_path = os.path.join(self.__log_save_dir,
                                f'{self.run_name}.test_log')
        with open(log_path, mode='a', encoding='utf-8') as f:
            f.write(msg + "\n")
        print(msg.encode('ascii', errors='replace').decode())

    def make_model(self, is_train: bool):
        with self.__sess.graph.as_default():
            random.seed(self.hyperparameters['seed'])
            np.random.seed(self.hyperparameters['seed'])
            tf.set_random_seed(self.hyperparameters['seed'])

            self._make_model(is_train=is_train)
            self._make_loss()
            if is_train:
                self._make_training_step()
                self.__summary_writer = tf.summary.FileWriter(self.__tensorboard_dir, self.__sess.graph)

    def _make_model(self, is_train: bool) -> None:
        """
        Create the actual model.

        Note: This has to create self.ops['code_representations'] and self.ops['query_representations'],
        tensors of the same shape and rank 2.
        """
        self.__placeholders['dropout_keep_rate'] = tf.placeholder(tf.float32,
                                                                  shape=(),
                                                                  name='dropout_keep_rate')
        self.__placeholders['sample_loss_weights'] = \
            tf.placeholder_with_default(input=np.ones(shape=[self.hyperparameters['batch_size']],
                                                      dtype=np.float32),
                                        shape=[self.hyperparameters['batch_size']],
                                        name='sample_loss_weights')

        with tf.variable_scope("code_encoder"):
            language_encoders = []
            for (language, language_metadata) in sorted(self.__per_code_language_metadata.items(), key=lambda kv: kv[0]):
                encoder_type = self.code_encoder_type(language)
                with tf.variable_scope(language):
                    self.__code_encoders[language] = encoder_type(
                        label="code", hyperparameters=self.hyperparameters, metadata=language_metadata)
                    language_encoders.append(self.__code_encoders[language].make_model(is_train=is_train))
            self.ops['code_representations'] = tf.concat(language_encoders, axis=0)
        with tf.variable_scope("query_encoder"):
            self.__query_encoder = self.query_encoder_type(
                label="query", hyperparameters=self.hyperparameters, metadata=self.__query_metadata)
            self.ops['query_representations'] = self.__query_encoder.make_model(is_train=is_train)

        code_representation_size = next(iter(self.__code_encoders.values())).output_representation_size
        query_representation_size = self.__query_encoder.output_representation_size
        assert code_representation_size == query_representation_size, \
            f'Representations produced for code ({code_representation_size}) and query ({query_representation_size}) cannot differ!'

    def get_code_token_embeddings(self, language: str) -> Tuple[tf.Tensor, List[str]]:
        with self.__sess.graph.as_default():
            with tf.variable_scope("code_encoder"):
                return self.__code_encoders[language].get_token_embeddings()

    def get_query_token_embeddings(self) -> Tuple[tf.Tensor, List[str]]:
        with self.__sess.graph.as_default():
            with tf.variable_scope("query_encoder"):
                return self.__query_encoder.get_token_embeddings()

    def _make_loss(self) -> None:
        import loss
        if self.hyperparameters['loss'] == 'softmax':
            loss = loss.SoftMax
        elif self.hyperparameters['loss'] == 'cosine':
            loss = loss.Cosine(self.hyperparameters['margin'])
        elif self.hyperparameters['loss'] == 'max-margin':
            loss = loss.MaxMargin(self.hyperparameters['margin'])
        elif self.hyperparameters['loss'] == 'triplet':
            loss = loss.Triplet(self.hyperparameters['margin'])
        else:
            raise Exception(f'Unrecognized loss-type "{self.hyperparameters["loss"]}"')
        similarity_scores, per_sample_loss = loss.make_loss(
            self.ops['query_representations'], self.ops['code_representations'])

        per_sample_loss = per_sample_loss * self.placeholders['sample_loss_weights']
        self.ops['loss'] = tf.reduce_sum(per_sample_loss) / tf.reduce_sum(self.placeholders['sample_loss_weights'])

        # extract the logits from the diagonal of the matrix, which are the logits corresponding to the ground-truth
        correct_scores = tf.diag_part(similarity_scores)
        # compute how many queries have bigger logits than the ground truth (the diagonal) -> which will be incorrectly ranked
        compared_scores = similarity_scores >= tf.expand_dims(correct_scores, axis=-1)
        # for each row of the matrix (query), sum how many logits are larger than the ground truth
        # ...then take the reciprocal of that to get the MRR for each individual query (you will need to take the mean later)
        self.ops['mrr'] = 1 / tf.reduce_sum(tf.to_float(compared_scores), axis=1)

    def _make_training_step(self) -> None:
        """
        Constructs self.ops['train_step'] from self.ops['loss'] and hyperparameters.
        """
        optimizer_name = self.hyperparameters['optimizer'].lower()
        if optimizer_name == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.hyperparameters['learning_rate'])
        elif optimizer_name == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.hyperparameters['learning_rate'],
                                                  decay=self.hyperparameters['learning_rate_decay'],
                                                  momentum=self.hyperparameters['momentum'])
        elif optimizer_name == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.hyperparameters['learning_rate'])
        else:
            raise Exception('Unknown optimizer "%s".' % (self.hyperparameters['optimizer']))

        # Calculate and clip gradients
        trainable_vars = self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        gradients = tf.gradients(self.ops['loss'], trainable_vars)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.hyperparameters['gradient_clip'])
        pruned_clipped_gradients = []
        for (gradient, trainable_var) in zip(clipped_gradients, trainable_vars):
            if gradient is None:
                continue
            if trainable_var.name.startswith("code_encoder/"):
                gradient *= tf.constant(self.hyperparameters['learning_rate_code_scale_factor'],
                                        dtype=tf.float32)
            elif trainable_var.name.startswith("query_encoder/"):
                gradient *= tf.constant(self.hyperparameters['learning_rate_query_scale_factor'],
                                        dtype=tf.float32)

            pruned_clipped_gradients.append((gradient, trainable_var))
        self.ops['train_step'] = optimizer.apply_gradients(pruned_clipped_gradients)

    def load_metadata(self, data_dirs: List[RichPath], max_files_per_dir: Optional[int] = None, parallelize: bool = True) -> None:
        from metadata import MetadataLoader
        loader = MetadataLoader(self.query_encoder_type, self.code_encoder_type, self.hyperparameters)

        if parallelize:
            loader.load_parallel(data_dirs, max_files_per_dir)
            finalized = loader.finalize_parallel()
        else:
            loader.load_sequential(data_dirs, max_files_per_dir)
            finalized = loader.finalize_metadata()

        self.__query_metadata, self.__per_code_language_metadata = finalized

    def load_existing_metadata(self, metadata_path: RichPath):
        saved_data = metadata_path.read_by_file_suffix()

        hyper_names = set(self.hyperparameters.keys())
        hyper_names.update(saved_data['hyperparameters'].keys())
        for hyper_name in hyper_names:
            old_hyper_value = saved_data['hyperparameters'].get(hyper_name)
            new_hyper_value = self.hyperparameters.get(hyper_name)
            if old_hyper_value != new_hyper_value:
                self.train_log("I: Hyperparameter %s now has value '%s' but was '%s' when tensorising data."
                               % (hyper_name, new_hyper_value, old_hyper_value))

        self.__query_metadata = saved_data['query_metadata']
        self.__per_code_language_metadata = saved_data['per_code_language_metadata']

    def dataloader(self):
        from dataloader import DataLoader
        return DataLoader(
            self.query_encoder_type,
            self.code_encoder_type,
            self.hyperparameters,
            self.__query_metadata,
            self.__per_code_language_metadata,
        )

    def __init_minibatch(self) -> Dict[str, Any]:
        """
        Returns:
            An empty data holder for minibatch construction.
        """
        batch_data: Dict[str, Any] = dict()
        batch_data['samples_in_batch'] = 0
        batch_data['batch_finished'] = False
        # This bit is a bit tricky. To keep the alignment between code and query bits, we need
        # to keep everything separate here (including the query data). When we finalise a minibatch,
        # we will join (concatenate) all the query info and send it to the query encoder. The
        # language-specific bits get sent to the language-specific encoders, but are ordered such
        # that concatenating the results of the code encoders gives us something that is aligned
        # with the query encoder output.
        batch_data['per_language_query_data'] = {}
        batch_data['per_language_code_data'] = {}
        for (language, language_encoder) in self.__code_encoders.items():
            batch_data['per_language_query_data'][language] = {}
            batch_data['per_language_query_data'][language]['query_sample_ids'] = []
            self.__query_encoder.init_minibatch(batch_data['per_language_query_data'][language])
            batch_data['per_language_code_data'][language] = {}
            batch_data['per_language_code_data'][language]['code_sample_ids'] = []
            language_encoder.init_minibatch(batch_data['per_language_code_data'][language])
        return batch_data

    def __extend_minibatch_by_sample(self,
                                     batch_data: Dict[str, Any],
                                     sample: Dict[str, Any],
                                     language: str,
                                     sample_id: SampleId,
                                     include_query: bool = True,
                                     include_code: bool = True,
                                     is_train: bool = False) -> bool:
        """
        Extend a minibatch under construction by one sample.

        Args:
            batch_data: The minibatch data.
            sample: The sample to add.
            language: The (programming) language of the same to add.
            sample_id: Unique identifier of the example.
            include_code: Flag indicating if the code data needs to be added.
            include_query: Flag indicating if the query data needs to be added.
            is_train: Flag indicating if we are in train mode (which causes data augmentation)

        Returns:
            True iff the minibatch is full after this sample.
        """
        minibatch_is_full = False

        # Train with some fraction of samples having their query set to the function name instead of the docstring, and
        # their function name replaced with out-of-vocab in the code:
        query_type = QueryType.DOCSTRING.value
        if is_train and sample[f'query_tokens_{QueryType.FUNCTION_NAME.value}'] is not None and \
                random.uniform(0., 1.) < self.hyperparameters['fraction_using_func_name']:
            query_type = QueryType.FUNCTION_NAME.value

        if include_query:
            batch_data['per_language_query_data'][language]['query_sample_ids'].append(sample_id)
            minibatch_is_full |= self.__query_encoder.extend_minibatch_by_sample(
                batch_data['per_language_query_data'][language], sample, is_train=is_train, query_type=query_type)
        if include_code:
            batch_data['per_language_code_data'][language]['code_sample_ids'].append(sample_id)
            minibatch_is_full |= self.__code_encoders[language].extend_minibatch_by_sample(
                batch_data['per_language_code_data'][language], sample, is_train=is_train, query_type=query_type)
        return minibatch_is_full or batch_data['samples_in_batch'] >= self.hyperparameters['batch_size']

    def __minibatch_to_feed_dict(self,
                                 batch_data: Dict[str, Any],
                                 language_to_reweighting_factor: Optional[Dict[str, float]],
                                 is_train: bool) -> Tuple[Dict[tf.Tensor, Any], List[SampleId]]:
        """
        Take a collected minibatch and turn it into something that can be fed directly to the constructed model

        Args:
            batch_data: The minibatch data (initialised by __init_minibatch and repeatedly filled by __extend_minibatch_by_sample)
            language_to_reweighting_factor: Optional map from language to the language-specific weighting factor. If not present,
              no reweighting will be performed.
            is_train: Flag indicating if we are in train mode (to set dropout properly)

        Returns:
            A pair of a map from model placeholders to appropriate data structures and a list of sample ids
            such that id_list[i] = id means that the i-th minibatch entry corresponds to the sample identified by id.
        """
        final_minibatch = {self.__placeholders['dropout_keep_rate']: self.hyperparameters['dropout_keep_rate'] if is_train else 1.0}

        # Finalise the code representations while joining the query information:
        full_query_batch_data: Dict[str, Any] = {'code_sample_ids': []}
        language_weights = []
        for (language, language_encoder) in self.__code_encoders.items():
            language_encoder.minibatch_to_feed_dict(batch_data['per_language_code_data'][language], final_minibatch, is_train)
            full_query_batch_data['code_sample_ids'].extend(batch_data['per_language_code_data'][language]['code_sample_ids'])

            for (key, value) in batch_data['per_language_query_data'][language].items():
                if key in full_query_batch_data:
                    if isinstance(value, list):
                        full_query_batch_data[key].extend(value)
                    elif isinstance(value, int):
                        full_query_batch_data[key] += value
                    else:
                        raise ValueError()
                else:
                    full_query_batch_data[key] = value
            if language_to_reweighting_factor is not None:
                language_weights.extend([language_to_reweighting_factor[language]] * len(batch_data['per_language_code_data'][language]['tokens']))

        self.__query_encoder.minibatch_to_feed_dict(full_query_batch_data, final_minibatch, is_train)
        if language_to_reweighting_factor is not None:
            final_minibatch[self.__placeholders['sample_loss_weights']] = language_weights
        if len(full_query_batch_data['query_sample_ids']) > 0:  # If we are only computing code representations, this will be empty
            return final_minibatch, full_query_batch_data['query_sample_ids']
        else:
            return final_minibatch, full_query_batch_data['code_sample_ids']

    def __split_data_into_minibatches(self,
                                      data: LoadedSamples,
                                      is_train: bool = False,
                                      include_query: bool = True,
                                      include_code: bool = True,
                                      drop_incomplete_final_minibatch: bool = True,
                                      compute_language_weightings: bool = False) \
            -> Iterable[Tuple[Dict[tf.Tensor, Any], Any, int, List[SampleId]]]:
        """
        Take tensorised data and chunk into feed dictionaries corresponding to minibatches.

        Args:
            data: The tensorised input data.
            is_train: Flag indicating if we are in train mode (which causes shuffling and the use of dropout)
            include_query: Flag indicating if query data should be included.
            include_code: Flag indicating if code data should be included.
            drop_incomplete_final_minibatch: If True, all returned minibatches will have the configured size
             and some examples from data may not be considered at all. If False, the final minibatch will
             be shorter than the configured size.
            compute_language_weightings: If True, produces weights for samples that normalise the loss
             contribution of each language to be 1/num_languages.

        Returns:
            Iterable sequence of 4-tuples:
              (1) A feed dict mapping placeholders to values,
              (2) Number of samples in the batch
              (3) Total number of datapoints processed
              (4) List of IDs that connect the minibatch elements to the inputs. Concretely,
                  element id_list[i] = (lang, j) indicates that the i-th result in the batch
                  corresponds to the sample data[lang][j].
        """
        # We remove entries from language_to_num_remaining_samples once None are remaining:
        language_to_num_remaining_samples, language_to_idx_list = {}, {}
        for (language, samples) in data.items():
            num_samples = len(samples)
            language_to_num_remaining_samples[language] = num_samples
            sample_idx_list = np.arange(num_samples)
            if is_train:
                np.random.shuffle(sample_idx_list)
            language_to_idx_list[language] = sample_idx_list

        if compute_language_weightings:
            # We want to weigh languages equally, and thus normalise the loss of their samples with
            # total_num_samples * 1/num_languages * 1/num_samples_per_language.
            # Then, assuming a loss of 1 per sample for simplicity, the total loss attributed to a language is
            #    \sum_{1 \leq i \leq num_samples_per_language} total_num_samples / (num_languages * num_samples_per_language)
            #  =  num_samples_per_language * total_num_samples / (num_languages * num_samples_per_language)
            #  =                             total_num_samples / num_languages
            total_num_samples = sum(language_to_num_remaining_samples.values())
            num_languages = len(language_to_num_remaining_samples)
            language_to_reweighting_factor = {language: float(total_num_samples)/(num_languages * num_samples_per_language)
                                              for (language, num_samples_per_language) in language_to_num_remaining_samples.items()}
        else:
            language_to_reweighting_factor = None  # type: ignore

        total_samples_used = 0
        batch_data = self.__init_minibatch()

        while len(language_to_num_remaining_samples) > 0:
            # Pick a language for the sample, by weighted sampling over the remaining data points:
            remaining_languages = list(language_to_num_remaining_samples.keys())
            total_num_remaining_samples = sum(language_to_num_remaining_samples.values())
            picked_language = np.random.choice(a=remaining_languages,
                                               p=[float(language_to_num_remaining_samples[lang]) / total_num_remaining_samples
                                                  for lang in remaining_languages])

            # Pick an example for the given language, and update counters:
            picked_example_idx = language_to_num_remaining_samples[picked_language] - 1  # Note that indexing is 0-based and counting 1-based...
            language_to_num_remaining_samples[picked_language] -= 1
            if language_to_num_remaining_samples[picked_language] == 0:
                del(language_to_num_remaining_samples[picked_language])  # We are done with picked_language now
            picked_sample = data[picked_language][language_to_idx_list[picked_language][picked_example_idx]]

            # Add the example to the current minibatch under preparation:
            batch_data['samples_in_batch'] += 1
            batch_finished = self.__extend_minibatch_by_sample(batch_data,
                                                               picked_sample,
                                                               language=picked_language,
                                                               sample_id=(picked_language, language_to_idx_list[picked_language][picked_example_idx]),
                                                               include_query=include_query,
                                                               include_code=include_code,
                                                               is_train=is_train
                                                               )
            total_samples_used += 1

            if batch_finished:
                feed_dict, original_sample_ids = self.__minibatch_to_feed_dict(batch_data, language_to_reweighting_factor, is_train)
                yield feed_dict, batch_data['samples_in_batch'], total_samples_used, original_sample_ids
                batch_data = self.__init_minibatch()

        if not drop_incomplete_final_minibatch and batch_data['samples_in_batch'] > 0:
            feed_dict, original_sample_ids = self.__minibatch_to_feed_dict(batch_data, language_to_reweighting_factor, is_train)
            yield feed_dict, batch_data['samples_in_batch'], total_samples_used, original_sample_ids

    def __run_epoch_in_batches(self, data: LoadedSamples, epoch_name: str, is_train: bool, quiet: bool = False) -> Tuple[float, float, float]:
        """
        Args:
            data: Data to run on; will be broken into minibatches.
            epoch_name: Name to use in CLI output.
            is_train: Flag indicating if we should training ops (updating weights) as well.
            quiet: Flag indicating that we should print only few lines (useful if not run in interactive shell)

        Returns:
            Triple of epoch loss (average over samples), MRR (average over batches), total time used for epoch (in s)
        """
        """Run the training ops and return the loss and the MRR."""
        epoch_loss, loss = 0.0, 0.0
        mrr_sum, mrr = 0.0, 0.0
        epoch_start = time.time()
        data_generator = self.__split_data_into_minibatches(data, is_train=is_train, compute_language_weightings=True)
        samples_used_so_far = 0
        printed_one_line = False
        for minibatch_counter, (batch_data_dict, samples_in_batch, samples_used_so_far, _) in enumerate(data_generator):
            if not quiet or (minibatch_counter % 100) == 99:
                print("%s: Batch %5i (has %i samples). Processed %i samples. Loss so far: %.4f.  MRR so far: %.4f "
                      % (epoch_name, minibatch_counter, samples_in_batch,
                         samples_used_so_far - samples_in_batch, loss, mrr),
                      flush=True,
                      end="\r" if not quiet else '\n')
                printed_one_line = True
            ops_to_run = {'loss': self.__ops['loss'], 'mrr': self.__ops['mrr']}
            if is_train:
                ops_to_run['train_step'] = self.__ops['train_step']
            op_results = self.__sess.run(ops_to_run, feed_dict=batch_data_dict)
            assert not np.isnan(op_results['loss'])

            epoch_loss += op_results['loss'] * samples_in_batch
            mrr_sum += np.sum(op_results['mrr'])

            loss = epoch_loss / max(1, samples_used_so_far)
            mrr = mrr_sum / max(1, samples_used_so_far)

            # additional training logs
            if (minibatch_counter % 100) == 0 and is_train:
                wandb.log({'train-loss': op_results['loss'],
                           'train-mrr': op_results['mrr']})

            minibatch_counter += 1

        used_time = time.time() - epoch_start
        if printed_one_line:
            print("\r\x1b[K", end='')
        self.train_log("  Epoch %s took %.2fs [processed %s samples/second]"
                       % (epoch_name, used_time, int(samples_used_so_far/used_time)))

        return loss, mrr, used_time

    @property
    def model_save_path(self) -> str:
        return os.path.join(self.__model_save_dir,
                            f'{self.run_name}_model_best.pkl.gz')

    def train(self,
              train_data: LoadedSamples,
              valid_data: LoadedSamples,
              azure_info_path: Optional[str],
              quiet: bool = False,
              resume: bool = False) -> RichPath:
        model_path = RichPath.create(self.model_save_path, azure_info_path)
        with self.__sess.as_default():
            tf.set_random_seed(self.hyperparameters['seed'])
            train_data_per_lang_nums = {language: len(samples) for language, samples in train_data.items()}
            print('Training on %s samples.' % (", ".join("%i %s" % (num, lang) for (lang, num) in train_data_per_lang_nums.items())))
            valid_data_per_lang_nums = {language: len(samples) for language, samples in valid_data.items()}
            print('Validating on %s samples.' % (", ".join("%i %s" % (num, lang) for (lang, num) in valid_data_per_lang_nums.items())))

            if resume:
                # Variables should have been restored.
                best_val_mrr_loss, best_val_mrr, _ = self.__run_epoch_in_batches(valid_data, "RESUME (valid)", is_train=False, quiet=quiet)
                self.train_log('Validation Loss on Resume: %.6f' % (best_val_mrr_loss,))
            else:
                init_op = tf.variables_initializer(self.__sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
                self.__sess.run(init_op)
                self.save(model_path)
                best_val_mrr = 0
            no_improvement_counter = 0
            epoch_number = 0
            while (epoch_number < self.hyperparameters['max_epochs']
                   and no_improvement_counter < self.hyperparameters['patience']):

                self.train_log('==== Epoch %i ====' % (epoch_number,))

                # run training loop and log metrics
                train_loss, train_mrr, train_time = self.__run_epoch_in_batches(train_data, "%i (train)" % (epoch_number,),
                                                                                is_train=True,
                                                                                quiet=quiet)
                self.train_log(' Training Loss: %.6f' % (train_loss,))

                # run validation calcs and log metrics
                val_loss, val_mrr, val_time = self.__run_epoch_in_batches(valid_data, "%i (valid)" % (epoch_number,),
                                                                          is_train=False,
                                                                          quiet=quiet)
                self.train_log(' Validation:  Loss: %.6f | MRR: %.6f' % (val_loss, val_mrr,))

                log = {'epoch': epoch_number,
                       'train-loss': train_loss,
                       'train-mrr': train_mrr,
                       'train-time-sec': train_time,
                       'val-loss': val_loss,
                       'val-mrr': val_mrr,
                       'val-time-sec': val_time}
                
                # log to wandb
                wandb.log(log)
            
                # log to tensorboard
                for key in log:
                    if key != 'epoch':
                        self._log_tensorboard_scalar(tag=key, 
                                                     value=log[key],
                                                     step=epoch_number)

                #  log the final epoch number
                wandb.run.summary['epoch'] = epoch_number

                if val_mrr > best_val_mrr:
                    # save the best val_mrr encountered
                    best_val_mrr_loss, best_val_mrr = val_loss, val_mrr

                    wandb.run.summary['best_val_mrr_loss'] = best_val_mrr_loss
                    wandb.run.summary['best_val_mrr'] = val_mrr
                    wandb.run.summary['best_epoch'] = epoch_number

                    no_improvement_counter = 0
                    self.save(model_path)
                    self.train_log("  Best result so far -- saved model as '%s'." % (model_path,))
                else:
                    # record epochs without improvement for early stopping
                    no_improvement_counter += 1
                epoch_number += 1

            log_path = os.path.join(self.__log_save_dir,
                                    f'{self.run_name}.train_log')
            wandb.save(log_path)
            tf.io.write_graph(self.__sess.graph,
                              logdir=wandb.run.dir,
                              name=f'{self.run_name}-graph.pbtxt')

        self.__summary_writer.close()
        return model_path

    def __compute_representations_batched(self,
                                          raw_data: List[Dict[str, Any]],
                                          data_loader_fn: Callable[[Dict[str, Any], Dict[str, Any]], bool],
                                          model_representation_op: tf.Tensor,
                                          representation_type: RepresentationType) -> List[Optional[np.ndarray]]:
        """Return a list of vector representation of each datapoint or None if the representation for that datapoint
        cannot be computed.

        Args:
            raw_data: a list of raw data point as dictionanries.
            data_loader_fn: A function f(in, out) that attempts to load/preprocess the necessary data from
             in and store it in out, returning a boolean success value. If it returns False, the sample is
             skipped and no representation is computed.
            model_representation_op: An op in the computation graph that represents the desired
             representations.
            representation_type: type of the representation we are interested in (either code or query)

        Returns:
             A list of either a 1D numpy array of the representation of the i-th element in data or None if a
             representation could not be computed.
        """
        tensorized_data = defaultdict(list)  # type: Dict[str, List[Dict[str, Any]]]
        sample_to_tensorised_data_id = []  # type: List[Optional[SampleId]]
        for raw_sample in raw_data:
            language = raw_sample['language']
            if language.startswith('python'):
                language = 'python'
            sample: Dict = {}
            valid_example = data_loader_fn(raw_sample, sample)
            if valid_example:
                sample_to_tensorised_data_id.append((language, len(tensorized_data[language])))
                tensorized_data[language].append(sample)
            else:
                sample_to_tensorised_data_id.append(None)
        assert len(sample_to_tensorised_data_id) == len(raw_data)

        data_generator = self.__split_data_into_minibatches(tensorized_data,
                                                            is_train=False,
                                                            include_query=representation_type == RepresentationType.QUERY,
                                                            include_code=representation_type == RepresentationType.CODE,
                                                            drop_incomplete_final_minibatch=False)

        computed_representations = []
        original_tensorised_data_ids = []  # type: List[SampleId]
        for minibatch_counter, (batch_data_dict, samples_in_batch, samples_used_so_far, batch_original_tensorised_data_ids) in enumerate(data_generator):
            op_results = self.__sess.run(model_representation_op, feed_dict=batch_data_dict)
            computed_representations.append(op_results)
            original_tensorised_data_ids.extend(batch_original_tensorised_data_ids)

        computed_representations = np.concatenate(computed_representations, axis=0)
        tensorised_data_id_to_representation_idx = {tensorised_data_id: repr_idx
                                                    for (repr_idx, tensorised_data_id) in enumerate(original_tensorised_data_ids)}
        reordered_representations: List = []
        for tensorised_data_id in sample_to_tensorised_data_id:
            if tensorised_data_id is None:
                reordered_representations.append(None)
            else:
                reordered_representations.append(computed_representations[tensorised_data_id_to_representation_idx[tensorised_data_id]])
        return reordered_representations

    def get_query_representations(self, query_data: List[Dict[str, Any]]) -> List[Optional[np.ndarray]]:
        def query_data_loader(sample_to_parse, result_holder):
            function_name = sample_to_parse.get('func_name')
            return self.query_encoder_type.load_data_from_sample(
                "query",
                self.hyperparameters,
                self.__query_metadata,
                [d.lower() for d in sample_to_parse['docstring_tokens']],
                function_name,
                result_holder=result_holder,
                is_test=True)

        return self.__compute_representations_batched(query_data,
                                                      data_loader_fn=query_data_loader,
                                                      model_representation_op=self.__ops['query_representations'],
                                                      representation_type=RepresentationType.QUERY)

    def get_code_representations(self, code_data: List[Dict[str, Any]]) -> List[Optional[np.ndarray]]:
        def code_data_loader(sample_to_parse, result_holder):
            code_tokens = sample_to_parse['code_tokens']
            language = sample_to_parse['language']
            if language.startswith('python'):
                language = 'python'

            if code_tokens is not None:
                function_name = sample_to_parse.get('func_name')
                return self.code_encoder_type(language).load_data_from_sample(
                    "code",
                    self.hyperparameters,
                    self.__per_code_language_metadata[language],
                    code_tokens,
                    function_name,
                    result_holder=result_holder,
                    is_test=True)
            else:
                return False

        return self.__compute_representations_batched(code_data,
                                                      data_loader_fn=code_data_loader,
                                                      model_representation_op=self.__ops['code_representations'],
                                                      representation_type=RepresentationType.CODE)
