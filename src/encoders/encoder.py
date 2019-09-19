from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple

import tensorflow as tf


class QueryType(Enum):
    DOCSTRING = 'docstring_as_query'
    FUNCTION_NAME = 'func_name_as_query'


class Encoder(ABC):
    @classmethod
    @abstractmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        """
        Returns:
             Default set of hyperparameters for encoder.
             Note that at use, the hyperparameters names will be prefixed with '${label}_' for the
             chosen encoder label.
        """
        return {}

    def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
        """
        Args:
            label: Label for the encoder, used in names of hyperparameters.
            hyperparameters: Hyperparameters used.
            metadata: Dictionary with metadata (e.g., vocabularies) used by this encoder.
        """
        self.__label = label
        self.__hyperparameters = hyperparameters
        self.__metadata = metadata
        self.__placeholders = {}

    @property
    def label(self):
        return self.__label

    @property
    def hyperparameters(self):
        return self.__hyperparameters

    @property
    def metadata(self):
        return self.__metadata

    @property
    def placeholders(self):
        return self.__placeholders

    @property
    @abstractmethod
    def output_representation_size(self) -> int:
        raise Exception("Encoder.output_representation_size not implemented!")

    def get_hyper(self, hyper_name: str) -> Any:
        """
        Retrieve hyper parameter, prefixing the given name with the label of the encoder.

        Args:
            hyper_name: Some hyperparameter name.

        Returns:
            self.hyperparameters['%s_%s' % (self.label, hyper_name)]
        """
        return self.hyperparameters['%s_%s' % (self.label, hyper_name)]

    def _make_placeholders(self):
        """
        Creates placeholders for encoders.
        """
        self.__placeholders['dropout_keep_rate'] = \
            tf.placeholder(tf.float32,
                           shape=(),
                           name='dropout_keep_rate')

    @abstractmethod
    def make_model(self, is_train: bool=False) -> tf.Tensor:
        """
        Create the actual encoder model, including necessary placeholders and parameters.

        Args:
            is_train: Bool flag indicating if the model is used for training or inference.

        Returns:
            A tensor encoding the passed data.
        """
        pass

    @classmethod
    @abstractmethod
    def init_metadata(cls) -> Dict[str, Any]:
        """
        Called to initialise the metadata before looking at actual data (i.e., set up Counters, lists, sets, ...)

        Returns:
            A dictionary that will be used to collect the raw metadata (token counts, ...).
        """
        return {}

    @classmethod
    @abstractmethod
    def load_metadata_from_sample(cls, data_to_load: Any, raw_metadata: Dict[str, Any],
                                  use_subtokens: bool=False, mark_subtoken_end: bool=False) -> None:
        """
        Called to load metadata from a single sample.

        Args:
            data_to_load: Raw data to load; type depens on encoder. Usually comes from a data parser such as
             tokenize_python_from_string or tokenize_docstring_from_string.
            raw_metadata: A dictionary that will be used to collect the raw metadata (token counts, ...).
            use_subtokens: subtokenize identifiers
            mark_subtoken_end: add a special marker for subtoken ends. Used only if use_subtokens=True
        """
        pass

    @classmethod
    @abstractmethod
    def finalise_metadata(cls, encoder_label: str, hyperparameters: Dict[str, Any], raw_metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Called to finalise the metadata after looking at actual data (i.e., compute vocabularies, ...)

        Args:
            encoder_label: Label used for this encoder.
            hyperparameters: Hyperparameters used.
            raw_metadata_list: List of dictionaries used to collect the raw metadata (token counts, ...) (one per file).

        Returns:
            Finalised metadata (vocabs, ...).
        """
        return {}

    @classmethod
    @abstractmethod
    def load_data_from_sample(cls,
                              encoder_label: str,
                              hyperparameters: Dict[str, Any],
                              metadata: Dict[str, Any],
                              data_to_load: Any,
                              function_name: Optional[str],
                              result_holder: Dict[str, Any],
                              is_test: bool=True) -> bool:
        """
        Called to convert a raw sample into the internal format, allowing for preprocessing.
        Result will eventually be fed again into the split_data_into_minibatches pipeline.

        Args:
            encoder_label: Label used for this encoder.
            hyperparameters: Hyperparameters used to load data.
            metadata: Computed metadata (e.g. vocabularies).
            data_to_load: Raw data to load; type depens on encoder. Usually comes from a data parser such as
             tokenize_python_from_string or tokenize_docstring_from_string.
             function_name: The name of the function.
            result_holder: Dictionary used to hold the prepared data.
            is_test: Flag marking if we are handling training data or not.

        Returns:
            Flag indicating if the example should be used (True) or dropped (False)
        """
        return True

    @abstractmethod
    def init_minibatch(self, batch_data: Dict[str, Any]) -> None:
        """
        Initialise a minibatch that will be constructed.

        Args:
            batch_data: The minibatch data.
        """
        pass

    @abstractmethod
    def extend_minibatch_by_sample(self, batch_data: Dict[str, Any], sample: Dict[str, Any], is_train: bool=False,
                                   query_type: QueryType=QueryType.DOCSTRING.value) -> bool:
        """
        Extend a minibatch under construction by one sample. This is where the data may be randomly perturbed in each
        epoch for data augmentation.

        Args:
            batch_data: The minibatch data.
            sample: The sample to add.
            is_train: Flag indicating if we are in train mode (which causes data augmentation)
            query_type: Indicates what should be used as the query, the docstring or the function name.

        Returns:
            True iff the minibatch is full after this sample.
        """
        return True

    @abstractmethod
    def minibatch_to_feed_dict(self, batch_data: Dict[str, Any], feed_dict: Dict[tf.Tensor, Any], is_train: bool) -> None:
        """
        Take a collected minibatch and add it to a feed dict that can be fed directly to the constructed model.

        Args:
            batch_data: The minibatch data.
            feed_dict: The feed dictionary that we will send to tensorflow.
            is_train: Flag indicating if we are in training mode.
        """
        feed_dict[self.placeholders['dropout_keep_rate']] = self.hyperparameters['dropout_keep_rate'] if is_train else 1.0

    @abstractmethod
    def get_token_embeddings(self) -> Tuple[tf.Tensor, List[str]]:
        """Returns the tensorflow embeddings tensor (VxD) along with a list (of size V) of the names of the
        embedded elements."""
        pass
