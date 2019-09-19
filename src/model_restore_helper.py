from typing import Dict, Any, Optional, Type

import tensorflow as tf
from dpu_utils.utils import RichPath

from models import Model, NeuralBoWModel, RNNModel, SelfAttentionModel, ConvolutionalModel, ConvSelfAttentionModel


def get_model_class_from_name(model_name: str) -> Type[Model]:
    model_name = model_name.lower()
    if model_name in ['neuralbow', 'neuralbowmodel']:
        return NeuralBoWModel
    elif model_name in ['rnn', 'rnnmodel']:
        return RNNModel
    elif model_name in {'selfatt', 'selfattention', 'selfattentionmodel'}:
        return SelfAttentionModel
    elif model_name in {'1dcnn', 'convolutionalmodel'}:
        return ConvolutionalModel
    elif model_name in {'convselfatt', 'convselfattentionmodel'}:
        return ConvSelfAttentionModel
    else:
        raise Exception("Unknown model '%s'!" % model_name)


def restore(path: RichPath, is_train: bool, hyper_overrides: Optional[Dict[str, Any]]=None) -> Model:
    saved_data = path.read_as_pickle()

    if hyper_overrides is not None:
        saved_data['hyperparameters'].update(hyper_overrides)

    model_class = get_model_class_from_name(saved_data['model_type'])
    model = model_class(saved_data['hyperparameters'], saved_data.get('run_name'))
    model.query_metadata.update(saved_data['query_metadata'])
    for (language, language_metadata) in saved_data['per_code_language_metadata'].items():
        model.per_code_language_metadata[language] = language_metadata
    model.make_model(is_train=is_train)

    variables_to_initialize = []
    with model.sess.graph.as_default():
        with tf.name_scope("restore"):
            restore_ops = []
            used_vars = set()
            for variable in sorted(model.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), key=lambda v: v.name):
                used_vars.add(variable.name)
                if variable.name in saved_data['weights']:
                    # print('Initializing %s from saved value.' % variable.name)
                    restore_ops.append(variable.assign(saved_data['weights'][variable.name]))
                else:
                    print('Freshly initializing %s since no saved value was found.' % variable.name)
                    variables_to_initialize.append(variable)
            for var_name in sorted(saved_data['weights']):
                if var_name not in used_vars:
                    if var_name.endswith('Adam:0') or var_name.endswith('Adam_1:0') or var_name in ['beta1_power:0', 'beta2_power:0']:
                        continue
                    print('Saved weights for %s not used by model.' % var_name)
            restore_ops.append(tf.variables_initializer(variables_to_initialize))
            model.sess.run(restore_ops)
    return model
