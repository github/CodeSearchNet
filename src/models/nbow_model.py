from typing import Any, Dict, Optional, Type

from encoders import NBoWEncoder, Encoder
from .model import Model


class NeuralBoWModel(Model):
    query_encoder_type = NBoWEncoder

    @classmethod
    def code_encoder_type(cls, language: str) -> Type[Encoder]:
        return NBoWEncoder

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        hypers = {}
        for label in ["code", "query"]:
            hypers.update({f'{label}_{key}': value
                           for key, value in NBoWEncoder.get_default_hyperparameters().items()})
        model_hypers = {
            'code_use_subtokens': False,
            'code_mark_subtoken_end': False,
            'loss': 'cosine',
            'batch_size': 1000
        }
        hypers.update(super().get_default_hyperparameters())
        hypers.update(model_hypers)
        return hypers

    def __init__(self,
                 hyperparameters: Dict[str, Any],
                 run_name: str = None,
                 model_save_dir: Optional[str] = None,
                 log_save_dir: Optional[str] = None):
        super().__init__(
            hyperparameters,
            run_name=run_name,
            model_save_dir=model_save_dir,
            log_save_dir=log_save_dir)
