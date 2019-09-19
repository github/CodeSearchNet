from typing import Any, Dict, Optional

from encoders import ConvSelfAttentionEncoder
from .model import Model


class ConvSelfAttentionModel(Model):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        hypers = {}
        for label in ["code", "query"]:
            hypers.update({f'{label}_{key}': value
                           for key, value in ConvSelfAttentionEncoder.get_default_hyperparameters().items()})
        model_hypers = {
            'learning_rate': 5e-4,
            'code_use_subtokens': False,
            'code_mark_subtoken_end': False,
            'batch_size': 650,
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
            code_encoder_type=ConvSelfAttentionEncoder,
            query_encoder_type=ConvSelfAttentionEncoder,
            run_name=run_name,
            model_save_dir=model_save_dir,
            log_save_dir=log_save_dir)
