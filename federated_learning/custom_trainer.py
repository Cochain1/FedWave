import torch
from transformers import Trainer
import types
from typing import Dict, List, Optional, Tuple, Union
import builtins
from transformers import Trainer
class ValueChainTrainer(Trainer):
    def __init__(self, full_model=None, **kwargs):
        self.full_model = full_model

        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self.full_model.train()
        outputs = self.full_model(**inputs)

        if return_outputs:
            return (outputs.loss, outputs) if outputs.loss is not None else (None, outputs)

        return outputs.loss