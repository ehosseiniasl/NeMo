from typing import Callable, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from prettytable import PrettyTable
from pytorch_lightning.callbacks import Callback

from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.utils import logging


def collect_precision(tensor: torch.Tensor) -> Dict[str, str]:
    return {"Precision": str(tensor.dtype)}


def collect_precision_and_shape(tensor: torch.Tensor) -> Dict[str, str]:
    return {"Shape": str(tensor.shape), "Precision": str(tensor.dtype)}


class ParameterDebugger(Callback):
    """
    Debugging tool to help inspect parameters and gradients at any callback event.

    Since accessing the gradient tensors with MegatronCore optimizers is less intuitive than pure PyTorch,
    this callback handles the boilerplate needed to iterate over the model parameters and gradients,
    and applies user specified functions to them. These functions can be used to log attributes or
    apply asserts on the param and grad tensors. Attributes are logged in a table, with a row for each parameter name.
    Default behavior is to log the precision and shapes of each parameter and its gradient.

    Args:
        param_attr_fn: Function to apply to model parameters. Can be used to apply assertions on the tensor,
            or return a mapping of labels and values to log for each parameter.
        grad_attr_fn: Function to apply to model gradients.
        log_on_hooks: PTL callback hook name or list of hook names on which to apply param_attr_fn and grad_attr_fn.
            See `PTL docs <https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html#hooks>`_ for more info
            on callback hooks. Note that some hooks that occur before the model is constructed are invalid.

    Example:
        >>> fn = lambda x: {"Norm": str(x.norm(2).item())}
        >>> callback = ParameterDebugger(param_attr_fn=fn, log_on_hooks=["on_train_start", "on_train_end"])
        >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        param_attr_fn: Optional[Callable[[torch.Tensor], Dict[str, str]]] = collect_precision_and_shape,
        grad_attr_fn: Optional[Callable[[torch.Tensor], Dict[str, str]]] = collect_precision,
        log_on_hooks: Union[List[str], str] = "on_train_start",
    ):
        self.param_attr_fn = param_attr_fn
        self.grad_attr_fn = grad_attr_fn

        valid_hooks = set(
            [
                "teardown",
                "on_fit_end",
                "on_sanity_check_start",
                "on_sanity_check_end",
                "on_train_batch_start",
                "on_train_batch_end",
                "on_train_epoch_start",
                "on_train_epoch_end",
                "on_validation_epoch_start",
                "on_validation_epoch_end",
                "on_test_epoch_start",
                "on_test_epoch_end",
                "on_predict_epoch_start",
                "on_predict_epoch_end",
                "on_validation_batch_start",
                "on_validation_batch_end",
                "on_test_batch_start",
                "on_test_batch_end",
                "on_predict_batch_start",
                "on_predict_batch_end",
                "on_train_start",
                "on_train_end",
                "on_validation_start",
                "on_validation_end",
                "on_test_start",
                "on_test_end",
                "on_predict_start",
                "on_predict_end",
                "on_exception",
                "on_save_checkpoint",
                "on_load_checkpoint",
                "on_before_backward",
                "on_after_backward",
                "on_before_optimizer_step",
                "on_before_zero_grad",
            ]
        )

        if isinstance(log_on_hooks, str):
            log_on_hooks = [log_on_hooks]
        for hook_name in log_on_hooks:
            assert (
                hook_name in valid_hooks
            ), f"Hook {hook_name} supplied to log_on_hooks is not valid or can not be used. Valid hooks are {valid_hooks}"
            setattr(self, hook_name, self._log_param_and_grad_attrs)

    def _log_param_and_grad_attrs(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Iterate over model parameters, find gradient tensor, apply and collect outputs of
        param_attr_fn and grad_attr_fn, and log outputs in a table.
        """

        def find_grad_tensor(param: torch.Tensor) -> Optional[torch.Tensor]:
            """If using MCore optimizer, search the grad buckets for param's grad tensor."""
            if not isinstance(pl_module.optim, MegatronOptimizerModule):
                return param.grad

            for buf in pl_module.buffers:
                if param in buf.param_to_bucket:
                    return buf.param_to_bucket[param].grad_data

            return None

        # create table and get table column headers
        debug_table = PrettyTable()
        debug_table.align = "l"
        param_keys = self.param_attr_fn(torch.zeros(0)).keys() if self.param_attr_fn else []
        grad_keys = self.grad_attr_fn(torch.zeros(0)).keys() if self.grad_attr_fn else []
        debug_table.field_names = ["Parameter"] + ["Param " + k for k in param_keys] + ["Grad " + k for k in grad_keys]

        for param_name, param_tensor in pl_module.named_parameters():
            short_name = param_name.replace("module.", "").replace(".weight", "")
            grad_tensor = find_grad_tensor(param_tensor)

            row = [short_name]
            for tensor, attr_fn, attr_keys in zip(
                [param_tensor, grad_tensor], [self.param_attr_fn, self.grad_attr_fn], [param_keys, grad_keys]
            ):
                if attr_fn is not None:
                    if tensor is not None:
                        # iterate instead of just appending attrs.values() to ensure order
                        attrs = attr_fn(tensor)
                        row += [attrs[k] for k in attr_keys]
                    else:
                        row += [None for k in attr_keys]

            debug_table.add_row(row)

        logging.info("\n" + debug_table.get_string())
