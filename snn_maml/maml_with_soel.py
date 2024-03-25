import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pickle

from collections import OrderedDict
from . import plasticity_rules
from .utils import tensors_to_device, compute_accuracy
from .maml import ModelAgnosticMetaLearning

__all__ = ['ModelAgnosticMetaLearning', 'MAML', 'FOMAML']

from tensorboardX import SummaryWriter

import pdb
import os

# default `log_dir` is "runs" - we'll be more specific here

def batch_one_hot(targets, num_classes=5):
    one_hot = torch.zeros((targets.shape[0],num_classes))
    #print("targets shape", targets.shape)
    for i in range(targets.shape[0]):
        one_hot[i][targets[i]] = 1
        
    return one_hot

def undo_onehot(targets):
    not_hot = torch.zeros((targets.shape[0]))
    
    for i in range(targets.shape[0]):
        not_hot[i] = torch.nonzero(targets[0])[0][0].item()
        
    return not_hot.to(targets.device)


class ModelAgnosticMetaLearning_With_SOEL(ModelAgnosticMetaLearning):
    """Meta-learner class for Model-Agnostic Meta-Learning [1].

    Parameters
    ----------
    model : `torchmeta.modules.MetaModule` instance
        The model.

    optimizer : `torch.optim.Optimizer` instance, optional
        The optimizer for the outer-loop optimization procedure. This argument
        is optional for evaluation.

    step_size : float (default: 0.1)
        The step size of the gradient descent update for fast adaptation
        (inner-loop update).

    first_order : bool (default: False)
        If `True`, then the first-order approximation of MAML is used.

    learn_step_size : bool (default: False)
        If `True`, then the step size is a learnable (meta-trained) additional
        argument [2].

    per_param_step_size : bool (default: False)
        If `True`, then the step size parameter is different for each parameter
        of the model. Has no impact unless `learn_step_size=True`.

    num_adaptation_steps : int (default: 1)
        The number of gradient descent updates on the loss function (over the
        training dataset) to be used for the fast adaptation on a new task.

    scheduler : object in `torch.optim.lr_scheduler`, optional
        Scheduler for the outer-loop optimization [3].

    loss_function : callable (default: `torch.nn.functional.cross_entropy`)
        The loss function for both the inner and outer-loop optimization.
        Usually `torch.nn.functional.cross_entropy` for a classification
        problem, of `torch.nn.functional.mse_loss` for a regression problem.

    device : `torch.device` instance, optional
        The device on which the model is defined.

    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)

    .. [2] Li Z., Zhou F., Chen F., Li H. (2017). Meta-SGD: Learning to Learn
           Quickly for Few-Shot Learning. (https://arxiv.org/abs/1707.09835)

    .. [3] Antoniou A., Edwards H., Storkey A. (2018). How to train your MAML.
           International Conference on Learning Representations (ICLR).
           (https://arxiv.org/abs/1810.09502)
    """
    def __init__(self,
            model, 
            optimizer=None,
            threshold=.05,
            step_size=0.1,
            first_order=False,
            learn_step_size=False,
            learn_threshold=False,
            per_param_step_size=False,
            num_adaptation_steps=1, 
            scheduler=None,
            loss_function=F.cross_entropy,
            device=None, 
            boil=False,
            outer_loop_quantizer = None,
            inner_loop_quantizer = None):

        self.threshold = torch.tensor([threshold], requires_grad=True, dtype=torch.float).to(model.get_input_layer_device())

        super(ModelAgnosticMetaLearning_With_SOEL, self).__init__(
            model=model, 
            optimizer=optimizer,
            step_size=step_size,
            first_order=first_order,
            learn_step_size=learn_step_size,
            per_param_step_size=per_param_step_size,
            num_adaptation_steps=num_adaptation_steps,
            scheduler=scheduler,
            loss_function=loss_function,
            custom_outer_update_fn = None, custom_inner_update_fn = None,
            device=device, 
            boil=boil,
            outer_loop_quantizer = outer_loop_quantizer,
            inner_loop_quantizer = inner_loop_quantizer)
        
   
    #Inner loop
    def adapt(self, inputs, targets, task_id=0, is_classification_task=None,
              num_adaptation_steps=1, step_size=0.1, first_order=False):        
        
        if is_classification_task is None:
            is_classification_task = (not targets.dtype.is_floating_point)
        params = None

        results = {'inner_losses': np.zeros(
            (num_adaptation_steps,), dtype=np.float32)}

        for step in range(num_adaptation_steps):
            
            logits = self.model(inputs, params=params)
            
            inner_loss = self.loss_function(logits, targets)
            results['inner_losses'][step] = inner_loss.item()
            #pdb.set_trace()
            if (step == num_adaptation_steps-1) and is_classification_task: 
                results['accuracy_before'] = compute_accuracy(logits, targets)

            #print("updating params...")
            self.model.zero_grad()
            
            params = plasticity_rules.maml_soel(self.model,
                                                logits,
                                                batch_one_hot(targets).to(self.device),
                                                step_size=step_size,
                                                params=params,
                                                first_order=(not self.model.training) or first_order,
                                                threshold = self.threshold)
            
            
            #print("updated parameters")
        return params, results
