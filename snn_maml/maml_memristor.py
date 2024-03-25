import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from snn_maml.utils import quantize_parameters
from collections import OrderedDict
from .import plasticity_rules
from .utils import tensors_to_device, compute_accuracy
from .maml import ModelAgnosticMetaLearning

__all__ = ['ModelAgnosticMetaLearning', 'MAML', 'FOMAML']

from tensorboardX import SummaryWriter

import pdb

# default `log_dir` is "runs" - we'll be more specific here

def batch_one_hot(targets, num_classes=10):
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


class ModelAgnosticMetaLearningCustomRule(ModelAgnosticMetaLearning):
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
    def __init__(self, model, optimizer=None, step_size=0.1, first_order=False,
                 learn_step_size=False, per_param_step_size=False,
                 num_adaptation_steps=1, scheduler=None,
                 loss_function=F.cross_entropy,
                 device_model = None,
                 device=None, 
                 boil=False,
                 baseline = False,
                 outer_loop_quantizer = None,
                 inner_loop_quantizer = None):
        
        self.model = model.to(device=device)
        self.outer_loop_quantizer = outer_loop_quantizer 
        self.inner_loop_quantizer = inner_loop_quantizer 
        self.optimizer = optimizer
        self.step_size = step_size
        self.first_order = first_order
        self.num_adaptation_steps = num_adaptation_steps
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.device = device
        self.device_model = device_model
        self.baseline = baseline

        if per_param_step_size or boil:
            self.step_size = OrderedDict((name, torch.tensor(step_size, dtype=param.dtype, device=self.device, requires_grad=learn_step_size)) for (name, param) in model.meta_named_parameters())
            if boil:
                assert learn_step_size is False, 'boil is not compatible with learning step sizes'
                last_layer_names = [k for k in self.step_size.keys()][-2:]#assumed bias and weight in last layer
                for k in last_layer_names:
                    self.step_size[k] = torch.tensor(0., dtype=self.step_size[k].dtype, device=self.device, requires_grad=False)
                print('step_size', self.step_size)
        else:
            self.step_size = torch.tensor(step_size, dtype=torch.float32,
                device=self.device, requires_grad=learn_step_size)

        if (self.optimizer is not None) and learn_step_size:
            self.optimizer.add_param_group({'params': self.step_size.values() if per_param_step_size else [self.step_size]})
            if scheduler is not None:
                for group in self.optimizer.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                #self.scheduler.base_lrs([group['initial_lr'] for group in self.optimizer.param_groups])


    
    #Inner loop
    def adapt(self, inputs, targets, is_classification_task=None,
              num_adaptation_steps=1, step_size=0.1, first_order=False):
        if is_classification_task is None:
            is_classification_task = (not targets.dtype.is_floating_point)
            
        params = OrderedDict(self.model.meta_named_parameters())
        if self.outer_loop_quantizer is not None:
            params = quantize_parameters(params, self.outer_loop_quantizer)

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
            
            if self.device_model is not None and not self.baseline:
                params = plasticity_rules.custom_sgd(self.model,
                                                inner_loss,
                                                step_size=step_size,
                                                params=params,
                                                first_order=(not self.model.training) or first_order,
                                                custom_update_fn = self.device_model.cond_update)
            else:
                params = plasticity_rules.custom_sgd(self.model,
                                inner_loss,
                                step_size=step_size,
                                params=params,
                                first_order=(not self.model.training) or first_order,
                                custom_update_fn = None)
            
            if self.inner_loop_quantizer is not None:
                params = quantize_parameters(params, self.inner_loop_quantizer)
            
            #print("updated parameters")
        return params, results

    #Outer loop
    def train_iter(self, dataloader, max_batches=500, epoch=-1):
        if self.optimizer is None:
            raise RuntimeError('Trying to call `train_iter`, while the '
                'optimizer is `None`. In order to train `{0}`, you must '
                'specify a Pytorch optimizer as the argument of `{0}` '
                '(eg. `{0}(model, optimizer=torch.optim.SGD(model.'
                'parameters(), lr=0.01), ...).'.format(__class__.__name__))
        num_batches = 0
        self.model.train()
        
        #print(self.model)
        
        while num_batches < max_batches:
            for batch in dataloader:
                if num_batches >= max_batches:
                    break



                self.optimizer.zero_grad()

                batch = tensors_to_device(batch, device=self.device)
                outer_loss, results = self.get_outer_loss(batch)    
                yield results
                outer_loss.backward()
                if self.device_model is not None:
                    self.device_model.soft_clamp(self.model)#baseline keep soft clamp


                self.optimizer.step()
                if hasattr(self.step_size, '__len__'):
                    if len(self.step_size.shape)>0:
                        for name, value in self.step_size.items():
                            if value.data<0:
                                value.data.zero_()
                                print('Negative step values detected')
                                
                #if self.custom_outer_update_fn is not None:
                #    pass
                ##hard clamp here

                if self.scheduler is not None:
                    self.scheduler.step()

                num_batches += 1

