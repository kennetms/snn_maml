import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from collections import OrderedDict
from . import plasticity_rules
from .utils import (tensors_to_device,
                    compute_accuracy,
                    hessian_vector_product,
                    matrix_evaluator,
                    cg_solve)

__all__ = ['ImplicitModelAgnosticMetaLearning', 'iMAML']

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


class ImplicitModelAgnosticMetaLearning(object):
    """Implicit Meta-learner class for Model-Agnostic Meta-Learning [1].

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
    .. [1] Rajeswaran, Aravind, et al. "Meta-learning with implicit gradients." (2019).
    """
    def __init__(self, model, optimizer=None, step_size=0.1, lamda=.1,
                 learn_step_size=False, per_param_step_size=False,
                 num_adaptation_steps=1, scheduler=None,
                 loss_function=F.cross_entropy, custom_outer_update_fn = None, custom_inner_update_fn = None, device=None):
        self.model = model.to(device=device)
        self.optimizer = optimizer
        self.step_size = step_size
        self.num_adaptation_steps = num_adaptation_steps
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.device = device
        self.custom_inner_update_fn = custom_inner_update_fn
        self.custom_outer_update_fn = custom_outer_update_fn
        self.lamda = lamda

        if per_param_step_size:
            self.step_size = OrderedDict((name, torch.tensor(step_size, dtype=param.dtype, device=self.device, requires_grad=learn_step_size)) for (name, param) in model.meta_named_parameters())

        else:
            self.step_size = torch.tensor(step_size, dtype=torch.float32,
                device=self.device, requires_grad=learn_step_size)

        if (self.optimizer is not None) and learn_step_size:
            self.optimizer.add_param_group({'params': self.step_size.values() if per_param_step_size else [self.step_size]})
            if scheduler is not None:
                for group in self.optimizer.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                #self.scheduler.base_lrs([group['initial_lr'] for group in self.optimizer.param_groups])

    def get_outer_loss(self, batch, outer_train=True):
        if 'test' not in batch:
            raise RuntimeError('The batch does not contain any test dataset.')

        _, test_targets = batch['test']
        num_tasks = test_targets.size(0)
        is_classification_task = (not test_targets.dtype.is_floating_point)
        results = {
            'num_tasks': num_tasks,
            'inner_losses': np.zeros((self.num_adaptation_steps,
                num_tasks), dtype=np.float32),
            'outer_losses': np.zeros((num_tasks,), dtype=np.float32),
            'mean_outer_loss': 0.
        }
        if is_classification_task:
            results.update({
                'accuracies_before': np.zeros((num_tasks,), dtype=np.float32),
                'accuracies_after': np.zeros((num_tasks,), dtype=np.float32)
            })

        meta_grad = 0.0
        mean_outer_loss = torch.tensor(0., device=self.device)
        # One task per batch
        for task_id, (train_inputs, train_targets, test_inputs, test_targets) \
                in enumerate(zip(*batch['train'], *batch['test'])):
            
            params, adaptation_results, tloss, grads = self.adapt(
                train_inputs, train_targets,
                is_classification_task=is_classification_task,
                num_adaptation_steps=self.num_adaptation_steps,
                step_size=self.step_size)

            results['inner_losses'][:, task_id] = adaptation_results['inner_losses']
            
            if is_classification_task:
                results['accuracies_before'][task_id] = adaptation_results['accuracy_before']

            with torch.set_grad_enabled(self.model.training):
                #f_params = OrderedDict(self.model.meta_named_parameters())
                test_logits = self.model(test_inputs, params=params)
                partial_outer_loss = self.loss_function(test_logits, test_targets)
                    
#                results['outer_losses'][task_id] = outer_loss.item()
#                mean_outer_loss += outer_loss
                if outer_train:
                    grads = torch.autograd.grad(partial_outer_loss, params.values()) #, create_graph=False, allow_unused=False)
                    flat_grad = torch.cat([g.contiguous().view(-1) for g in grads])

                    #Returns f(g) = (g+g/lambda H)
                    task_matrix_evaluator = matrix_evaluator(tloss, params.values(), regu_coef=1.0, lamda=self.lamda)  
                    task_outer_grad = cg_solve(task_matrix_evaluator, b=flat_grad, cg_iters=20, verbose=False, x_init=None)

                    meta_grad += task_outer_grad.detach()/num_tasks
                mean_outer_loss += partial_outer_loss.detach() #TODO: probably not the value we need here

            if is_classification_task:
                results['accuracies_after'][task_id] = compute_accuracy(test_logits, test_targets)


        mean_outer_loss.div_(num_tasks)
        results['mean_outer_loss'] = mean_outer_loss.item()

        return partial_outer_loss, meta_grad, results
    
    #Inner loop
    def adapt(self,
            inputs,
            targets,
            is_classification_task=None,
            num_adaptation_steps=1,
            step_size=0.1):

        if is_classification_task is None:
            is_classification_task = (not targets.dtype.is_floating_point)

        params = None

        results = {'inner_losses': np.zeros((num_adaptation_steps,), dtype=np.float32)}

        for step in range(num_adaptation_steps):
            
            logits = self.model(inputs, params=params)

            inner_loss = self.loss_function(logits, targets)
            results['inner_losses'][step] = inner_loss.item()
            #pdb.set_trace()
            if (step == 0) and is_classification_task: 
                results['accuracy_before'] = compute_accuracy(logits, targets)

            self.model.zero_grad()
            params, grads = plasticity_rules.custom_sgd_reg(self.model,
                                                inner_loss,
                                                step_size=step_size,
                                                params=params,
                                                anchor_params=self.model.meta_named_parameters(),
                                                lamda=self.lamda)

        logits = self.model(inputs, params=params)
        tloss = self.loss_function(logits, targets)

            
        return params, results, tloss, grads

    def train(self, dataloader, max_batches=500, verbose=True, epoch=-1, **kwargs):
        """
        Run outer (and inner) optimization steps
        """
        with tqdm(total=max_batches, disable=False, **kwargs) as pbar:
            for results in self.train_iter(dataloader, max_batches=max_batches, epoch=epoch):
                pbar.update(1)
                postfix = {'loss': '{0:.4f}'.format(results['mean_outer_loss'])}
                if 'accuracies_after' in results:
                    postfix['after'] = '{0:.4f}'.format(np.mean(results['accuracies_after']))
                if 'accuracies_before' in results:
                    postfix['before']  = '{0:.4f}'.format(np.mean(results['accuracies_before']))
                pbar.set_postfix(**postfix)

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
                outer_loss, outer_grad, results = self.get_outer_loss(batch)    
                yield results

                #Actual gradient calculation
                #outer_loss.backward()
                flat_grad=True

                check = 0

                for task_id, (train_inputs, train_targets, test_inputs, test_targets) in enumerate(zip(*batch['train'], *batch['test'])):
                    logits = self.model(train_inputs)
                    dummy_loss = self.loss_function(logits, train_targets)*0
                    break

                for p in self.model.parameters():
                    check = check + 1 if type(p.grad) == type(None) else check
                if check > 0:
                    # initialize the grad fields properly
                    dummy_loss.backward()

                offset = 0
                for p in self.model.parameters():
                    if p.requires_grad:
                        if p.nelement()==1 and len(p.size())==0:
                            this_grad = outer_grad[offset]
                            offset += 1
                        else:
                            this_grad = outer_grad[offset:offset + p.nelement()].view(p.size())
                            offset += p.nelement()
                        p.grad.copy_(this_grad)
                    else:
                        pass #print('unused', p.size())
                assert offset == len(outer_grad)

                self.optimizer.step()
                if hasattr(self.step_size, '__len__'):
                    if len(self.step_size.shape)>0:
                        for name, value in self.step_size.items():
                            if value.data<0:
                                value.data.zero_()
                                print('Negative step values detected')

                if self.scheduler is not None:
                    self.scheduler.step()

                num_batches += 1

    def evaluate(self, dataloader, max_batches=500, verbose=True, **kwargs):
        mean_outer_loss, mean_accuracy, count = 0., 0., 0
        with tqdm(total=max_batches, disable=False, **kwargs) as pbar:
            for results in self.evaluate_iter(dataloader, max_batches=max_batches):
                pbar.update(1)
                count += 1
                mean_outer_loss += (results['mean_outer_loss'] - mean_outer_loss) / count
                postfix = {'loss': '{0:.4f}'.format(mean_outer_loss)}
                if 'accuracies_after' in results:
                    mean_accuracy += (np.mean(results['accuracies_after'])
                        - mean_accuracy) / count
                    postfix['after in-loop'] = '{0:.4f}'.format(np.mean(mean_accuracy))
                if 'accuracies_before' in results:
                    postfix['before in-loop']  = '{0:.4f}'.format(np.mean(results['accuracies_before']))
                pbar.set_postfix(**postfix)

        mean_results = {'mean_outer_loss': mean_outer_loss}
        if 'accuracies_after' in results:
            mean_results['accuracies_after'] = mean_accuracy

        return mean_results

    def evaluate_iter(self, dataloader, max_batches=500):
        num_batches = 0
        self.model.eval()
        while num_batches < max_batches:
            for batch in dataloader:
                if num_batches >= max_batches:
                    break

                batch = tensors_to_device(batch, device=self.device)
                outer_loss, outer_grad, results = self.get_outer_loss(batch, outer_train=False)    
                yield results

                num_batches += 1

IMAML = ImplicitModelAgnosticMetaLearning


