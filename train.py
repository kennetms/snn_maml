import torch
import math
import os
import time
import json
import logging
import numpy as np
import warnings

from torchmeta.utils.data import BatchMetaDataLoader
from torchvision.transforms import ToTensor, Resize, Compose
from tensorboardX import SummaryWriter 
from snn_maml.utils import tensors_to_device, compute_accuracy

from snn_maml.benchmarks import get_benchmark_by_name
from snn_maml.maml import ModelAgnosticMetaLearning
import snn_maml.utils as utils

import argparse

import pdb

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]="0" 

parser = argparse.ArgumentParser('MAML')

# General
parser.add_argument('--output-folder', default = './logs_tmp', type=str, help='Path to the output folder to save the model.') 
parser.add_argument('--benchmark', type=str, default='doublenmnistsequence', help='Name of the dataset (default: doublenmnistsequence).')
parser.add_argument('--folder', type=str, default='./', help='Root path containing parameters/ and data/.')
parser.add_argument('--num-ways', type=int, default=5, help='Number of classes per task (N in "N-way", default: 5).')
parser.add_argument('--num-ways-val', type=int, default=5, help='Number of classes per task validation (N in "N-way", default: 5).')
parser.add_argument('--num-shots', type=int, default=1, help='Number of training example per class (k in "k-shot", default: 5).')
parser.add_argument('--num-shots-test', type=int, default=10, help='Number of test example per class. If negative, same as the number of training examples `--num-shots` (default: 10).')
parser.add_argument('--warm-start', type=str, default='', help='model file to load for warm start')
parser.add_argument('--boil', action='store_true', help='body only in inner loop')
parser.add_argument('--quantize', type=str, default='', help='quantize weights')
parser.add_argument('--quantize_in', type=str, default='', help='quantize weights')
parser.add_argument('--learn-step-size', action='store_true', help='Learn step sizes')
parser.add_argument('--per-param-step-size', action='store_true', help='Learn per module step size')

# Model
parser.add_argument('--hidden-size', type=int, default=64,
    help='Number of channels in each convolution layer of the VGG network '
    '(default: 64).')

parser.add_argument('--load-model',type=str, default='', help='Path to the model file to load (default: "")')

# Optimization
parser.add_argument('--batch-size', type=int, default=5, help='Number of tasks in a batch of tasks (default: 5).')
parser.add_argument('--num-steps', type=int, default=1, help='Number of fast adaptation steps, ie. gradient descent '
    'updates (default: 1).')
parser.add_argument('--num-epochs', type=int, default=200, help='Number of epochs of meta-training (default: 50).')
parser.add_argument('--num-batches', type=int, default=100, help='Number of batch of tasks per epoch (default: 100).')
parser.add_argument('--num-batches-test', type=int, default=100, help='Number of batch of tasks per epoch (default: 100).')
parser.add_argument('--step-size', type=float, default=1.0, help='Size of the fast adaptation step, ie. learning rate in the gradient descent update (default: 1.0).')
parser.add_argument('--first-order', action='store_true', help='Use the first order approximation, do not use higher-order derivatives during meta-optimization.')
parser.add_argument('--meta-lr', type=float, default=.1e-2,  help='Learning rate for the meta-optimizer (optimization of the outer loss). The default optimizer is Adam (default: 2e-3).')

# Misc
parser.add_argument('--num-workers', type=int, default=8, help='Number of workers to use for data-loading (default: 4).')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--no-cuda', action='store_true')
parser.add_argument('--device-nonlin', action='store_true')
parser.add_argument('--weight-clamp', action='store_true')
parser.add_argument('--params_file', type=str, default='parameters/decolle_params-CNN.yml', help='Path to the parameters file if dcll is used.')

parser.add_argument('--do-test', action='store_true')

parser.add_argument('--do-train', action='store_true')

parser.add_argument('--do-noinner', action='store_true')

parser.add_argument('--do-noinner-test', action='store_true')

parser.add_argument('--deltaw', type=float, default=None, help='Force larger weight changes. The larger the value the larger the deltaw needs to be for params to update. (default None)')

parser.add_argument('--detach-at', type=int, default=None, help='Detach part of the network from specified layer (default None).')

parser.add_argument('--device', type=int, default=0, help='Which gpu to use if multiple available (default 0).')

# parser.add_argument('--deltaw', type=float, default=None, help='Force larger weight changes. The larger the value the larger the deltaw needs to be for params to update. (default None)')

args = parser.parse_args()

if not args.do_train and not args.do_test and not args.do_noinner:
    args.do_train = True # default to training

if args.num_shots_test <= 0:
    args.num_shots_test = args.num_shots

logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
device = torch.device(f'cuda:{args.device}' if not args.no_cuda and torch.cuda.is_available() else 'cpu')

if (args.output_folder is not None):
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        logging.debug('Creating folder `{0}`'.format(args.output_folder))

    output_folder = os.path.join(args.output_folder,
                          time.strftime('%Y-%m-%d_%H%M%S'))
    os.makedirs(output_folder)
    args.output_folder=output_folder
    logging.debug('Creating folder `{0}`'.format(output_folder))

    args.folder = os.path.abspath(args.folder)
    args.model_path = os.path.abspath(os.path.join(output_folder, 'model.th'))
    args.opt_path = os.path.abspath(os.path.join(output_folder, 'optim.th'))
    args.stepsize_path = os.path.abspath(os.path.join(output_folder, 'stepsize.th'))
    # Save the configuration in a config.json file
    with open(os.path.join(output_folder, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    logging.info('Saving configuration file in `{0}`'.format(
                 os.path.abspath(os.path.join(output_folder, 'config.json'))))



benchmark = get_benchmark_by_name(name = args.benchmark,
                                  folder = args.folder,
                                  num_ways = args.num_ways,
                                  num_ways_val = args.num_ways, #validation
                                  num_shots = args.num_shots, 
                                  num_shots_test = args.num_shots_test,
                                  detach_at=args.detach_at,
                                  hidden_size=args.hidden_size,
                                  params_file = args.params_file,
                                  device=device)
net = benchmark.model

meta_train_dataloader = BatchMetaDataLoader(benchmark.meta_train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers,
                                            pin_memory=True)


if not args.do_noinner:
    meta_val_dataloader = BatchMetaDataLoader(benchmark.meta_val_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True)

    if args.do_test or args.do_noinner_test:
        meta_test_dataloader = BatchMetaDataLoader(benchmark.meta_test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True)
        
        meta_val_dataloader = BatchMetaDataLoader(benchmark.meta_val_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True)



if hasattr(benchmark.model, 'get_trainable_parameters'):
    print('Using get_trainable_parameters instead of parameters for optimization parameters')
    meta_optimizer = torch.optim.Adam(benchmark.model.parameters(), lr=args.meta_lr) 
    if args.quantize:
        print('Quantize')
        pdb.set_trace()
        from snn_maml.utils import create_fixed_quantizers
        
        quantizer_out = create_fixed_quantizers()[args.quantize]
        
        ## only for quantizing the outer loop training - generally not needed
        # meta_optimizer = OptimLP(
        #         meta_optimizer,
        #         weight_quant=fixed_quantizers[int(args.quantize)],
        #         grad_scaling=1.e3)
    else:
        quantizer_out = None
        
    if args.quantize_in :
        print('Quantize')
        from snn_maml.utils import create_fixed_quantizers
        quantizer_in = create_fixed_quantizers()[args.quantize_in]
        
    else:
        quantizer_in = None

    meta_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(meta_optimizer,
                                                                eta_min=args.meta_lr/50,
                                                                T_max=args.num_epochs)
else:
    meta_optimizer = torch.optim.Adam(benchmark.model.parameters(), lr=args.meta_lr) 
    meta_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(meta_optimizer,
                                                                eta_min=args.meta_lr/50,
                                                                T_max=args.num_epochs)
    
    quantizer_out = None
    quantizer_in = None


    
print(benchmark.model, benchmark.input_size)

metalearner = ModelAgnosticMetaLearning(benchmark.model,
                                        meta_optimizer,
                                        first_order=args.first_order,
                                        num_adaptation_steps=args.num_steps,
                                        step_size=args.step_size,
                                        learn_step_size=args.learn_step_size,
                                        #deltaw=args.deltaw,
                                        loss_function=benchmark.loss_function,
                                        scheduler=meta_scheduler,
                                        custom_inner_update_fn = utils.device_update_nonlin_asymm if args.device_nonlin else None,
                                        custom_outer_update_fn = utils.inplace_soft_clamp_model_weights_asymm if args.weight_clamp else None,
                                        device=device,
                                        per_param_step_size=args.learn_step_size and args.per_param_step_size,
                                        boil = args.boil,
                                        outer_loop_quantizer = quantizer_out,
                                        inner_loop_quantizer = quantizer_in)

best_value = None



if args.warm_start != '':
    net.load_state_dict(torch.load(args.warm_start+'model.th'))
    #try:
    #    metalearner.optimizer.load_state_dict(torch.load(args.warm_start+'optim.th'))
    #except:
    #    warnings.warn('No optimization loading')
    try:
        step_size = torch.load(args.warm_start+'stepsize.th')
        metalearner.step_size.data = step_size
    except FileNotFoundError:
        warnings.warn('No step size loading')
        
elif args.load_model:
    print("loading model")
    with open(args.load_model, 'rb') as f:
        benchmark.model.load_state_dict(torch.load(f, map_location=device))
        print(benchmark.model)
        if os.path.exists(args.load_model[:-8]+'optim.th'):
            metalearner.optimizer.load_state_dict(torch.load(args.load_model[:-8]+'optim.th')) # -8 because we need to remove model.th

elif hasattr(net, 'LIF_layers'):
    out = next(iter(meta_train_dataloader))
    out_c = tensors_to_device(out, device=device)
    if args.params_file is not None:
        with open(args.params_file, 'r') as f:
            import yaml
            params = yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise Exception('Must provide params_file')
    
    #params['input_shape'] = benchmark.input_size
    from decolle.init_functions import init_LSUV_actrate
    dd=out_c['train'][0].reshape(-1,params['chunk_size_train'],*params['input_shape'])#params['input_shape'])
    print("dd",dd.shape)
    init_LSUV_actrate(net, dd, params['act_rate']) #0.288 is hard coded from params['actrate']
    

#pdb.set_trace()           
if args.do_noinner_test:
    # need to reinitialize final layer weights
    torch.nn.init.xavier_uniform(benchmark.model.LIF_layers[-1].base_layer.weight)
    benchmark.model.LIF_layers[-2].base_layer.requires_grad = False
    benchmark.model.LIF_layers[-3].base_layer.requires_grad = False
    benchmark.model.LIF_layers[-4].base_layer.requires_grad = False
    #i=1/0
    args.do_train=False

epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(args.num_epochs)))
results_accuracy_after = []

writer = SummaryWriter(log_dir=output_folder) 

all_test = np.zeros(args.num_epochs)

all_train = np.zeros(args.num_epochs)

from shutil import copy2
copy2(args.params_file, os.path.join(output_folder, 'params.yml'))

for epoch in range(args.num_epochs):
    print(epoch, meta_scheduler.get_last_lr())
    if args.do_train:
        results_train = metalearner.train(meta_train_dataloader,
                          max_batches=args.num_batches,
                          verbose=args.verbose,
                          desc='Training',
                          leave=False,
                          epoch = epoch)#,
                          #deltaw=args.deltaw)
        
        # if results_train is not None:
        #     all_train[epoch] = np.mean(results_train['accuracies_after'])
            
    if args.do_train or args.do_test:
        results = metalearner.evaluate(meta_val_dataloader,
                                       max_batches=args.num_batches_test,
                                       verbose=args.verbose,
                                       desc=epoch_desc.format(epoch + 1))
        all_train[epoch] = np.mean(results['accuracies_after'])
        
    if args.do_noinner:
        results = metalearner.train_no_inner(meta_train_dataloader,
                          max_batches=args.num_batches,
                          verbose=args.verbose,
                          desc='Training',
                          leave=False,
                          epoch = epoch)
        
        all_train[epoch] = np.mean(results['accuracies_after'])
        print("average of training", all_train)
        
    if args.do_noinner_test:
        results = metalearner.train_no_inner(meta_test_dataloader,
                          max_batches=1,#args.num_batches,
                          verbose=args.verbose,
                          desc='Training',
                          leave=False,
                          epoch = epoch,
                          is_train=True)
        
        all_train[epoch] = np.mean(results['accuracies_after'])
        print("average of training", all_train)
        
        results_test = metalearner.train_no_inner(meta_test_dataloader,
                          max_batches=20,#args.num_batches,
                          verbose=args.verbose,
                          desc='Training',
                          leave=False,
                          epoch = epoch,
                          is_train=False)
        
        all_test[epoch] = np.mean(results_test['accuracies_after'])
        print("average of testing", all_test)
        

    print("Results",results)
    # Save best model
    if 'accuracies_after' in results:
        results_accuracy_after.append(results['accuracies_after'])
        writer.add_scalar('accuracies_after/', results['accuracies_after'], epoch) 
        np.save(args.output_folder+'test_acc.npy',results_accuracy_after)
        save_model =True

        if save_model and (args.output_folder is not None):
            with open(args.model_path, 'wb') as f:
                torch.save(benchmark.model.state_dict(), f)
            with open(args.opt_path, 'wb') as f:
                torch.save(metalearner.optimizer.state_dict(), f)
            with open(args.stepsize_path, 'wb') as f:
                torch.save(metalearner.step_size, f)
                
    if args.do_test:
        results_test = metalearner.evaluate(meta_test_dataloader,
                           max_batches=args.num_batches_test,
                           verbose=args.verbose,
                           desc=epoch_desc.format(epoch + 1))#,
                           #deltaw=args.deltaw)
        
        print("Test results: ", np.mean(results_test['accuracies_after']))

        all_test[epoch] = np.mean(results_test['accuracies_after'])
                
if args.do_train:
    print("mean train", np.mean(all_train))
    print("stddev train", np.std(all_train))
    
    if (best_value is None) or (best_value < results['accuracies_after']):
        best_value = results['accuracies_after']
        save_model = True
    elif (best_value is None) or (best_value > results['mean_outer_loss']):
        best_value = results['mean_outer_loss']
        save_model = True
    else:
        save_model = False
        
    if save_model and (args.output_folder is not None):
        with open(args.model_path, 'wb') as f:
            torch.save(benchmark.model.state_dict(), f)
        with open(args.opt_path, 'wb') as f:
            torch.save(metalearner.optimizer.state_dict(), f)
        with open(args.stepsize_path, 'wb') as f:
                torch.save(metalearner.step_size, f)

if args.do_test:
    print("mean test", np.mean(all_test))
    print("stddev test", np.std(all_test))

if hasattr(benchmark.meta_train_dataset, 'close'):
    benchmark.meta_train_dataset.close()
    benchmark.meta_val_dataset.close()
