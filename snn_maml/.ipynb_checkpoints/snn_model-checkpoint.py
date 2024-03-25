#!/bin/python
#-----------------------------------------------------------------------------
# File Name : meta_lenet_decolle.py
# Author: Emre Neftci
#
# Creation Date : Tue 08 Sep 2020 11:18:03 AM PDT
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 

from decolle.base_model import *
from decolle.lenet_decolle_model import *
from collections import OrderedDict
from torchmeta.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
                               MetaSequential, MetaLinear)

import torch.nn as nn

from decolle.utils import get_output_shape

import numpy as np

import warnings

import pdb

from matplotlib import pyplot as plt

m = nn.Sigmoid()

class FastSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        return  input_ / (1+torch.abs(input_))

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input / (torch.abs(input_) + 1.0) ** 2

# class FastSigmoid(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input_, th=1.25):#0):
#         ctx.save_for_backward(input_)
#         return  (input_>th).type(input_.dtype)

#     @staticmethod
#     def backward(ctx, grad_output):
#         (input_,) = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         return grad_input / (10 * torch.abs(input_) + 1.0) ** 2#, None
    
fast_sigmoid = FastSigmoid.apply



class MetaModuleNg(MetaModule):
    """
    MetaModule that returns only elements that require_grad
    """
    def meta_named_parameters(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._parameters.items() if isinstance(module, MetaModule) else [],
            prefix=prefix, recurse=recurse)
        for elem in gen:
            if elem[1].requires_grad:
                yield elem
                
class MetaLenetDECOLLE(LenetDECOLLE,MetaModuleNg):
    def __init__(self, burnin, detach_at=-1, sg_function_baseline = False, *args, **kwargs):
        self.non_spiking_baseline = sg_function_baseline
        if self.non_spiking_baseline is True:
            print('Using non-spiking model!')
        super(MetaLenetDECOLLE, self).__init__(*args, **kwargs)
        self.burnin = burnin
        self.detach_at = detach_at
    
    def build_conv_stack(self, Nhid, feature_height, feature_width, pool_size, kernel_size, stride, out_channels):
        output_shape = None
        padding = (np.array(kernel_size) - 1) // 2  
        for i in range(self.num_conv_layers):
            feature_height, feature_width = get_output_shape(
                [feature_height, feature_width], 
                kernel_size = kernel_size[i],
                stride = stride[i],
                padding = padding[i],
                dilation = 1)
            feature_height //= pool_size[i]
            feature_width //= pool_size[i]
            base_layer = MetaConv2d(Nhid[i], Nhid[i + 1], kernel_size[i], stride[i], padding[i])
            layer = self.lif_layer_type[i](base_layer,
                             alpha=self.alpha[i],
                             beta=self.beta[i],
                             alpharp=self.alpharp[i],
                             wrp=self.wrp[i],
                             deltat=self.deltat,
                             do_detach= True if self.method == 'rtrl' else False)
            pool = nn.MaxPool2d(kernel_size=pool_size[i])
            if self.lc_ampl is not None:
                readout = nn.Linear(int(feature_height * feature_width * Nhid[i + 1]), out_channels)

                # Readout layer has random fixed weights
                for param in readout.parameters():
                    param.requires_grad = False
                self.reset_lc_parameters(readout, self.lc_ampl[i])
            else:
                readout = nn.Identity()
            self.readout_layers.append(readout)

            if self.dropout[i] > 0.0:
                dropout_layer = nn.Dropout(self.dropout[i])
            else:
                dropout_layer = nn.Identity()


            self.LIF_layers.append(layer)
            self.pool_layers.append(pool)
            self.dropout_layers.append(dropout_layer)
        return (Nhid[-1],feature_height, feature_width)

    def build_mlp_stack(self, Mhid, out_channels): 
        output_shape = None

        for i in range(self.num_mlp_layers):
            print("MHID IS",Mhid)
            #Mhid[0] = Mhid[0]*2
            base_layer = MetaLinear(Mhid[i], Mhid[i+1], bias=False)
            layer = self.lif_layer_type[i+self.num_conv_layers](base_layer,
                         alpha=self.alpha[i],
                         beta=self.beta[i],
                         alpharp=self.alpharp[i],
                         wrp=self.wrp[i],
                         deltat=self.deltat,
                         do_detach=True if self.method == 'rtrl' else False)
            if self.lc_ampl is not None:
                readout = nn.Linear(Mhid[i+1], out_channels)
                # Readout layer has random fixed weights
                for param in readout.parameters():
                    param.requires_grad = False
                self.reset_lc_parameters(readout, self.lc_ampl[i])
            else:
                readout = nn.Identity()

            if self.dropout[i] > 0.0:
                dropout_layer = nn.Dropout(self.dropout[i])
            else:
                dropout_layer = nn.Identity()
            output_shape = out_channels

            self.LIF_layers.append(layer)
            self.pool_layers.append(nn.Sequential())
            self.readout_layers.append(readout)
            self.dropout_layers.append(dropout_layer)
        return (output_shape,)

    def build_output_layer(self, Mhid, out_channels):
        if self.with_output_layer:
            i=self.num_mlp_layers
            print("MHID",Mhid[i])
            base_layer = MetaLinear(Mhid[i], out_channels, bias=False)
            layer = self.lif_layer_type[-1](base_layer,
                         alpha=self.alpha[i],
                         beta=self.beta[i],
                         alpharp=self.alpharp[i],
                         wrp=self.wrp[i],
                         deltat=self.deltat,
                         do_detach=True if self.method == 'rtrl' else False)
            readout = nn.Identity()
            if self.dropout[i] > 0.0:
                dropout_layer = nn.Dropout(self.dropout[i])
            else:
                dropout_layer = nn.Identity()
                
            output_shape = out_channels
            print("output shape",output_shape)

            self.LIF_layers.append(layer)
            self.pool_layers.append(nn.Sequential())
            self.readout_layers.append(readout)
            self.dropout_layers.append(dropout_layer)
            
        return (output_shape,)

    def step(self, input, params = None):
        s_out = []
        r_out = []
        u_out = []
        i = 0
        for lif, pool, ro, do in zip(self.LIF_layers, self.pool_layers, self.readout_layers, self.dropout_layers):
            if i == self.num_conv_layers: 
                input = input.view(input.size(0), -1)
            #pdb.set_trace()
            s, u = lif(input, self.get_subdict(params,'LIF_layers.{}.base_layer'.format(i)))
            if i==self.detach_at:
                warnings.warn('detaching layer {0}'.format(lif))
                s=s.detach()
                u=u.detach()
                
            if self.num_conv_layers>0:
                u_p = pool(u)
                if i+1 == self.num_layers and self.with_output_layer:
                    s_ = sigmoid(u_p)
                    #sd_ = u_p
                    #r_ = ro(sd_.reshape(sd_.size(0), -1))
                elif self.non_spiking_baseline:
                    s_ = fast_sigmoid(u_p) #m(u_p) #Fastsigmoid
                else:
                    s_ = lif.sg_function(u_p)
                    #sd_ = do(s_)
                    #r_ = ro(sd_.reshape(sd_.size(0), -1))

                s_out.append(s_) 
                #r_out.append(r_)
                u_out.append(u_p)
            else:
                s_ = s
                s_out.append(s_)
                u_out.append(u)
            input = s_.detach() if lif.do_detach else s_
            i+=1

        return s_out, r_out, u_out
    
    
    def grad_flow(self, path):
            # helps monitor the gradient flow
            #pdb.set_trace()
            grad = [b.base_layer.weight.grad for b in self.LIF_layers]
            
            grad_norm = [torch.norm(g).item()/torch.numel(g) for g in grad] 

            plt.figure()
            plt.semilogy(grad_norm)
            plt.savefig(path + 'gradFlow_D.png')
            plt.close()

            return grad

class MetaRecLIFLayer(LIFLayer,MetaModuleNg):
    def __init__(*args, **kwargs):
        raise NotImplementedError('See previous commits of pytorch-maml:update__kennetms')

class MetaLIFLayer(LIFLayer,MetaModuleNg):
    def forward(self, Sin_t, params = None, *args, **kwargs ): 
        if self.state is None:
            self.init_state(Sin_t)
        if Sin_t.shape[0] != self.state.P.shape[0]:
            warnings.warn('Reinitializing state')
            self.init_state(Sin_t)
        
        state = self.state
        Q = self.beta * state.Q + (1-self.beta)*Sin_t*self.gain
        P = self.alpha * state.P + (1-self.alpha)*state.Q
        R = self.alpharp * state.R - (1-self.alpharp)*state.S * self.wrp
        U = self.base_layer(P, params=params) + R #soft Reset acts as some kind of bias?
        
        # is this how to implement hard reset?
        #U = (1-state.S.detach())*U
        
        S = self.sg_function(U)
        
        self.state = self.NeuronState(P=P, Q=Q, R=R, S=S)
        if self.do_detach: 
            state_detach(self.state)
        return S, U
    
    def init_parameters(self, *args, **kwargs):
        self.reset_parameters(self.base_layer, *args, **kwargs)
        
    @property
    def grad_norm(self):
        """Norm of weight gradients. Useful for monitoring gradient flow."""
        if self.weight_norm_enabled is False:
            if self.weight.grad is None:
                return 0
            else:
                return torch.norm(
                    self.weight.grad
                ).item() / torch.numel(self.weight.grad)
        else:
            if self.weight_g.grad is None:
                return 0
            else:
                return torch.norm(
                    self.weight_g.grad
                ).item() / torch.numel(self.weight_g.grad)
        
        
class CUBALIFLayer(BaseLIFLayer):
    """
        Use the same kind of dynamics as the Loihi CUBA neuron implemented by SLAYER/lava-dl
    """
    NeuronState = namedtuple('NeuronState', ['P', 'Q', 'R', 'U', 'S', 't','th'])
    sg_function  = FastSigmoid.apply

    def init_state(self, Sin_t):
        self.z = 0
        dtype = Sin_t.dtype
        if type(self.base_layer) == torch.nn.Sequential:
            # layer is changed to a Sequential module by the quantisation package
            device = self.base_layer[0].weight.device
        else:
            device = self.base_layer.weight.device
        input_shape = list(Sin_t.shape)
        out_ch = self.get_out_channels(self.base_layer)
        out_shape = self.get_out_shape(self.base_layer, input_shape)
        self.state = self.NeuronState(P=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device),#torch.zeros(input_shape).type(dtype).to(device),
                                      Q=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device),#torch.zeros(input_shape).type(dtype).to(device),    # Still keep Q for comaptability
                                      R=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device), # keep R for compatability
                                      U=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device),
                                      S=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device),
                                      t=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device),
                                      th=torch.zeros(1).type(dtype).to(device))
        
        
class MetaLIFLayerCUBA(LIFLayer,MetaModuleNg):#(CUBALIFLayer,MetaModuleNg):
    
    # try to just do a hard reset and see how performance changes
    def forward(self, Sin_t, params = None, *args, **kwargs ): 
        if self.state is None:
            self.init_state(Sin_t)
        if Sin_t.shape[0] != self.state.P.shape[0]:
            warnings.warn('Reinitializing state')
            self.init_state(Sin_t)
        
        state = self.state
        Q = self.beta * state.Q + (1-self.beta)*Sin_t*self.gain
        P = self.alpha * state.P + (1-self.alpha)*state.Q
        R = self.alpharp * state.R - (1-self.alpharp)*state.S * self.wrp
        U = self.base_layer(P, params=params) + 0 #R #soft Reset acts as some kind of bias?
        
        #pdb.set_trace()
        #threshold = 0.0000002
        spike_new = (R==-1)
        U = U * (spike_new < 0.5)
        
        # is this how to implement hard reset?
        #U = (1-state.S.detach())*U
        
        S = self.sg_function(U)
        
        self.state = self.NeuronState(P=P, Q=Q, R=R, S=S)
        if self.do_detach: 
            state_detach(self.state)
        return S, U
    
    # lava-dl replica(mathematically same? or close, but decolle does not do this so it's really bad performance)
#     def forward(self, Sin_t, params = None, *args, **kwargs ): 
#         if self.state is None:
#             self.init_state(Sin_t)
#         if Sin_t.shape[0] != self.state.P.shape[0]:
#             warnings.warn('Reinitializing state')
#             self.init_state(Sin_t)
            
#         state = self.state
        
#         # to handle the reset, from simple lif
#         #R = self.alpha * state.R + self.alpha * state.U * state.S
        
#         w_scale = 1 #4096
        
#         # exactly as lava-dl???
#         pdb.set_trace()
#         inp = self.base_layer(Sin_t,params=params)*w_scale
        
#         decayed_P = state.P * (w_scale-(self.beta*w_scale)/w_scale) #(1-self.beta)
        
#         #pdb.set_trace()
#         P = decayed_P + inp #(decayed_P + w_scale*Sin_t)/w_scale 
        
#         Q = state.Q * (w_scale-(self.alpha*w_scale)/w_scale)#(1-self.alpha)  #*(1-state.S)
        
#         # soft reset removed, R kept for compatibility
#         R = state.R #self.alpharp * state.R - (1-self.alpharp) * state.S
        
#         U = (Q + P) # + R  #(decayed_Q + w_scale*P)/w_scale
        
#         #pdb.set_trace()
        
#         # hard reset implemented in lava-dl, this breaks decolle. In lava-dl the params are scaled up, maybe that will help...
#         threshold = 0.0000002
#         spike_new = (U >= threshold)
#         U = U * (spike_new < 0.5)
        
#         #pdb.set_trace()
        

#         S = self.sg_function(U)#U)
        
#         t = state.t + 1
    
#         self.state = self.NeuronState(P=P, Q=Q, R=R, U=U.detach(), S=S, t=t.detach(), th=state.th.detach())
#         if self.do_detach:
#             state_detach(self.state)
#         return S, U
    
    def init_parameters(self, *args, **kwargs):
        self.reset_parameters(self.base_layer, *args, **kwargs)

    

def build_model_DECOLLE(out_features, params_file, device, detach_at, sg_function_baseline): # = 'maml/decolle_params-CNN.yml'
    from decolle.utils import parse_args, prepare_experiment, cross_entropy_one_hot
    import datetime, os, socket, tqdm
    import torch
    import torch.nn.functional as F

    params_file = params_file 
    with open(params_file, 'r') as f:
        import yaml
        params = yaml.safe_load(f)
    verbose = True
    
    reg_l = params['reg_l'] if 'reg_l' in params else None

    #d, t = next(iter(gen_train))
    input_shape = params['input_shape']
    ## Create Model, Optimizer and Loss
    net = MetaLenetDECOLLE(
                        out_channels=out_features,
                        Nhid=params['Nhid'],
                        Mhid=params['Mhid'],
                        kernel_size=params['kernel_size'],
                        pool_size=params['pool_size'],
                        stride = params['stride'],
                        input_shape=params['input_shape'],
                        alpha=params['alpha'],
                        alpharp=params['alpharp'],
                        beta=params['beta'],
                        dropout=params['dropout'],
                        num_conv_layers=params['num_conv_layers'],
                        num_mlp_layers=params['num_mlp_layers'],
                        lc_ampl=params['lc_ampl'],
                        lif_layer_type = MetaLIFLayer,
                        method=params['learning_method'],
                        with_output_layer=True,
                        wrp=params['wrp'],
                        burnin=params['burnin_steps'],
                        detach_at=detach_at,
                        sg_function_baseline=sg_function_baseline).to(device)
    
    net.LIF_layers[0].gain=10
    
    ##Initialize
    print("initializing parameters")
    net.init_parameters(torch.zeros([1,params['chunk_size_train']]+params['input_shape']).to(device))

    return net


def build_model_DECOLLECUBA(out_features, params_file, device, detach_at, sg_function_baseline): # = 'maml/decolle_params-CNN.yml'
    from decolle.utils import parse_args, prepare_experiment, cross_entropy_one_hot
    import datetime, os, socket, tqdm
    import torch
    import torch.nn.functional as F

    params_file = params_file 
    with open(params_file, 'r') as f:
        import yaml
        params = yaml.safe_load(f)
    verbose = True
    
    reg_l = params['reg_l'] if 'reg_l' in params else None

    #d, t = next(iter(gen_train))
    input_shape = params['input_shape']
    ## Create Model, Optimizer and Loss
    net = MetaLenetDECOLLE(
                        out_channels=out_features,
                        Nhid=params['Nhid'],
                        Mhid=params['Mhid'],
                        kernel_size=params['kernel_size'],
                        pool_size=params['pool_size'],
                        stride = params['stride'],
                        input_shape=params['input_shape'],
                        alpha=params['alpha'],
                        alpharp=params['alpharp'],
                        beta=params['beta'],
                        dropout=params['dropout'],
                        num_conv_layers=params['num_conv_layers'],
                        num_mlp_layers=params['num_mlp_layers'],
                        lc_ampl=None,#params['lc_ampl'],
                        lif_layer_type = MetaLIFLayerCUBA,
                        method=params['learning_method'],
                        with_output_layer=True,
                        wrp=params['wrp'],
                        burnin=params['burnin_steps'],
                        detach_at=detach_at,
                        sg_function_baseline=sg_function_baseline).to(device)
    
    net.LIF_layers[0].gain=10
    
    ##Initialize
    print("initializing parameters")
    net.init_parameters(torch.zeros([1,params['chunk_size_train']]+params['input_shape']).to(device))

    return net


def build_model_REDECOLLE(out_features, params_file, rec_layer=MetaRecLIFLayer, out_layer=MetaLIFLayer):
    from decolle.utils import parse_args, prepare_experiment, cross_entropy_one_hot
    import datetime, os, socket, tqdm
    import torch
    import torch.nn.functional as F

    params_file = params_file 
    with open(params_file, 'r') as f:
        import yaml
        params = yaml.load(f)
    verbose = True
    
    reg_l = params['reg_l'] if 'reg_l' in params else None

    #d, t = next(iter(gen_train))
    input_shape = params['input_shape']
    ## Create Model, Optimizer and Loss
    rec_lif_layer_type = rec_layer


    net = MetaLenetREDECOLLE(
                        out_channels=out_features,
                        Nhid=params['Nhid'],
                        Mhid=params['Mhid'],
                        kernel_size=params['kernel_size'],
                        pool_size=params['pool_size'],
                        stride = params['stride'],
                        input_shape=params['input_shape'],
                        alpha=params['alpha'],
                        alpharp=params['alpharp'],
                        beta=params['beta'],
                        dropout=params['dropout'],
                        num_conv_layers=params['num_conv_layers'],
                        num_mlp_layers=params['num_mlp_layers'],
                        lc_ampl=params['lc_ampl'],
                        lif_layer_type = [MetaLIFLayer]*len(params['Nhid'])+[rec_layer]*len(params['Mhid'])+[out_layer],
                        method=params['learning_method'],
                        with_output_layer=True,
                        wrp=params['wrp'],
                        burnin=params['burnin_steps']).cuda()
    
    net.init_parameters(torch.zeros([1,params['chunk_size_train']]+params['input_shape']).cuda())

    return net

