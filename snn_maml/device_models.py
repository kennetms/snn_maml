import torch
from collections import namedtuple
from .utils import softsign

def tonp(x):
    return x.detach().cpu().numpy()

class AbstractMemristorModel():
    def cond_update(self, update_tensor, weight_tensor, eta=1.0):
        raise NotImplementedError('Abstract Class')
        
    def soft_clamp(self, model):
        '''
        In-place function for clamping weights using L2 loss
        '''
        for name, param in model.get_trainable_named_parameters().items():
            if 'weight' in name:
                param.grad.add_(param*(param>self.wmax).float(), alpha=1.)            
                param.grad.add_(param*(param<=self.wmin).float(), alpha=1.)            
        
    def hard_clamp(self, model):
        for name, param in model.get_trainable_named_parameters().items():
            if 'weight' in name:
                param.data[:] = torch.clamp(param, self.wmin, self.wmax)

class GokmenHaenschModel(AbstractMemristorModel):
    # Original code parameters
    alpha = 0.02
    gamma = -.3125
    beta  = -.035
    sigma = 0.
    
    @property
    def wmax(self):
        return -self.alpha/(self.beta+self.gamma)
        
    @property
    def wmin(self):
        return -self.alpha/(self.beta-self.gamma)
    
    @property
    def wrange(self):
        return self.wmax-self.wmin
     
    def cond_update(self, update_tensor, weight_tensor, eta=1.0):
        dw = update_tensor
        ws = weight_tensor 

        A = (self.beta + self.gamma*softsign((-dw)))    
        B = self.alpha/A 
        w = -B + (B + ws)*torch.exp(A*(-dw*eta)) 
        deltaw = w - ws
        if self.sigma > 0:
            noise = torch.sqrt(torch.abs(deltaw)*self.wrange)*self.sigma
            return w + torch.normal(mean=0, std=torch.ones_like(w))*noise.detach()
        else:
            return w
        
        
    
class Model1(GokmenHaenschModel):
    # Parameters with similar asymm to original code parameters but normalized step size
    name = 'Model1'
    beta = -.54*.1
    gamma = -1.12*.2
    alpha = 1+beta/(beta-gamma)
    
class Model2(GokmenHaenschModel):
    # big asymmetry
    name = 'Model2'
    gamma = -1.21*1.2
    alpha=1.45
    beta = (alpha-1)* gamma/(alpha- 2)  
    
class Model1noise_high(GokmenHaenschModel):
    # Parameters with similar asymm to original code parameters but normalized step size
    name = 'Model1sighigh'
    beta = -.54*.1
    gamma = -1.12*.2
    alpha = 1+beta/(beta-gamma)
    sigma = .1
    
class Model2noise_high(GokmenHaenschModel):
    name = 'Model2sighigh'
    # big asymmetry
    gamma = -1.21*1.2
    alpha=1.45
    beta = (alpha-1)* gamma/(alpha- 2)  
    sigma = .1
    
class Model1noise_low(GokmenHaenschModel):
    name = 'Model1siglow'
    # Parameters with similar asymm to original code parameters but normalized step size
    beta = -.54*.1
    gamma = -1.12*.2
    alpha = 1+beta/(beta-gamma)
    sigma = .05
    
class Model2noise_low(GokmenHaenschModel):
    name = 'Model2siglow'
    # big asymmetry
    gamma = -1.21*1.2
    alpha=1.45
    beta = (alpha-1)* gamma/(alpha- 2)  
    sigma = .05
    
class Model3(GokmenHaenschModel):
    # big asymmetry
    name = 'Model3'
    gamma = -10
    alpha=1.45
    
    @property
    def beta(self):
        return (self.alpha-1)* self.gamma/(self.alpha- 2)     



            
class parameter_list():
    def __init__(self, a=0, b=0, c=0, d=0):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

class v1bModel(AbstractMemristorModel):
    # parameters
    SET_fit_parameter = parameter_list(  a= 0.000463337017370862, b=0.02528164200503319,   c=5192.9999635551,   d=9.448845809269882e-05)
    RESET_fit_parameter = parameter_list(a=-0.000463337017370862, b=-0.029557803724508176, c=4781.029599777505, d=9.448845809269882e-05)
    pulse_count = 10000
    dw_scaling_ratio = 1.2/pulse_count
    sigma = 0.
    
    @property
    def wmax(self):
        return 0.6
        
    @property
    def wmin(self):
        return -0.6
    
    @property
    def wrange(self):
        return self.wmax-self.wmin
        
    def map_fit_parameters_with_weights(self, SET_fit, RESET_fit):
        RESET_parameter = parameter_list()
        SET_parameter = parameter_list()
        min_conductance = RESET_fit.d
        max_conductance = SET_fit.a+SET_fit.d

        a_scaling_ratio = self.wrange/(max_conductance-min_conductance)
        SET_parameter.a = SET_fit.a * a_scaling_ratio
        RESET_parameter.a = RESET_fit.a * a_scaling_ratio

        d_drift = self.wmin-min_conductance
        SET_parameter.d = SET_fit.d + d_drift
        RESET_parameter.d = RESET_fit.d + d_drift

        SET_parameter.c = (SET_fit.c - (self.pulse_count/2))*self.dw_scaling_ratio
        RESET_parameter.c = (RESET_fit.c - (self.pulse_count/2))*self.dw_scaling_ratio

        SET_parameter.b = SET_fit.b/self.dw_scaling_ratio
        RESET_parameter.b = RESET_fit.b/self.dw_scaling_ratio

        return SET_parameter, RESET_parameter

    def __init__(self):
        self.SET_parameter, self.RESET_parameter = self.map_fit_parameters_with_weights(SET_fit=self.SET_fit_parameter, RESET_fit=self.RESET_fit_parameter)

    def v1b_fit_func(self, input_weight,parameter_list):
        # x nolmalized to 10000 pulses
        input = parameter_list.b*(parameter_list.c-input_weight)
        sign = softsign(input)
        cbrt = torch.pow(softsign(input)*input, 1/3)*sign
        twisted_weight = parameter_list.a/(1+torch.exp(cbrt))+parameter_list.d
        return twisted_weight

    def v1b_fit_func_inverted(self, twisted_weight,parameter_list):
        # x nolmalized to 10000 pulses
        input_weight = parameter_list.c-(torch.pow((torch.log((parameter_list.a/(twisted_weight-parameter_list.d))-1)), 3)/parameter_list.b)
        return input_weight
     
    def cond_update(self, update_tensor, weight_tensor, eta=1.0):
        dw = update_tensor
        ws = weight_tensor 

        weight_at_SET_x_axis = self.v1b_fit_func_inverted(ws, self.SET_parameter)
        update_SET = self.v1b_fit_func(weight_at_SET_x_axis-eta*dw, self.SET_parameter) - self.v1b_fit_func(weight_at_SET_x_axis, self.SET_parameter)
        weight_at_RESET_x_axis = self.v1b_fit_func_inverted(ws, self.RESET_parameter)
        update_RESET = self.v1b_fit_func(weight_at_RESET_x_axis-eta*dw, self.RESET_parameter) - self.v1b_fit_func(weight_at_RESET_x_axis, self.RESET_parameter)

        deltaw = -((update_SET+update_RESET)+softsign(-update_tensor)*(update_SET-update_RESET))/2

        if self.sigma > 0:
            noise = torch.sqrt(torch.abs(deltaw)*self.wrange)*self.sigma
            return deltaw + torch.normal(mean=0, std=torch.ones_like(ws))*noise.detach()
        else:
            return deltaw