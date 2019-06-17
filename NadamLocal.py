# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 11:27:34 2017

@author: Ashish Katiyar
"""


import math

from collections import defaultdict, Iterable

import torch

from copy import deepcopy

from itertools import chain

from torch.autograd import Variable



required = object()





class Optimizer(object):

    """Base class for all optimizers.



    Arguments:

        params (iterable): an iterable of :class:`Variable` s or

            :class:`dict` s. Specifies what Variables should be optimized.

        defaults: (dict): a dict containing default values of optimization

            options (used when a parameter group doesn't specify them).

    """



    def __init__(self, params, defaults):

        self.defaults = defaults



        if isinstance(params, Variable) or torch.is_tensor(params):

            raise TypeError("params argument given to the optimizer should be "

                            "an iterable of Variables or dicts, but got " +

                            torch.typename(params))



        self.state = defaultdict(dict)

        self.param_groups = []



        param_groups = list(params)

        if len(param_groups) == 0:

            raise ValueError("optimizer got an empty parameter list")

        if not isinstance(param_groups[0], dict):

            param_groups = [{'params': param_groups}]



        for param_group in param_groups:

            self.add_param_group(param_group)



    def __getstate__(self):

        return {

            'state': self.state,

            'param_groups': self.param_groups,

        }



    def __setstate__(self, state):

        self.__dict__.update(state)



    def state_dict(self):

        """Returns the state of the optimizer as a :class:`dict`.



        It contains two entries:



        * state - a dict holding current optimization state. Its content

            differs between optimizer classes.

        * param_groups - a dict containing all parameter groups

        """

        # Save ids instead of Variables

        def pack_group(group):

            packed = {k: v for k, v in group.items() if k != 'params'}

            packed['params'] = [id(p) for p in group['params']]

            return packed

        param_groups = [pack_group(g) for g in self.param_groups]

        # Remap state to use ids as keys

        packed_state = {(id(k) if isinstance(k, Variable) else k): v

                        for k, v in self.state.items()}

        return {

            'state': packed_state,

            'param_groups': param_groups,

        }



    def load_state_dict(self, state_dict):

        """Loads the optimizer state.



        Arguments:

            state_dict (dict): optimizer state. Should be an object returned

                from a call to :meth:`state_dict`.

        """

        # deepcopy, to be consistent with module API

        state_dict = deepcopy(state_dict)

        # Validate the state_dict

        groups = self.param_groups

        saved_groups = state_dict['param_groups']



        if len(groups) != len(saved_groups):

            raise ValueError("loaded state dict has a different number of "

                             "parameter groups")

        param_lens = (len(g['params']) for g in groups)

        saved_lens = (len(g['params']) for g in saved_groups)

        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):

            raise ValueError("loaded state dict contains a parameter group "

                             "that doesn't match the size of optimizer's group")



        # Update the state

        id_map = {old_id: p for old_id, p in

                  zip(chain(*(g['params'] for g in saved_groups)),

                      chain(*(g['params'] for g in groups)))}



        def cast(param, value):

            """Make a deep copy of value, casting all tensors to device of param."""

            if torch.is_tensor(value):

                # Floating-point types are a bit special here. They are the only ones

                # that are assumed to always match the type of params.

                if any(tp in type(param.data).__name__ for tp in {'Half', 'Float', 'Double'}):

                    value = value.type_as(param.data)

                value = value.cuda(param.get_device()) if param.is_cuda else value.cpu()

                return value

            elif isinstance(value, dict):

                return {k: cast(param, v) for k, v in value.items()}

            elif isinstance(value, Iterable):

                return type(value)(cast(param, v) for v in value)

            else:

                return value



        # Copy state assigned to params (and cast tensors to appropriate types).

        # State that is not assigned to params is copied as is (needed for

        # backward compatibility).

        state = defaultdict(dict)

        for k, v in state_dict['state'].items():

            if k in id_map:

                param = id_map[k]

                state[param] = cast(param, v)

            else:

                state[k] = v



        # Update parameter groups, setting their 'params' value

        def update_group(group, new_group):

            new_group['params'] = group['params']

            return new_group

        param_groups = [

            update_group(g, ng) for g, ng in zip(groups, saved_groups)]

        self.__setstate__({'state': state, 'param_groups': param_groups})



    def zero_grad(self):

        """Clears the gradients of all optimized :class:`Variable` s."""

        for group in self.param_groups:

            for p in group['params']:

                if p.grad is not None:

                    data = p.grad.data

                    p.grad = Variable(data.new().resize_as_(data).zero_())



    def step(self, closure):

        """Performs a single optimization step (parameter update).



        Arguments:

            closure (callable): A closure that reevaluates the model and

                returns the loss. Optional for most optimizers.

        """

        raise NotImplementedError



    def add_param_group(self, param_group):

        """Add a param group to the :class:`Optimizer` s `param_groups`.



        This can be useful when fine tuning a pre-trained network as frozen layers can be made

        trainable and added to the :class:`Optimizer` as training progresses.



        Arguments:

            param_group (dict): Specifies what Variables should be optimized along with group

            specific optimization options.

        """

        assert isinstance(param_group, dict), "param group must be a dict"



        params = param_group['params']

        if isinstance(params, Variable):

            param_group['params'] = [params]

        else:

            param_group['params'] = list(params)



        for param in param_group['params']:

            if not isinstance(param, Variable):

                raise TypeError("optimizer can only optimize Variables, "

                                "but one of the params is " + torch.typename(param))

            if not param.requires_grad:

                raise ValueError("optimizing a parameter that doesn't require gradients")

            if not param.is_leaf:

                raise ValueError("can't optimize a non-leaf Variable")



        for name, default in self.defaults.items():

            if default is required and name not in param_group:

                raise ValueError("parameter group didn't specify a value of required optimization parameter " +

                                 name)

            else:

                param_group.setdefault(name, default)



        param_set = set()

        for group in self.param_groups:

            param_set.update(set(group['params']))



        if not param_set.isdisjoint(set(param_group['params'])):

            raise ValueError("some parameters appear in more than one parameter group")



        self.param_groups.append(param_group)

class Nadam(Optimizer):

    """Implements Adam algorithm.



    It has been proposed in `Adam: A Method for Stochastic Optimization`_.



    Arguments:

        params (iterable): iterable of parameters to optimize or dicts defining

            parameter groups

        lr (float, optional): learning rate (default: 1e-3)

        betas (Tuple[float, float], optional): coefficients used for computing

            running averages of gradient and its square (default: (0.9, 0.999))

        eps (float, optional): term added to the denominator to improve

            numerical stability (default: 1e-8)

        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)



    .. _Adam\: A Method for Stochastic Optimization:

        https://arxiv.org/abs/1412.6980

    """



    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,

                 weight_decay=0):

        defaults = dict(lr=lr, betas=betas, eps=eps,

                        weight_decay=weight_decay)

        super(Nadam, self).__init__(params, defaults)



    def step(self, closure=None):

        """Performs a single optimization step.



        Arguments:

            closure (callable, optional): A closure that reevaluates the model

                and returns the loss.

        """

        loss = None

        if closure is not None:

            loss = closure()



        for group in self.param_groups:

            for p in group['params']:

                if p.grad is None:

                    continue

                grad = p.grad.data

                if grad.is_sparse:

                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')



                state = self.state[p]



                # State initialization

                if len(state) == 0:

                    state['step'] = 0

                    # Exponential moving average of gradient values

                    state['exp_avg'] = torch.zeros_like(p.data)

                    # Exponential moving average of squared gradient values

                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['prod_mu_t'] = 1.



                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                beta1, beta2 = group['betas']



                state['step'] += 1



                if group['weight_decay'] != 0:

                    grad = grad.add(group['weight_decay'], p.data)



                # Decay the first and second moment running average coefficient


                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)


                # prod_mu_t = 1
                mu_t = beta1*(1 - 0.5*0.96**(state['step']/250))
                mu_t_1 = beta1*(1 - 0.5*0.96**((state['step']+1)/250))
                prod_mu_t = state['prod_mu_t'] * mu_t
                prod_mu_t_1 = prod_mu_t * mu_t_1

                state['prod_mu_t'] = prod_mu_t
                # for i in range(state['step']):
                #     mu_t = beta1*(1 - 0.5*0.96**(i/250))
                #     mu_t_1 = beta1*(1 - 0.5*0.96**((i+1)/250))
                #     prod_mu_t = prod_mu_t * mu_t
                #     prod_mu_t_1 = prod_mu_t * mu_t_1

                g_hat = grad/(1-prod_mu_t)
                m_hat = exp_avg / (1-prod_mu_t_1)

                m_bar = (1-mu_t)*g_hat + mu_t_1*m_hat

                exp_avg_sq_hat = exp_avg_sq/(1 - beta2 ** state['step'])

                denom = exp_avg_sq_hat.sqrt().add_(group['eps'])



                bias_correction1 = 1 - beta1 ** state['step']

                bias_correction2 = 1 - beta2 ** state['step']

                #step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                step_size = group['lr']

                p.data.addcdiv_(-step_size, m_bar, denom)
                #p.data.addcdiv_(-step_size, exp_avg, denom)



        return loss
