import torch as t

class ParameterError(Exception):
    pass

class SGD:

    def __init__(self, params, **kwargs):
        '''Implements SGD with momentum.

        Accepts parameters in groups, or an iterable.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
        kwargs can contain lr, momentum or weight_decay
        '''
        
        if not isinstance(params, (list, tuple)):
            params = [{"params": params}]

        default_parameters = {"momentum":0.0, "weight_decay":0.0}
        
        self.params = params
        param_dedupe = set()
        
        for i in range(len(self.params)):
            self.params[i] = {**default_parameters, **kwargs, **self.params[i]}
            if "lr" not in self.params[i]:
                raise ParameterError("No learning rate (lr) specified")

            self.params[i]["params"] = list(self.params[i]["params"])
            self.params[i]["previous_g_t"] = [t.zeros_like(p) for p in self.params[i]["params"]]

            for param in self.params[i]["params"]:
                if param in param_dedupe:
                    raise ParameterError("Parameter appears in more than one group")
                param_dedupe.add(param)


    def zero_grad(self) -> None:
        for param_group in self.params:
            thetas = param_group["params"]
            for param in thetas:
                param.grad = None


    @t.inference_mode()
    def sgd(self, param_group):
        #print(param_group)
        gamma = param_group["lr"]
        mu = param_group["momentum"]
        lam = param_group["weight_decay"]
        thetas = param_group["params"]
        t = 0

        for i, param in enumerate(thetas):
            g_t = param.grad
            g_t += lam * param
            g_t += mu * param_group["previous_g_t"][i]
            param -= gamma * g_t
            param_group["previous_g_t"][i] = g_t

    
    def step(self):
        for param_group in self.params:
            self.sgd(param_group)
