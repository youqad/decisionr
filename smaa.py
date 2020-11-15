from dataclasses import dataclass
import torch
import numpy as np
import pyro
import matplotlib.pyplot as plt
from pyro.infer import MCMC, NUTS
# import pyro.infer
# import pyro.optim
from pyro.distributions import Normal

# def model(data):
#     """
#     Explanation
#     """
#     coefs_mean = torch.zeros(dim)
#     coefs = pyro.sample('beta', dist.Normal(coefs_mean, torch.ones(3)))
#     y = pyro.sample('y', Bernoulli(logits=(coefs * data).sum(-1)), obs=labels)
#     return y

# nuts_kernel = NUTS(model, adapt_step_size=True)
# mcmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=300)
# mcmc.run(data)
# print(mcmc.get_samples()['beta'].mean(0))
# mcmc.summary(prob=0.5)

# def conditioned_model(model, sigma, y):
#     return poutine.condition(model, data={"obs": y})(sigma)

#     pyro.sample("obs_{}".format(i), dist.Bernoulli(f), obs=data[i])
#     conditioned_scale = pyro.condition(scale, data={"measurement": 9.5})
#     pyro.sample("measurement", dist.Normal(weight, 0.75), obs=9.5)

# def deferred_conditioned_scale(measurement, guess):
#     return pyro.condition(scale, data={"measurement": measurement})(guess)

# svi = pyro.infer.SVI(model=conditioned_scale,
#                      guide=scale_parametrized_guide,
#                      optim=pyro.optim.SGD({"lr": 0.001, "momentum":0.1}),
#                      loss=pyro.infer.Trace_ELBO())

class Alternative:
    """
    An alternative is a potential outcome for a decision making problem.

    Example: Tesla is an alternative for the decision problem of choosing a car to buy.
    """
    def __init__(self,name):
        self.name=name

class Criterion:
    """
    A criterion is a paramater in a decision making problem.

    It is given y
        - a name 'name'
        - an optionnal boolean 'positive' to indicate whether the criterion has a positive or negative impact on the alternatives

    Example: manoeuvrability might be a criterion when the alternatives are car brands.
    """
    def __init__(self,name,positive=True):
        self.name=name
        self.positive=positive

class Weight:
    """
    A weight represents how much a person values a certain criterion in a decision making problem.

    A weight is given by
        - a name 'name'
        - an optionnal distribution name 'dist' for modelling its uncertainty
        - a value 'value' for the weight
        - a criterion 'criterion'      

    Example: a weight of 21 can be given for the criterion manoeuvrability when car brands is the decision making problem.
    """
    def __init__(self,name,dist="Unif",value,variance=0,criterion):
        self.name=name
        self.dist=dist
        self.positive=positive
        self.value=value
        self.variance=variance
        self.criterion= criterion.name

class AlternativeCriterionMatrix:
    """
    TODO:write
    """
    def __init__(self):

class DecisionProblem:
    """
    A decision problem consist of a choice of possible outcomes: alternatives
    These alternatives depend on parameters: criteria
    A person values certain criteria more than others. This is reflected in weights.

    The weights and criteria for each alternative are fuzzy and are modelled with distributions.
    These distributions may reflect a lack of knowledge, a lack of objective measure, 
    a true randomness in the process, etc.

    Following the SMAA method, a person is guided to take a decision with three indicators. 
    - acceptabilityIndex: represents the approximate probability that a certain alternative is ranked first. 
    - centralWeightVector: represents a typical value for the weights that make a certain alternative ranked first.
    - confidenceFactor: represents the probability of an alternative being ranked for weights given by centralWeightVector.
    """
    def __init__(self,name,weights,criteria,alternatives):
        self.name=name
        self.weights=weights
        self.criteria=criteria
        self.alternatives=alternatives

    def criteriaList(self):
        return True    

    def alternativesList(self):
        return True        

    def weightsSampler(self):
        return True
    
    def criteriaSampler(self):
        return True

    def rank(self,alternative_number,sample_crit_vector,sample_weight_vector):
        return True

    def rankAcceptabilityIndex(self,alternative_number,rank):
        return True

    def acceptabilityIndex(self,alternative_number):
        """
        Test
        """ 
        return self.rankAcceptabilityIndex(alternative_number,1)

    def centralWeightVector(self,alternative_number):
        """
        Test
        """
        return True
    
    def confidenceFactor(self,alternative_number):
        """
        Test
        """
        return True