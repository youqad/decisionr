from dataclasses import dataclass
import torch
import numpy as np
import pyro
import matplotlib.pyplot as plt
from pyro.infer import MCMC, NUTS
from pyro.distributions import Normal, Dirichlet, Uniform

class Helper:
    @staticmethod
    def CriterionAlternativeMatrix(number_criterion,number_alternatives):
        return True

class SMAA:
    def __init__(self,name,number_criterion,number_alternatives,dist):
        self.name = name
        weight_shape = (number_criterion,)
        self.weight_vector = dist.rsample(weight_shape)

def utility(critVal,weightVal): 
    return critVal.dot(weightVal).item() 

def critAltMatrix(number_alternatives, number_criterion): 
    return Uniform(0, 1).rsample([number_alternatives, number_criterion])

def rank(i, critValMatrix, weightValVector): 
    return sum(utility(c, weightValVector) > utility(critValMatrix[i], weightValVector) for c in critValMatrix)

def naiveAcceptabilityIndexAndCentralWeightVector(number_criterion, number_alternatives, number_iterations):
    centralWeightVector = torch.zeros([number_alternatives, number_criterion]) # at i,j for alternative i at coordinate j
    weight_shape =  torch.ones([number_criterion,])
    countMatrix = torch.zeros([number_alternatives, number_alternatives])  # at i,j for alternative i ranked j-th 
    
    for _ in range(number_iterations):
        weights = Dirichlet(weight_shape).sample()
        crit_alt_mat = critAltMatrix(number_alternatives, number_criterion) # at i,j for alternative i against criterion j
        rankVector = [rank(i, crit_alt_mat, weights) for i in range(number_alternatives)] # best rank is 0
        for i in range(number_alternatives):
            countMatrix[i, rankVector[i]] += 1
            if rankVector[i] == 0:
                centralWeightVector[i] += weights
    acceptabilityIndex = torch.zeros_like(countMatrix) # at i,j for approx proba of alternative i should be ranked j-th
    
    for i in range(number_alternatives):
        if  countMatrix[i, 0] > 0:
            centralWeightVector[i] /= countMatrix[i, 0] # average vector ranking alternative i on top
        for j in range(number_alternatives):
            acceptabilityIndex[i, j] = countMatrix[i, j]/number_iterations  # approx proba for alternative i should be on top
    return centralWeightVector, acceptabilityIndex

print(naiveAcceptabilityIndexAndCentralWeightVector(2, 3, 400))

def naiveConfidenceFactor(number_criterion, number_alternatives, number_iterations, centralWeightVector):
    conf_factor = torch.zeros(number_alternatives)

    for _ in range(number_iterations):
        crit_alt_mat = critAltMatrix(number_alternatives, number_criterion)
        for i in range(number_alternatives):
            t = utility(crit_alt_mat[i],centralWeightVector[i])
            for k in range(number_alternatives):
                get_out = False
                if utility(crit_alt_mat[k],centralWeightVector[i]) > t:
                    get_out = True
                    break
                if get_out:
                    break
            conf_factor[i] += 1
            
    conf_factor /= number_iterations
    return conf_factor
