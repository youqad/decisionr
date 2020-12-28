import torch
import numpy as np
import pyro
import matplotlib.pyplot as plt
from pyro.infer import MCMC, NUTS
from pyro.distributions import Normal, Dirichlet, Uniform

def utility(critVal, weight_val): 
    return critVal.dot(weight_val).item()

def crit_alt_matrix(number_alternatives, number_criterion):
    return Uniform(0, 1).rsample([number_alternatives, number_criterion])

def rank(i, crit_val_matrix, weight_val_vector):
    return sum(utility(c, weight_val_vector) > utility(crit_val_matrix[i], weight_val_vector) for c in crit_val_matrix)

def naive_acceptability_and_central_weight(number_criterion, number_alternatives, number_iterations):
    central_weight_vector = torch.zeros([number_alternatives, number_criterion]) # at i,j for alternative i at coordinate j
    weight_shape = torch.ones([number_criterion,])
    count_matrix = torch.zeros([number_alternatives, number_alternatives])  # at i,j for alternative i ranked j-th 
    
    for _ in range(number_iterations):
        weights = Dirichlet(weight_shape).sample()
        crit_alt_mat = crit_alt_matrix(number_alternatives, number_criterion) # at i,j for alternative i against criterion j
        rank_vector = [rank(i, crit_alt_mat, weights) for i in range(number_alternatives)] # best rank is 0
        for i in range(number_alternatives):
            count_matrix[i, rank_vector[i]] += 1
            if rank_vector[i] == 0:
                central_weight_vector[i] += weights
    acceptability_index = torch.zeros_like(count_matrix) # at i,j for approx proba of alternative i should be ranked j-th
    
    for i in range(number_alternatives):
        if  count_matrix[i, 0] > 0:
            central_weight_vector[i] /= count_matrix[i, 0] # average vector ranking alternative i on top
        for j in range(number_alternatives):
            acceptability_index[i, j] = count_matrix[i, j]/number_iterations  # approx proba for alternative i should be on top
    return central_weight_vector, acceptability_index

def naive_confidence_factor(number_criterion, number_alternatives, number_iterations, central_weight_vector):
    conf_factor = torch.zeros(number_alternatives)

    for _ in range(number_iterations):
        crit_alt_mat = crit_alt_matrix(number_alternatives, number_criterion)
        for i in range(number_alternatives):
            t = utility(crit_alt_mat[i], central_weight_vector[i])
            is_best_i = True
            for k in range(number_alternatives):
                if utility(crit_alt_mat[k], central_weight_vector[i]) > t:
                    is_best_i = False
                    break
            if is_best_i:
                conf_factor[i] += 1
            
    conf_factor /= number_iterations
    return conf_factor

class Weights:
    def __init__(self, interval_list):
        self.interval_list = interval_list

    def sample(self):
        return torch.FloatTensor([Uniform(a, b).sample() for a, b in self.interval_list])
        
class CriterionMatrix:
    def __init__(self, condition_matrix):
        self.condition_matrix = condition_matrix

    def sample(self):
        # TODO: write a better implementation
        f = lambda a: Uniform(a[0], a[1]).sample()
        lis = list(self.condition_matrix.size())
        sample = torch.zeros((lis[0], lis[1]))
        for i in range(lis[0]):
            for j in range(lis[1]):
                sample[i, j] += f(self.condition_matrix[i, j])
        return sample

condition_matrix = torch.FloatTensor([[(1,1),(1,2),(1,3)],[(1,1),(1,2),(1,3)],[(1,1),(1,2),(1,3)]])
print(condition_matrix)
print(condition_matrix.size())
print(CriterionMatrix(condition_matrix).sample())

a = naive_acceptability_and_central_weight(2, 3, 400)
print(a)
b = naive_confidence_factor(2, 3, 400, a[0])
print(b)

def acceptability_and_central_weight(number_criterion, number_alternatives, number_iterations, 
    interval_list, condition_matrix, return_acceptability = True, return_central_weight = True):
    central_weight_vector = torch.zeros([number_alternatives, number_criterion]) # at i,j for alternative i at coordinate j
    count_matrix = torch.zeros([number_alternatives, number_alternatives])  # at i,j for alternative i ranked j-th 
    weight_gen = Weights(interval_list)
    crit_alt_mat_gen = CriterionMatrix(condition_matrix)

    for _ in range(number_iterations):
        weights = weight_gen.sample()
        crit_alt_mat = crit_alt_mat_gen.sample() # at i,j for alternative i against criterion j
        rank_vector = [rank(i, crit_alt_mat, weights) for i in range(number_alternatives)] # best rank is 0
        for i in range(number_alternatives):
            count_matrix[i, rank_vector[i]] += 1
            if rank_vector[i] == 0:
                central_weight_vector[i] += weights
    acceptability_index = torch.zeros_like(count_matrix) # at i,j for approx proba of alternative i should be ranked j-th
    
    for i in range(number_alternatives):
        if  count_matrix[i, 0] > 0:
            central_weight_vector[i] /= count_matrix[i, 0] # average vector ranking alternative i on top
        for j in range(number_alternatives):
            acceptability_index[i, j] = count_matrix[i, j]/number_iterations  # approx proba for alternative i should be on top
    return central_weight_vector, acceptability_index

def confidence_factor(number_criterion, number_alternatives, number_iterations, central_weight_vector, condition_matrix):
    conf_factor = torch.zeros(number_alternatives)
    crit_alt_mat_gen = CriterionMatrix(condition_matrix)

    for _ in range(number_iterations):
        crit_alt_mat = crit_alt_mat_gen.sample()
        for i in range(number_alternatives):
            t = utility(crit_alt_mat[i], central_weight_vector[i])
            is_best_i = True
            for k in range(number_alternatives):
                if utility(crit_alt_mat[k], central_weight_vector[i]) > t:
                    is_best_i = False
                    break
            if is_best_i:
                conf_factor[i] += 1
            
    return conf_factor/number_iterations

    class SMAA:
        def __init__(self, name, number_criterion, number_alternatives, 
            interval_list, condition_matrix):
            self.name = name
            self._number_criterion = number_criterion
            self._number_alternatives = number_alternatives
            self._interval_list = interval_list
            self._condition_matrix = condition_matrix
            self.central_weight_value = None
            self.acceptability_value = None

        def get_acceptability_and_central_weight(self, number_iterations=1000):
            self.acceptability_value, self.central_weight = acceptability_and_central_weight(self._number_criterion, self._number_alternatives, number_iterations, self._interval_list, self._condition_matrix)
            return self.acceptability_value, self.central_weight

        def get_acceptability(self, number_iterations=1000):
            return self.get_acceptability_and_central_weight(number_iterations=number_iterations)[0]

        def get_central_weight(self, number_iterations=1000):
            return self.get_acceptability_and_central_weight(number_iterations=number_iterations)[1]

        def get_conf_factor(self, number_iterations=1000):
            if self.central_weight_value is None:
                self.central_weight(number_iterations)
            return confidence_factor(self._number_criterion, self._number_alternatives, number_iterations, self.central_weight_value, self._condition_matrix)
