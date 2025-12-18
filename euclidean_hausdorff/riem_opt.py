import torch 
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Euclidean, Stiefel, Product, SpecialOrthogonalGroup
from pymanopt.optimizers import SteepestDescent

def create_orth_cost(orth, A, B):
    # only apply group transformation 
    # also used for special orthogonal group
    @pymanopt.function.pytorch(orth)
    def cost_o(X):
        dists = torch.cdist(A,B@X)

        h_A_to_B = torch.max(torch.min(dists, dim=1).values)
        h_B_to_A = torch.max(torch.min(dists, dim=0).values)

        return torch.max(h_A_to_B, h_B_to_A)
    return cost_o 

def create_euc_cost(euc, A, B):
    # only apply translation
    @pymanopt.function.pytorch(euc)
    def cost_euc(X):
        dists = torch.cdist(A,B+X)

        h_A_to_B = torch.max(torch.min(dists, dim=1).values)
        h_B_to_A = torch.max(torch.min(dists, axis=0).values)

        return torch.max(h_A_to_B, h_B_to_A)
    return cost_euc

def create_prod_cost(prod, A, B):
    @pymanopt.function.pytorch(prod)
    def cost_o(O,t):
        dists = torch.cdist(A, B@O + t)

        h_A_to_B = torch.max(torch.min(dists, dim=1).values) 
        h_B_to_A = torch.max(torch.min(dists, dim=0).values)

        return torch.max(h_A_to_B, h_B_to_A)
    return cost_o

def optimize_deh_sum(A, B, special_eucl=False, verbose=0):
    # First optimize over orthogonal transformations, then over translations
    D = A.shape[1]
    opt = SteepestDescent(verbosity=verbose)
    
    ortho = SpecialOrthogonalGroup(D) if special_eucl else Stiefel(D,D)
    eucl = Euclidean(D)
    
    orth_cost = create_orth_cost(ortho, A, B)
    orth_problem = pymanopt.Problem(ortho,orth_cost)
    
    orth_result = opt.run(orth_problem)
    o_inv = orth_result.point
    
    # feed in the point cloud with the optimal rotation applied
    euc_cost = create_euc_cost(eucl, A, B@o_inv)
    euc_problem = pymanopt.Problem(eucl,euc_cost)
    
    trans_result = opt.run(euc_problem)
    t_inv = trans_result.point

    return trans_result.cost, o_inv, t_inv

def optimize_deh_prod(A, B, special_eucl=False, verbose=0):
    D = A.shape[1]
    opt = SteepestDescent(verbosity=verbose)
    
    ortho = SpecialOrthogonalGroup(D) if special_eucl else Stiefel(D,D)
    eucl = Euclidean(D)
    prod = Product([ortho,eucl])
    
    prod_cost = create_prod_cost(prod,A,B)
    prod_problem = pymanopt.Problem(prod,prod_cost)
    
    result = opt.run(prod_problem)
    o_inv,t_inv = result.point

    return result.cost, o_inv, t_inv

