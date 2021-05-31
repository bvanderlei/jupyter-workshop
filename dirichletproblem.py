# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:16:51 2021
@author: Ben Vanderlei

This module contains the routine dirichlet_solver for computing
a finite difference solution to Poisson's equation on a square.
Use of the solver requires a configuration file of the following 
structure.

[DIRICHLET DATA]
xlower = 0
xupper = 1
ylower = 0
yupper = 1
laplacian = 0
bc_all = y**2-x**2
bc_xlower = np.sin(2*np.pi*y)
bc_xupper = -np.sin(2*np.pi*y)
bc_ylower = 0
bc_yupper = 0

[FD SOLUTION]
n = 40
solver = lu
tolerance = 1e-06

xlower, xupper, ylower, yupper define domain boundaries.
laplacian is an expression in x and y that specifies poisson data.
bc_all, bc_xupper, bc_xlower, bc_yupper, bc_ylower are expressions
in x and y that define boundary data.  If bc_all is set, all boundaries
values are computed with the same expression.

n defines the number of grid points in each dimension
solver specifies the linear solver.  options are direct solve with 'lu'
or iterative solve with sparse matrix with 'cg'
tolerance is used with 'cg' solver
"""

import configparser
import numpy as np
import scipy.linalg as sla
from scipy.sparse import csr_matrix, linalg

def build_5pt_laplacian_matrix(N):
    '''
    build_5pt_laplacian_matrix(N)
    
    Parameters
    ----------
    N : int
        Specifies the number of points in each dimension of a square grid.

    Returns
    -------
    A : NumPy array
        Represents the standard 5-point finite difference discretization
        of the Laplacian
    '''
    A = np.zeros((N*N,N*N))
    
    ## Fill row by row an N*N x N*N square matrix 
    ## i,j from 0 to N-1 correspond to position x = xlower + (i+1)*h, y = ylower + (j+1)*h
    ## (x,y) position corresponds to row i+j*N in the matrix
    ## i or j = to 0 or N-1 correspond to grid boundaries
    
    for i in range(N):
        for j in range(N):
            A[i+j*N][i+j*N] = 4
            if (i>0):
                A[i+j*N][(i-1)+j*N] = -1
            if (i<N-1):
                A[i+j*N][(i+1)+j*N] = -1
            if (j>0):
                A[i+j*N][i+(j-1)*N] = -1
            if (j<N-1):
                A[i+j*N][i+(j+1)*N] = -1
    
    return A

def build_RHS(config):
    '''
    build_RHS(config)

    Parameters
    ----------
    config : ConfigParser object
        Contains problem data and parameters

    Returns
    -------
    b:  NumPy array
        Right hand vector of the linear system to be solved for the finite
        difference solution
    '''
    
    N = int(config['FD SOLUTION']['N'])
    xlower = float(config['DIRICHLET DATA']['xlower']) 
    ylower = float(config['DIRICHLET DATA']['ylower']) 
    xupper = float(config['DIRICHLET DATA']['xupper']) 
    yupper = float(config['DIRICHLET DATA']['yupper']) 

    laplacian = config['DIRICHLET DATA']['laplacian'] 
    
    if 'bc_all' in config['DIRICHLET DATA']:
            bc_xlower = config['DIRICHLET DATA']['bc_all']
            bc_ylower = config['DIRICHLET DATA']['bc_all']
            bc_xupper = config['DIRICHLET DATA']['bc_all']
            bc_yupper = config['DIRICHLET DATA']['bc_all']
    else:
            bc_xlower = config['DIRICHLET DATA']['bc_xlower']
            bc_ylower = config['DIRICHLET DATA']['bc_ylower']
            bc_xupper = config['DIRICHLET DATA']['bc_xupper']
            bc_yupper = config['DIRICHLET DATA']['bc_yupper']
        
    b = [0 for k in range(N*N)]
    h = (xupper-xlower)/(N+1)

    for i in range(N):
        for j in range(N):
            ## (x,y) are the coordinates of the grid point to be evaluated
            ## in the expressions for the boundary conditions and laplacian
            x,y  = xlower + h*(i+1), ylower + h*(j+1)
            b[i+j*N] = h*h*eval(laplacian)
            
            ## Add terms for Dirichlet BC
            if (i == 0):  
                x,y = xlower, ylower + h*(j+1)
                b[i+j*N] += eval(bc_xlower)
                
            if (i == N-1):
                x,y =  xupper, ylower + h*(j+1)
                b[i+j*N] += eval(bc_xupper)
                
            if (j == 0):
                x,y = xlower + h*(i+1), ylower
                b[i+j*N] += eval(bc_ylower)
                
            if (j == N-1):
                x,y = xlower + h*(i+1), yupper
                b[i+j*N] += eval(bc_yupper)
            
    return np.array(b)

def dirichlet_solver(configfile):
    '''
    Constructs finite difference solution for Poisson's problem on a square
    domain.  Problem data and solution parameters are specified by a
    configuration file described in dirichletproblem docstring.

    Parameters
    ----------
    configfile : filename
        Contains configuration specified in dirichletproblem docstring

    Returns
    -------
    U : NumPy array
        Contains the finite difference solution on the NxN grid.
    '''
    ## Read in configuration
    config = configparser.ConfigParser()
    config.read(configfile)
    N = int(config['FD SOLUTION']['N'])
    solver = config['FD SOLUTION']['solver']
    if 'tolerance' in config['FD SOLUTION']:
        cg_tol = config['FD SOLUTION']['tolerance']
    else:
        cg_tol = 1e-06
    
    A = build_5pt_laplacian_matrix(N)
    B = build_RHS(config)    
    
    if solver == 'lu':
        A_factorization = sla.lu_factor(A)
        U = sla.lu_solve(A_factorization,B)
    else:
        A_S = csr_matrix(A)
        U,flags = linalg.cg(A_S,B,tol=cg_tol)

    U = U.reshape((N,N))
    print("Solution using",solver,"complete.  Grid:",N,"x",N)
    return U