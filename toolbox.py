import numpy as np
import sys

class newton:
    
    def newton(f,dfdx,x0,tol=1e-6,max_iters=1000):
        diff = 1
        index = 0
        while diff >= tol:
            delta_x = -f(x0)/dfdx(x0)
            x = x0 + delta_x
            diff = np.abs(x - x0)
            x0 = x
            if index == max_iters:
                raise ValueError('Max iterations exceeded')
        return x0
    
class Dof:
    
    def Dof(grid):
        #given a vector where the length represents the number of layer (including input and output)
        #and each element represents the number of neurons per layer, we compute the degrees of freedom
        Dof = 0
        for i in range(grid.shape[0]-1):
            Dof += (grid[i]*grid[i+1])
        return Dof