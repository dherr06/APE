import numpy as np
import sys

class Dof:
    
    def Dof(grid):
        #given a vector where the length represents the number of layer (including input and output)
        #and each element represents the number of neurons per layer, we compute the degrees of freedom
        Dof = 0
        for i in range(grid.shape[0]-1):
            Dof += (grid[i]*grid[i+1])
        return Dof