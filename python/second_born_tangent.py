"""Linearized finite-system electronic second-Born self-energy."""
import numpy as np


class SecondBornTangent:
    """Cache the three differentiated Coulomb contractions at fixed G(tau).

    Each kernel maps an ordered pair of input orbitals to an ordered output
    pair. The third line uses the reversed time and carries the transfer phase.
    """
    def __init__(self, g, v, multiplicity):
        nt, ns, n, _ = g.shape
        n2 = n*n
        shape = (nt,ns,ns,n2,n2)
        self.forward = np.zeros(shape,complex)
        self.reverse = np.zeros(shape,complex)

        def paired(vertex, order):
            return np.ascontiguousarray(vertex.transpose(order)).reshape(n2,n2)

        def kernel(left, middle, right, reversed_pair=False):
            block = (left @ middle.reshape(-1,n2,n2) @ right.T).reshape(-1,n,n,n,n)
            order = (0,1,4,3,2) if reversed_pair else (0,1,4,2,3)
            return block.transpose(order).reshape(-1,n2,n2)

        left = [paired(v,p) for p in ((0,1,2,3),(0,3,1,2),(0,2,1,3))]
        for s in range(ns):
            for r in range(ns):
                interaction = multiplicity*v
                if s==r:
                    interaction = interaction-v.transpose(0,3,2,1)
                right = [paired(interaction,p) for p in ((0,1,2,3),(2,1,0,3),(3,1,0,2))]
                for start in range(0,nt,24):
                    sl = slice(start,start+24)
                    a,b,c = g[sl,s],g[sl,r],g[::-1][sl,r]
                    m = np.einsum('tce,tfa->tacef',b,c)
                    self.forward[sl,s,s] += kernel(left[0],m,right[0])
                    m = np.einsum('tbd,tfa->tbadf',a,c)
                    self.forward[sl,s,r] += kernel(left[1],m,right[1])
                    m = np.einsum('tbd,tce->tbcde',a,b)
                    self.reverse[sl,s,r] += kernel(left[2],m,right[2],True)

    def apply(self, dg, phase):
        flat = dg.reshape(*dg.shape[:2],-1)
        forward = np.einsum('tsrij,trj->tsi',self.forward,flat)
        reverse = np.einsum('tsrij,trj->tsi',self.reverse,flat[::-1])
        return (forward+phase[:,None,None]*reverse).reshape(dg.shape)
