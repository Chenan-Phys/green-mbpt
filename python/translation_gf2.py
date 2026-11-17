"""Exact GF2 contractions in a translation-invariant strip at zero transfer.

Orbitals are ordered by guiding center, then local orbital. Unwrapped integer
momentum is conserved; no periodic wrap or magnetic/time-reversal reduction is
introduced. Off-block probes are rejected. This is the same cubic functional,
evaluated with its zero matrix elements omitted.
"""
import numpy as np


class MomentumGF2:
    def __init__(self,v,block_size,time_chunk=16,tolerance=1e-10):
        self.v=np.asarray(v);self.d=int(block_size);self.n=len(v);self.tolerance=tolerance;self.time_chunk=time_chunk
        if self.d<1 or self.n%self.d:raise ValueError('Uniform contiguous orbital blocks required')
        self.nk=self.n//self.d;self.ids=np.arange(self.n).reshape(self.nk,self.d)
        p=np.repeat(np.arange(self.nk),self.d)
        conserved=p[:,None,None,None]-p[None,:,None,None]+p[None,None,:,None]-p[None,None,None,:]==0
        if np.max(abs(v[~conserved]),initial=0.)>tolerance:raise ValueError('Interaction does not conserve unwrapped strip momentum')
        self.off=p[:,None]!=p[None,:]
        triplets=[];self.ranges=[]
        for ki in range(self.nk):
            begin=len(triplets)
            for kb in range(self.nk):
                for kc in range(self.nk):
                    ka=kb+kc-ki
                    if 0<=ka<self.nk:triplets.append((ki,kb,kc,ka))
            self.ranges.append((begin,len(triplets)))
        self.ki,self.kb,self.kc,self.ka=np.array(triplets).T
        vv=v.reshape((self.nk,self.d)*4)
        self.left=vv[self.ki,:,self.kb,:,self.ka,:,self.kc,:]
        self.direct=vv[self.kb,:,self.ki,:,self.kc,:,self.ka,:]
        self.exchange=vv[self.kb,:,self.ka,:,self.kc,:,self.ki,:]

    def check(self,a):
        if a.shape[-2:]!=(self.n,self.n):raise ValueError('Unexpected orbital dimensions')
        if np.max(abs(a[...,self.off]),initial=0.)>self.tolerance*(1+np.max(abs(a),initial=0.)):
            raise ValueError('Probe/propagator does not preserve strip momentum')

    def __call__(self,g1,g2,g3,spin_weight=1):
        for a in (g1,g2,g3):self.check(a)
        out=np.zeros_like(g1,dtype=complex);nt,ns=g1.shape[:2]
        for start in range(0,nt,self.time_chunk):
            stop=min(start+self.time_chunk,nt);total=np.zeros((stop-start,ns,self.nk,self.d,self.d),complex)
            for s in range(ns):
                first=g1[start:stop,s][:,self.ids[:,:,None],self.ids[:,None,:]]
                left=np.einsum('Mibac,tMbd->tMiacd',self.left,first[:,self.kb],optimize=True)
                for r in range(ns):
                    second=g2[start:stop,r][:,self.ids[:,:,None],self.ids[:,None,:]]
                    third=g3[start:stop,r][:,self.ids[:,:,None],self.ids[:,None,:]]
                    mid=np.einsum('tMiacd,tMce->tMiade',left,second[:,self.kc],optimize=True)
                    three=np.einsum('tMiade,tMfa->tMifde',mid,third[:,self.ka],optimize=True)
                    partial=spin_weight*np.einsum('tMifde,Mdjef->tMij',three,self.direct,optimize=True)
                    if r==s:partial-=np.einsum('tMifde,Mdfej->tMij',three,self.exchange,optimize=True)
                    for k,(a,b) in enumerate(self.ranges):total[:,s,k]+=partial[:,a:b].sum(axis=1)
                view=out[start:stop,s]
                for k in range(self.nk):view[:,self.ids[k,:,None],self.ids[k,None,:]]=total[:,s,k]
        return out
