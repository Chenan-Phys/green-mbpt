"""Exact orbital-block contractions for a longitudinal GW Coulomb channel.

Unlike a dense orbital contraction, this stores only the nonzero (k,k+q)
vertex blocks. Both orientations and complex cosine/sine partners remain.
No change to the GW functional, screening derivative or response equation.
"""
import numpy as np
from translation_gw import BlockGWResponse


class OrbitalChannel:
    def __init__(self,vertices,block_size):
        self.d=int(block_size);self.n=vertices.shape[-1];self.nk=self.n//self.d
        if self.n%self.d:raise ValueError('Uniform orbital blocks required')
        self.ids=np.arange(self.n).reshape(self.nk,self.d)
        edges=[];values=[]
        for k in range(self.nk):
            for l in range(self.nk):
                block=vertices[:,self.ids[k,:,None],self.ids[l,None,:]]
                if np.max(abs(block),initial=0.)>0.:
                    edges.append((k,l));values.append(block)
        self.left=np.stack(values,axis=1)
        self.source,self.target=np.array(edges).T
        self.right=self.left.conj().swapaxes(-1,-2)
        paths=[(e,r,k,m) for e,(k,l) in enumerate(edges) for r,(n,m) in enumerate(edges) if l==n]
        self.path_left,self.path_right,ps,pt=np.array(paths).T
        self.output_paths=[(k,m,np.flatnonzero((ps==k)&(pt==m))) for k,m in sorted(set(zip(ps,pt)))]
        # Hermitian vertices give the reversed edge block exactly; no
        # conjugation is applied to either electron propagator or W.
        for e,(k,l) in enumerate(edges):
            actual=vertices[:,self.ids[l,:,None],self.ids[k,None,:]]
            if np.max(abs(actual-self.right[:,e]))>1e-12:raise ValueError('Hermitian channel vertices required')

    def blocks(self,g):
        return g[...,self.ids[:,:,None],self.ids[:,None,:]]

    def pair(self,left,right,spin_weight,time_chunk=16):
        result=np.empty((len(left),len(self.left),len(self.left)),complex)
        for begin in range(0,len(left),time_chunk):
            sl=slice(begin,begin+time_chunk)
            a=self.blocks(left[sl])[:,:,self.target]
            b=self.blocks(right[sl])[:,:,self.source]
            lg=np.einsum('Qeab,tsebc->tsQeac',self.left,a,optimize=True)
            rg=np.einsum('Recd,tseda->tsReca',self.right,b,optimize=True)
            result[sl]=-spin_weight*np.einsum('tsQeac,tsReca->tQR',lg,rg,optimize=True)
        return result

    def dynamic(self,g,w,time_chunk=16):
        result=np.zeros_like(g,dtype=complex)
        for begin in range(0,len(g),time_chunk):
            sl=slice(begin,begin+time_chunk)
            a=self.blocks(g[sl])[:,:,self.target]
            lg=np.einsum('Qeab,tsebc->tsQeac',self.left,a,optimize=True)
            screened=np.einsum('tQR,tsQeac->tsReac',w[sl],lg,optimize=True)
            # Retain every two-vertex path, including off-block outputs for an
            # arbitrary complex W. Physical translation-invariant W cancels
            # those outputs; the contraction does not impose that cancellation.
            partial=-np.einsum('tsRpac,Rpcb->tspab',screened[:,:,:,self.path_left],self.left[:,self.path_right],optimize=True)
            for k,m,paths in self.output_paths:
                result[sl,:,self.ids[k,:,None],self.ids[m,None,:]]=partial[:,:,paths].sum(axis=2)
        return result


class SparseBlockGWResponse(BlockGWResponse):
    def __init__(self,*args,**kwargs):
        # Keep the independent native-functional certificate in the original
        # dense contraction. Only repeated response actions use sparse blocks.
        super().__init__(*args,**kwargs)
        self.orbital_channels=[OrbitalChannel(c['vertices'],self.orbital_block_size) for c in self.channels]

    def sigma_variation(self,x,setup,screening=True,hartree_exchange=True,dynamic=True):
        self.check(x);dg=setup['fmaps'].backward(x)
        ds1=self.hf(-dg[-1]) if hartree_exchange else np.zeros_like(self.s1)
        ds=np.zeros_like(dg)
        if dynamic:
            for i,(c,kernel) in enumerate(zip(self.channels,self.orbital_channels)):
                ds+=kernel.dynamic(dg,c['time'])
                if screening:
                    dp=kernel.pair(dg,self.gt[::-1],self.spin_weight)+setup['phase'][:,None,None]*kernel.pair(self.gt,dg[::-1],self.spin_weight)
                    dw=setup['channel_shift'][i]@setup['bmaps'].forward(dp)@setup['channel_w'][i]
                    ds+=kernel.dynamic(self.gt,setup['bmaps'].backward(dw))
        return ds1[None]+setup['fmaps'].forward(ds)
