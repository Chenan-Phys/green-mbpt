#ifndef GREEN_MBPT_THC_CPU_KERNELS_H
#define GREEN_MBPT_THC_CPU_KERNELS_H

#include "kernels.h"
#include <green/integrals/thc_factor_data.h>
#include <Eigen/LU>

namespace green::mbpt::kernels {
  inline void native_thc_scope(const params::params& p,bool X2C,const symmetry::brillouin_zone_utils& bz) {
    if(X2C || bz.ink()!=bz.nk() || bz.inq()!=bz.nq())
      throw std::runtime_error("native THC v1 requires scalar full-BZ one-body and response data; disable spatial/TR reduction");
    if(p["q0_treatment"].as<sigma_q0_treatment_e>()==extrapolate)
      throw std::runtime_error("native THC GW extrapolation/AqQ is unsupported");
    if(p["P_sp"].as<bool>() || p["Sigma_sp"].as<bool>())
      throw std::runtime_error("native THC v1 requires double precision P and Sigma");
  }

  class thc_hf_cpu_kernel : public hf_kernel {
    std::shared_ptr<integrals::thc_factor_data> _factors;
  public:
    thc_hf_cpu_kernel(const params::params& p,size_t nao,size_t nso,size_t ns,size_t NQ,double madelung,
                      const bz_utils_t& bz,const ztensor<4>& S) : hf_kernel(p,nao,nso,ns,NQ,madelung,bz,S) {
      native_thc_scope(p,nao!=nso,bz);
      _factors=std::make_shared<integrals::thc_factor_data>(_hf_path,_nk,_nao,_NQ,_thc_options);
      if(_factors->set_kind()!="hf") throw std::runtime_error("native HF requires HF core");
    }

    ztensor<4> solve(const ztensor<4>& dm) {
      statistics.start("Native THC HF");
      ztensor<4> result(_ns,_nk,_nao,_nao); result.set_zero();
      const size_t r=_factors->rank(),q0=_factors->transfer(0,0);
      MatrixXcd m0=_factors->M(q0);
      // The legacy Hartree contracts two un-conjugated q0 factors. For the real
      // q0 frame this is Z; keep M M^T to preserve the precise complex ordering.
      MatrixXcd hartree=m0*m0.transpose();
      Eigen::VectorXcd density=Eigen::VectorXcd::Zero(r);
      for(size_t s=0;s<_ns;++s) for(size_t k=0;k<_nk;++k) {
        auto X=_factors->X(k);
        MatrixXcd projected=X*matrix(dm(s,k))*X.adjoint();
        density += projected.diagonal()/double(_nk);
      }
      Eigen::VectorXcd potential=hartree*density;
      const double prefactor=_ns==2?1.0:0.5;
      for(size_t sk=utils::context().global_rank;sk<_ns*_nk;sk+=utils::context().global_size) {
        const size_t s=sk/_nk,k=sk%_nk;
        auto X=_factors->X(k);
        matrix(result(s,k))=X.adjoint()*potential.asDiagonal()*X;
        for(size_t kp=0;kp<_nk;++kp) {
          auto Xp=_factors->X(kp);
          MatrixXcd projected=Xp*matrix(dm(s,kp))*Xp.adjoint();
          MatrixXcd Z=_factors->Z(_factors->transfer(k,kp));
          matrix(result(s,k)) -= prefactor/double(_nk)*(X.adjoint()*projected.cwiseProduct(Z)*X);
        }
        matrix(result(s,k)) -= prefactor*_madelung*matrix(_S_k(s,k))*matrix(dm(s,k))*matrix(_S_k(s,k));
      }
      utils::allreduce(MPI_IN_PLACE,result.data(),result.size(),MPI_C_DOUBLE_COMPLEX,MPI_SUM,utils::context().global);
      statistics.end(); statistics.print(utils::context().global); statistics.reset();
      return result;
    }
  };

  class thc_gw_cpu_kernel {
  public:
    using G_type=utils::shared_object<ztensor<5>>;
    thc_gw_cpu_kernel(const params::params& p,size_t nao,size_t nso,size_t ns,size_t NQ,
                      const grids::transformer_t& ft,const symmetry::brillouin_zone_utils& bz) :
      _n(nao),_ns(ns),_nk(bz.nk()),_nt(ft.sd().repn_fermi().nts()),_nw(ft.sd().repn_bose().nw()),_ft(ft) {
      native_thc_scope(p,nao!=nso,bz);
      _factors=std::make_shared<integrals::thc_factor_data>(p["dfintegral_file"],_nk,nao,NQ,integrals::thc_options(p));
      if(_factors->set_kind()!="correlation") throw std::runtime_error("native GW requires correlation core");
      if(_nt%2) throw std::runtime_error("native GW requires even fermionic tau grid");
      const size_t r=_factors->rank();
      if(double(_nt+_nw)*r*r*16 > double(p["thc_workspace_mb"].as<size_t>())*1024*1024)
        throw std::runtime_error("native THC GW tau/frequency workspace exceeds declared budget");
    }

    void solve(G_type& g,G_type& sigma) {
      auto ctx=g.cntx();
      sigma.fence(); if(!ctx.node_rank) sigma.object().set_zero(); sigma.fence();
      const size_t r=_factors->rank();
      // A node leader contracts full q/tau tiles; q is distributed over nodes.
      // Interpolation I is never partitioned by the old Gaussian-Q schedule.
      if(!ctx.node_rank) {
        for(size_t q=ctx.internode_rank;q<_factors->nq();q+=ctx.internode_size) {
          ztensor<4> chi(_nt,1,r,r),wc_w(_nw,1,r,r);
          chi.set_zero(); wc_w.set_zero();
          MatrixXcd Z=_factors->Z(q);
          for(size_t t=0;t<_nt/2;++t) {
            MatrixXcd bubble=MatrixXcd::Zero(r,r);
            for(size_t i=0;i<_nk;++i) for(size_t j=0;j<_nk;++j) {
              // GREEN response q has the opposite sign to the source provider.
              // Reverse this pair explicitly: source(j,i) is the Sigma core q.
              if(_factors->transfer(j,i)!=q) continue;
              for(size_t s=0;s<_ns;++s) {
                auto Xi=_factors->X(i),Xj=_factors->X(j);
                MatrixXcd left=Xi*matrix(g.object()(_nt-t-1,s,i))*Xi.adjoint();
                MatrixXcd right=Xj*matrix(g.object()(t,s,j))*Xj.adjoint();
                // P0_QR in GREEN is the transpose of the declared trace oracle.
                // Pair reversal changes M_plus to conj(M_minus), so chi^T is used.
                bubble -= (_ns==2?1.0:2.0)/double(_nk)*left.transpose().cwiseProduct(right);
              }
            }
            matrix(chi(t,0))=0.5*(bubble+bubble.adjoint()).eval();
            matrix(chi(_nt-t-1,0))=matrix(chi(t,0));
          }
          _ft.tau_f_to_w_b(chi,wc_w,0,_nw,true);
          MatrixXcd identity=MatrixXcd::Identity(r,r);
          for(size_t w=0;w<_nw;++w) {
            MatrixXcd response=matrix(wc_w(w,0));
            MatrixXcd A=identity-Z*response,rhs=Z*response*Z;
            Eigen::PartialPivLU<MatrixXcd> solver(A);
            MatrixXcd wc=solver.solve(rhs);
            const double residual=(A*wc-rhs).norm()/std::max(1.0,rhs.norm());
            if(!wc.allFinite() || residual>1e-9) throw std::runtime_error("native THC frequency screening solve failed");
            matrix(wc_w(w,0))=wc;
          }
          _ft.w_b_to_tau_f(wc_w,chi,0,_nt,true);
          for(size_t k=0;k<_nk;++k) for(size_t kp=0;kp<_nk;++kp) {
            if(_factors->transfer(k,kp)!=q) continue;
            auto X=_factors->X(k),Xp=_factors->X(kp);
            for(size_t t=0;t<_nt;++t) for(size_t s=0;s<_ns;++s) {
              MatrixXcd projected=Xp*matrix(g.object()(t,s,kp))*Xp.adjoint();
              matrix(sigma.object()(t,s,k)) -= X.adjoint()*projected.cwiseProduct(matrix(chi(t,0)))*X/double(_nk);
            }
          }
        }
        utils::allreduce(MPI_IN_PLACE,sigma.object().data(),sigma.object().size(),MPI_C_DOUBLE_COMPLEX,MPI_SUM,ctx.internode_comm);
      }
      sigma.fence();
    }
  private:
    size_t _n,_ns,_nk,_nt,_nw;
    const grids::transformer_t& _ft;
    std::shared_ptr<integrals::thc_factor_data> _factors;
  };
}
#endif
