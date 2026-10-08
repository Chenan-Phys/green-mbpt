#ifndef GREEN_MBPT_THC_CPU_KERNELS_H
#define GREEN_MBPT_THC_CPU_KERNELS_H

#include "kernels.h"
#include <green/integrals/thc_factor_data.h>
#include <Eigen/LU>
#include <green/tensors/thc_gw_fft.h>
#include <green/tensors/thc_sigma_orbital.h>
#include <exception>
#include <thread>
#include <mutex>
#include <chrono>
#include <map>
#include <iomanip>
#include <algorithm>

namespace green::mbpt::kernels {
  struct thc_cpu_component_timer {
    std::map<std::string,double>* times;
    const char* name;
    std::chrono::steady_clock::time_point begin;
    thc_cpu_component_timer(std::map<std::string,double>* output,const char* label):times(output),name(label) {
      if(times)begin=std::chrono::steady_clock::now();
    }
    ~thc_cpu_component_timer(){if(times)(*times)[name]+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();}
  };
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
      if(6.0*_factors->rank()*_factors->rank()*16>double(p["thc_workspace_mb"].as<size_t>())*1024*1024)
        throw std::runtime_error("native THC HF matrix workspace exceeds declared budget");
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
      _fft=p["thc_gw_k_contraction"].as<std::string>()=="fft";
      _workspace_bytes=p["thc_workspace_mb"].as<size_t>()*1024*1024;
      _screening=p["thc_gw_screening"].as<std::string>();
      _auxiliary=tensors::thc_auxiliary_screening(_screening,r,NQ);
      _orbital=tensors::thc_sigma_orbital(p["thc_gw_sigma"].as<std::string>(),_n,r,NQ,_nk,_ns,_auxiliary,!_fft);
      _reuse_fft=p["thc_fft_reuse_screening"].as<bool>();
      _cpu_threads=p["thc_cpu_threads"].as<size_t>();
      _profile=p["thc_profile"].as<bool>();
      if(!_cpu_threads || _cpu_threads>64)throw std::runtime_error("thc_cpu_threads must be in 1..64");
      if(_profile && _cpu_threads>1)throw std::runtime_error("native THC CPU component profiling requires thc_cpu_threads=1");
      if(_profile && (_fft || !_auxiliary))throw std::runtime_error("native THC CPU component profiling requires auxiliary screening and direct momentum mode");
      if(_cpu_threads>1) {
        int thread_support=MPI_THREAD_SINGLE;MPI_Query_thread(&thread_support);
        if(thread_support<MPI_THREAD_FUNNELED)
          throw std::runtime_error("thc_cpu_threads requires MPI_Init_thread with at least MPI_THREAD_FUNNELED");
      }
      _vertices.resize(_factors->nq());
      _pairs.resize(_factors->nq());
      for(size_t k=0;k<_nk;++k)for(size_t kp=0;kp<_nk;++kp)
        _pairs[_factors->transfer(k,kp)].emplace_back(k,kp);
      _bubble_pairs=_pairs;
      for(auto& pairs:_bubble_pairs)std::sort(pairs.begin(),pairs.end(),[](const auto& a,const auto& b){
        return std::make_pair(a.second,a.first)<std::make_pair(b.second,b.first);});
      if(_fft){tensors::thc_momentum_fft layout(*_factors);tensors::check_thc_fft_workspace(*_factors,_nt,_nw,_workspace_bytes,_auxiliary);}
      if(!utils::context().global_rank)std::cout<<"Native THC CPU GW momentum mode "<<(_fft?"host FFT":"direct sums")<<std::endl;
      const double direct_elements=_auxiliary?auxiliary_scratch_elements():double(_nt+_nw)*r*r;
      if(!_fft && direct_elements*sizeof(std::complex<double>)>double(_workspace_bytes))
        throw std::runtime_error("native THC GW tau/frequency workspace exceeds declared budget");
    }

    void solve(G_type& g,G_type& sigma) {
      _component_seconds.clear();
      auto ctx=g.cntx();
      sigma.fence(); if(!ctx.node_rank) sigma.object().set_zero(); sigma.fence();
      const size_t r=_factors->rank();
      // A node leader contracts full q/tau tiles; q is distributed over nodes.
      // Interpolation I is never partitioned by the old Gaussian-Q schedule.
      if(!ctx.node_rank) {
        if(_fft) {
          if(!ctx.internode_rank){tensors::thc_cpu_matrix_ops ops;tensors::thc_gw_fft_solve(*_factors,_ft,g.object(),sigma.object(),ops,_workspace_bytes,_screening,_reuse_fft);}
          utils::allreduce(MPI_IN_PLACE,sigma.object().data(),sigma.object().size(),MPI_C_DOUBLE_COMPLEX,MPI_SUM,ctx.internode_comm);
        } else {
        if(_auxiliary) {
          solve_auxiliary_direct(g.object(),sigma.object(),ctx.internode_rank,ctx.internode_size);
        } else {
        // Cache complete projected G only when it fits beside the one-q arrays.
        // Streaming remains available for larger meshes/ranks under the same budget.
        std::vector<MatrixXcd> projected_g;
        const double cache_bytes=double(_nt*_ns*_nk+_nt+_nw+12)*r*r*16;
        if(cache_bytes<=double(_workspace_bytes)) {
          projected_g.reserve(_nt*_ns*_nk);
          for(size_t t=0;t<_nt;++t)for(size_t s=0;s<_ns;++s)for(size_t k=0;k<_nk;++k) {
            auto X=_factors->X(k);
            projected_g.emplace_back(X*matrix(g.object()(t,s,k))*X.adjoint());
          }
        }
        if(!ctx.global_rank)std::cout<<"Native THC CPU projections "<<(projected_g.empty()?"streamed":"cached across q")
          <<"; screening "<<(_auxiliary?"auxiliary":"point")<<" dimension "<<(_auxiliary?_factors->naux():r)<<std::endl;
        for(size_t q=ctx.internode_rank;q<_factors->nq();q+=ctx.internode_size) {
          ztensor<4> chi(_nt,1,r,r),wc_w(_nw,1,r,r);
          chi.set_zero(); wc_w.set_zero();
          MatrixXcd m=_factors->M(q),Z;
          if(!_auxiliary)Z=m*m.adjoint();
          for(size_t t=0;t<_nt/2;++t) {
            MatrixXcd bubble=MatrixXcd::Zero(r,r);
            for(const auto& pair:_bubble_pairs[q]) {
              const size_t i=pair.second,j=pair.first;
              // GREEN response q has the opposite sign to the source provider.
              // Reverse this pair explicitly: source(j,i) is the Sigma core q.
              for(size_t s=0;s<_ns;++s) {
                auto Xi=_factors->X(i),Xj=_factors->X(j);
                // P0_QR in GREEN is the transpose of the declared trace oracle.
                // Pair reversal changes M_plus to conj(M_minus), so chi^T is used.
                if(!projected_g.empty()) {
                  const auto& left=projected_g[((_nt-t-1)*_ns+s)*_nk+i];
                  const auto& right=projected_g[(t*_ns+s)*_nk+j];
                  bubble -= (_ns==2?1.0:2.0)/double(_nk)*left.transpose().cwiseProduct(right);
                }else {
                  MatrixXcd left=Xi*matrix(g.object()(_nt-t-1,s,i))*Xi.adjoint();
                  MatrixXcd right=Xj*matrix(g.object()(t,s,j))*Xj.adjoint();
                  bubble -= (_ns==2?1.0:2.0)/double(_nk)*left.transpose().cwiseProduct(right);
                }
              }
            }
            matrix(chi(t,0))=0.5*(bubble+bubble.adjoint()).eval();
            matrix(chi(_nt-t-1,0))=matrix(chi(t,0));
          }
          _ft.tau_f_to_w_b(chi,wc_w,0,_nw,true);
          for(size_t w=0;w<_nw;++w) {
            MatrixXcd response=matrix(wc_w(w,0));
            matrix(wc_w(w,0))=tensors::thc_screened_correlation(m,Z,response,_auxiliary);
          }
          _ft.w_b_to_tau_f(wc_w,chi,0,_nt,true);
          for(const auto& pair:_pairs[q]) {
            const size_t k=pair.first,kp=pair.second;
            auto X=_factors->X(k),Xp=_factors->X(kp);
            for(size_t t=0;t<_nt;++t) for(size_t s=0;s<_ns;++s) {
              MatrixXcd temporary;const MatrixXcd* projected=nullptr;
              if(!projected_g.empty())projected=&projected_g[(t*_ns+s)*_nk+kp];
              else {temporary=Xp*matrix(g.object()(t,s,kp))*Xp.adjoint();projected=&temporary;}
              matrix(sigma.object()(t,s,k)) -= X.adjoint()*projected->cwiseProduct(matrix(chi(t,0)))*X/double(_nk);
            }
          }
        }
        }
        utils::allreduce(MPI_IN_PLACE,sigma.object().data(),sigma.object().size(),MPI_C_DOUBLE_COMPLEX,MPI_SUM,ctx.internode_comm);
        }
      }
      sigma.fence();
      if(_profile && !ctx.global_rank){
        std::cout<<"Native THC CPU components {";bool first=true;
        for(const auto& item:_component_seconds){if(!first)std::cout<<",";first=false;
          std::cout<<"\""<<item.first<<"\":"<<std::setprecision(17)<<item.second;}
        std::cout<<"}"<<std::endl;
      }
    }
  private:
    struct orbital_pair {size_t k,kp;tensors::thc_vertex_matrix v;};
    double auxiliary_scratch_elements() const {
      const double r=_factors->rank(),Q=_factors->naux(),n=_n;
      // One Q-space history/frequency transform, transform and LU temporaries,
      // point bubble/projections, one expanded W slice and one copied M core.
      // Source G/Sigma and loader-owned factors have separate ownership.
      return (2.*_nt+2.*_nw+8)*Q*Q+12.*r*r+4.*r*Q+4.*r*n+2.*n*n;
    }
    void solve_auxiliary_direct(const ztensor<5>& g,ztensor<5>& sigma,size_t owner,size_t owners) {
      const size_t r=_factors->rank(),Q=_factors->naux();
      // GREEN v1 ndarray slices copy a non-atomic storage reference count.
      // Borrow contiguous buffers once; workers must never create tensor views.
      const auto* green_data=g.data();auto* sigma_data=sigma.data();
      auto green_at=[&](size_t t,size_t s,size_t k) {
        return CMMatrixXcd(green_data+((t*_ns+s)*_nk+k)*_n*_n,_n,_n);
      };
      auto sigma_at=[&](size_t t,size_t s,size_t k) {
        return MMatrixXcd(sigma_data+((t*_ns+s)*_nk+k)*_n*_n,_n,_n);
      };
      std::vector<size_t> qs;
      for(size_t q=owner;q<_factors->nq();q+=owners)qs.push_back(q);
      if(qs.empty())return;
      const double rr=double(r)*r,budget=double(_workspace_bytes)/sizeof(std::complex<double>);
      const double vertex_scratch=_orbital?double(_n)*_n*(r+6.*Q):0.;
      const double vertex_cache=_orbital?double(qs.size())*_nk*_n*_n*Q:0.;
      const double scratch=auxiliary_scratch_elements()+vertex_scratch;
      const double histories=double(qs.size())*(double(_nt)*Q*Q+double(r)*Q);
      // The same allowance holds either both projected half-time G slices for
      // the bubble, or projected G plus point Sigma at one time, over spins/k.
      const double time_fields=(2.*_ns*_nk+2)*rr+2.*_nk*r*_n+2.*_nk*_n*_n;
      const bool retain=histories+scratch+time_fields+vertex_cache<=budget;
      const double one_vertex=_orbital?double(_nk)*_n*_n*Q:0.;
      const bool time_batch=scratch+time_fields+one_vertex<=budget;
      const double bubble_worker=time_fields+4.*rr+2.*r*Q;
      const double sigma_worker=_orbital?double(_n)*_n*(6.*Q+2.*_ns*_nk):bubble_worker;
      const double per_worker=std::max(bubble_worker,sigma_worker);
      const double base_working=scratch+(time_batch?time_fields:0.)+(retain?histories+vertex_cache:one_vertex);
      if(!retain && _orbital && scratch+double(_nk)*_n*_n*Q>budget)
        throw std::runtime_error("orbital THC Sigma workspace cannot hold one q vertex slice");
      const size_t workers=retain?std::min({_cpu_threads,_nt,size_t(1.+std::max(0.,budget-base_working)/per_worker)}):1;
      const double working=base_working+(workers-1)*per_worker;
      auto parallel_tau=[&](size_t count,auto&& operation) {
        std::exception_ptr failure;std::mutex failure_mutex;
        auto work=[&](size_t worker){
          try{for(size_t t=count*worker/workers;t<count*(worker+1)/workers;++t)operation(t);}
          catch(...){std::lock_guard<std::mutex> guard(failure_mutex);if(!failure)failure=std::current_exception();}
        };
        std::vector<std::thread> threads;
        try{for(size_t worker=1;worker<workers;++worker)threads.emplace_back(work,worker);}
        catch(...){for(auto& thread:threads)thread.join();throw;}
        work(0);for(auto& thread:threads)thread.join();if(failure)std::rethrow_exception(failure);
      };
      std::vector<MatrixXcd> projected_g;
      if((!_orbital || !retain) && working+double(_nt)*_ns*_nk*rr<=budget) {
        thc_cpu_component_timer timer(_profile?&_component_seconds:nullptr,"projected_G_cache");
        projected_g.resize(_nt*_ns*_nk);
        parallel_tau(projected_g.size(),[&](size_t index) {
          const size_t t=index/(_ns*_nk),s=index/_nk%_ns,k=index%_nk;
          auto X=_factors->X(k);
          projected_g[index]=X*green_at(t,s,k)*X.adjoint();
        });
      }
      if(!utils::context().global_rank)std::cout<<"Native THC CPU projections "<<(projected_g.empty()?"streamed":"cached across q")
        <<"; screening auxiliary dimension "<<Q<<"; histories "<<(retain?"owned q retained":"one q streamed")
        <<"; bubble projections "<<(retain?"shared across q per half-tau pair":"one q")
        <<"; Sigma "<<(retain?"owned q accumulated":time_batch?"one q accumulated":"one (spin,k) streamed")
        <<"; Sigma route "<<(_orbital?"orbital":"point")<<"; tau workers "<<workers<<" requested "<<_cpu_threads<<std::endl;
      auto build_vertices=[&](size_t q,const MatrixXcd& m) {
        std::vector<orbital_pair> pairs;
        for(const auto& pair:_pairs[q])
          pairs.push_back({pair.first,pair.second,tensors::thc_orbital_vertex(_factors->X(pair.first),_factors->X(pair.second),m)});
        return pairs;
      };
      auto orbital_tau=[&](size_t t,const MatrixXcd& core,const std::vector<orbital_pair>& pairs) {
        thc_cpu_component_timer timer(_profile?&_component_seconds:nullptr,"orbital_sigma");
        for(const auto& pair:pairs) {
          tensors::thc_vertex_matrix weighted=pair.v*core;
          for(size_t s=0;s<_ns;++s)
            sigma_at(t,s,pair.k)-=tensors::thc_orbital_sigma_weighted(pair.v,weighted,green_at(t,s,pair.kp))/double(_nk);
        }
      };
      auto screen_q=[&](size_t q,const MatrixXcd& m) {
        ztensor<4> core(_nt,1,Q,Q),frequency(_nw,1,Q,Q);
        core.set_zero();frequency.set_zero();
        for(size_t t=0;t<_nt/2;++t) {
          MatrixXcd bubble=MatrixXcd::Zero(r,r);
          for(const auto& pair:_bubble_pairs[q]) {
            const size_t i=pair.second,j=pair.first;
            // Preserve the source-pair reversal and the plain transpose of G.
            for(size_t s=0;s<_ns;++s) {
              if(!projected_g.empty()) {
                const auto& left=projected_g[((_nt-t-1)*_ns+s)*_nk+i];
                const auto& right=projected_g[(t*_ns+s)*_nk+j];
                bubble-=(_ns==2?1.:2.)/double(_nk)*left.transpose().cwiseProduct(right);
              } else {
                auto Xi=_factors->X(i),Xj=_factors->X(j);
                MatrixXcd left=Xi*matrix(g(_nt-t-1,s,i))*Xi.adjoint();
                MatrixXcd right=Xj*matrix(g(t,s,j))*Xj.adjoint();
                bubble-=(_ns==2?1.:2.)/double(_nk)*left.transpose().cwiseProduct(right);
              }
            }
          }
          // Sum spins before compression. M^H(.)M commutes with Hermitian
          // symmetrization, and the reflected tau row is an exact copy.
          MatrixXcd compressed=m.adjoint()*bubble*m;
          matrix(core(t,0))=0.5*(compressed+compressed.adjoint()).eval();
          matrix(core(_nt-t-1,0))=matrix(core(t,0));
        }
        _ft.tau_f_to_w_b(core,frequency,0,_nw,true);
        for(size_t w=0;w<_nw;++w)matrix(frequency(w,0))=tensors::thc_screened_core(matrix(frequency(w,0)));
        _ft.w_b_to_tau_f(frequency,core,0,_nt,true);
        return core;
      };
      auto project_time=[&](size_t t) {
        thc_cpu_component_timer timer(_profile?&_component_seconds:nullptr,"streamed_projection");
        std::vector<MatrixXcd> values;
        if(projected_g.empty()) {
          values.reserve(_ns*_nk);
          for(size_t s=0;s<_ns;++s)for(size_t k=0;k<_nk;++k) {
            auto X=_factors->X(k);
            values.emplace_back(X*green_at(t,s,k)*X.adjoint());
          }
        }
        return values;
      };
      auto point_fields=[&]() {
        std::vector<MatrixXcd> values;
        for(size_t sk=0;sk<_ns*_nk;++sk)values.emplace_back(MatrixXcd::Zero(r,r));
        return values;
      };
      auto accumulate=[&](size_t t,size_t q,const MatrixXcd& wc,const std::vector<MatrixXcd>& local,
                          std::vector<MatrixXcd>& point_sigma) {
        thc_cpu_component_timer timer(_profile?&_component_seconds:nullptr,"sigma_hadamard");
        for(const auto& pair:_pairs[q]) {
          const size_t k=pair.first,kp=pair.second;
          for(size_t s=0;s<_ns;++s) {
            const auto& projected=projected_g.empty()?local[s*_nk+kp]:projected_g[(t*_ns+s)*_nk+kp];
            point_sigma[s*_nk+k]+=projected.cwiseProduct(wc)/double(_nk);
          }
        }
      };
      auto backproject=[&](size_t t,const std::vector<MatrixXcd>& point_sigma) {
        thc_cpu_component_timer timer(_profile?&_component_seconds:nullptr,"sigma_backproject");
        for(size_t s=0;s<_ns;++s)for(size_t k=0;k<_nk;++k) {
          auto X=_factors->X(k);
          sigma_at(t,s,k)-=X.adjoint()*point_sigma[s*_nk+k]*X;
        }
      };
      if(retain) {
        std::vector<MatrixXcd> cores;
        std::vector<ztensor<4>> response;
        cores.reserve(qs.size());response.reserve(qs.size());
        for(size_t q:qs) {
          cores.emplace_back(_factors->M(q));
          response.emplace_back(_nt,1,Q,Q);response.back().set_zero();
        }
        std::vector<std::complex<double>*> bubble_response_data;
        for(auto& core:response)bubble_response_data.push_back(core.data());
        parallel_tau(_nt/2,[&](size_t t) {
          // Stream just the two projected time slices when the full G cache
          // does not fit. Every owned q borrows these same complete k fields.
          auto left=project_time(_nt-t-1),right=project_time(t);
          for(size_t iq=0;iq<qs.size();++iq) {
            MatrixXcd bubble=MatrixXcd::Zero(r,r);
            {
            thc_cpu_component_timer timer(_profile?&_component_seconds:nullptr,"bubble_correlate");
            for(const auto& pair:_bubble_pairs[qs[iq]]) {
              const size_t i=pair.second,j=pair.first;
              for(size_t s=0;s<_ns;++s) {
                const auto& first=projected_g.empty()?left[s*_nk+i]:projected_g[((_nt-t-1)*_ns+s)*_nk+i];
                const auto& second=projected_g.empty()?right[s*_nk+j]:projected_g[(t*_ns+s)*_nk+j];
                bubble-=(_ns==2?1.:2.)/double(_nk)*first.transpose().cwiseProduct(second);
              }
            }
            }
            thc_cpu_component_timer timer(_profile?&_component_seconds:nullptr,"bubble_compress");
            MatrixXcd compressed=cores[iq].adjoint()*bubble*cores[iq];
            MMatrixXcd current(bubble_response_data[iq]+t*Q*Q,Q,Q);
            MMatrixXcd reflected(bubble_response_data[iq]+(_nt-t-1)*Q*Q,Q,Q);
            current=0.5*(compressed+compressed.adjoint()).eval();reflected=current;
          }
        });
        // Projected half-time fields are released before transform/LU scratch.
        // Cores and Q-space histories stay resident across these one-q solves.
        {
          thc_cpu_component_timer timer(_profile?&_component_seconds:nullptr,"screen_transform_solve");
          ztensor<4> frequency(_nw,1,Q,Q);frequency.set_zero();
          for(auto& core:response) {
            _ft.tau_f_to_w_b(core,frequency,0,_nw,true);
            for(size_t w=0;w<_nw;++w)matrix(frequency(w,0))=tensors::thc_screened_core(matrix(frequency(w,0)));
            _ft.w_b_to_tau_f(frequency,core,0,_nt,true);
          }
        }
        if(_orbital)for(size_t iq=0;iq<qs.size();++iq)
          if(_vertices[qs[iq]].empty())_vertices[qs[iq]]=build_vertices(qs[iq],cores[iq]);
        std::vector<const std::complex<double>*> response_data;
        for(const auto& core:response)response_data.push_back(core.data());
        std::exception_ptr failure;
        std::mutex failure_mutex;
        auto run_worker=[&](size_t worker) {
        for(size_t t=_nt*worker/workers;t<_nt*(worker+1)/workers;++t) {
          try {
            if(_orbital) {
              for(size_t iq=0;iq<qs.size();++iq)
                orbital_tau(t,CMMatrixXcd(response_data[iq]+t*Q*Q,Q,Q),_vertices[qs[iq]]);
            } else {
              auto local=project_time(t),point_sigma=point_fields();
              for(size_t iq=0;iq<qs.size();++iq) {
                MatrixXcd wc;
                {thc_cpu_component_timer timer(_profile?&_component_seconds:nullptr,"sigma_expand");
                  wc=cores[iq]*CMMatrixXcd(response_data[iq]+t*Q*Q,Q,Q)*cores[iq].adjoint();}
                accumulate(t,qs[iq],wc,local,point_sigma);
              }
              backproject(t,point_sigma);
            }
          }catch(...) {
            std::lock_guard<std::mutex> guard(failure_mutex);
            if(!failure)failure=std::current_exception();
          }
        }
        };
        // Scope threading to THC; do not activate legacy ndarray/OpenMP loops.
        std::vector<std::thread> threads;
        try {
          for(size_t worker=1;worker<workers;++worker)threads.emplace_back(run_worker,worker);
        }catch(...) {
          for(auto& thread:threads)thread.join();
          throw;
        }
        run_worker(0);
        for(auto& thread:threads)thread.join();
        if(failure)std::rethrow_exception(failure);
      } else {
        // Bounded fallback: retain one q in Q space and combine its kp terms.
        for(size_t q:qs) {
          MatrixXcd m=_factors->M(q);
          auto response=screen_q(q,m);
          if(_orbital) {
            auto pairs=build_vertices(q,m);
            for(size_t t=0;t<_nt;++t)orbital_tau(t,matrix(response(t,0)),pairs);
            continue;
          }
          for(size_t t=0;t<_nt;++t) {
            MatrixXcd wc=m*matrix(response(t,0))*m.adjoint();
            if(time_batch) {
              auto local=project_time(t),point_sigma=point_fields();
              accumulate(t,q,wc,local,point_sigma);
              backproject(t,point_sigma);
            } else {
              // The smallest budget also streams Sigma one (s,k) at a time.
              for(size_t s=0;s<_ns;++s)for(size_t k=0;k<_nk;++k) {
                MatrixXcd point_sigma=MatrixXcd::Zero(r,r);
                for(size_t kp=0;kp<_nk;++kp) {
                  if(_factors->transfer(k,kp)!=q)continue;
                  if(!projected_g.empty())point_sigma+=projected_g[(t*_ns+s)*_nk+kp].cwiseProduct(wc)/double(_nk);
                  else {
                    auto Xp=_factors->X(kp);
                    MatrixXcd projected=Xp*matrix(g(t,s,kp))*Xp.adjoint();
                    point_sigma+=projected.cwiseProduct(wc)/double(_nk);
                  }
                }
                auto X=_factors->X(k);
                matrix(sigma(t,s,k))-=X.adjoint()*point_sigma*X;
              }
            }
          }
        }
      }
    }
    size_t _n,_ns,_nk,_nt,_nw;
    bool _fft=false,_auxiliary=false,_orbital=false,_reuse_fft=true;
    bool _profile=false;
    std::map<std::string,double> _component_seconds;
    std::vector<std::vector<std::pair<size_t,size_t>>> _pairs;
    std::vector<std::vector<std::pair<size_t,size_t>>> _bubble_pairs;
    size_t _cpu_threads=1;
    std::vector<std::vector<orbital_pair>> _vertices;
    std::string _screening="auto";
    size_t _workspace_bytes=0;
    const grids::transformer_t& _ft;
    std::shared_ptr<integrals::thc_factor_data> _factors;
  };
}
#endif
