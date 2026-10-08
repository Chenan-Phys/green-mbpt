#include <green/mbpt/mbpt_run.h>
#include <green/mbpt/kernel_factory.h>
#include <green/grids.h>
#include <Eigen/Eigenvalues>
#include <mpi.h>
#include <chrono>
#include <iomanip>

int main(int argc,char** argv) {
  MPI_Init(&argc,&argv); green::utils::context();
  int result=0;
  try {
    using namespace green::mbpt;
    green::params::params p("Frozen THC solver comparison");
    green::sc::define_parameters(p); green::grids::define_parameters(p);
    green::symmetry::define_parameters(p); define_parameters(p);
    p.define<std::string>("probe_output","New output HDF5 file");
    p.define<size_t>("probe_iterations","Repeated GW calls on one kernel to verify cache reuse",1);
    if(!p.parse(argc,argv)) throw std::runtime_error("probe parameters missing");
    if(!p["probe_iterations"].as<size_t>() || p["probe_iterations"].as<size_t>()>8)
      throw std::runtime_error("probe_iterations must be in 1..8");
    check_input(p);
    green::symmetry::brillouin_zone_utils bz(p);
    green::grids::transformer_t ft(p);
    size_t n,ns,NQ; double madelung;
    green::h5pp::archive input(p["input_file"]);
    input["params/nao"]>>n; input["params/ns"]>>ns; input["params/NQ"]>>NQ; input["HF/madelung"]>>madelung;
    dtensor<5> packedS,packedH;
    input["HF/S-k"]>>packedS; input["HF/H-k"]>>packedH; input.close();
    size_t nk=bz.nk(),ink=bz.ink(),nt=ft.sd().repn_fermi().nts();
    ztensor<4> Sfull(ns,nk,n,n),Hfull(ns,nk,n,n),S(ns,ink,n,n),dm(ns,ink,n,n);
    Sfull<<packedS.view<std::complex<double>>().reshape(ns,nk,n,n);
    Hfull<<packedH.view<std::complex<double>>().reshape(ns,nk,n,n);
    for(size_t s=0;s<ns;++s) S(s)<<bz.full_to_ibz(Sfull(s));
    green::utils::shared_object<ztensor<5>> g(nt,ns,ink,n,n),sigma(nt,ns,ink,n,n);
    g.fence();
    if(!green::utils::context().node_rank) {
      for(size_t s=0;s<ns;++s) for(size_t k=0;k<ink;++k) {
        size_t fk=bz.k_symmetry().full_point(k);
        Eigen::GeneralizedSelfAdjointEigenSolver<MatrixXcd> eig(matrix(Hfull(s,fk)),matrix(Sfull(s,fk)));
        const double beta=p["BETA"].as<double>();
        // A fixed one-body G with a deterministic chemical potential. Both
        // representations receive these same complete spectral coefficients.
        Eigen::VectorXd eps=eig.eigenvalues(); eps.array()-=eps.mean();
        for(size_t t=0;t<nt;++t) {
          const double tau=ft.sd().repn_fermi().tsample()(t);
          if(tau<0 || tau>beta) throw std::runtime_error("frozen G physical tau outside [0,beta]");
          Eigen::VectorXd diag(n);
          for(size_t a=0;a<n;++a) diag[a]=eps[a]>=0?-std::exp(-tau*eps[a])/(1+std::exp(-beta*eps[a])):
                                                              -std::exp((beta-tau)*eps[a])/(1+std::exp(beta*eps[a]));
          matrix(g.object()(t,s,k))=eig.eigenvectors()*diag.asDiagonal()*eig.eigenvectors().adjoint();
        }
        matrix(dm(s,k))=-(ns==1?2.0:1.0)*matrix(g.object()(nt-1,s,k));
      }
    }
    g.fence();
    // Density is replicated for the HF kernel's MPI work partition.
    if(green::utils::context().node_rank)
      for(size_t s=0;s<ns;++s) for(size_t k=0;k<ink;++k) matrix(dm(s,k))=-(ns==1?2.0:1.0)*matrix(g.object()(nt-1,s,k));
    auto before=std::chrono::steady_clock::now();
    auto [owner,hf]=kernels::hf_kernel_factory::get_kernel(false,p,n,n,ns,NQ,madelung,bz,S);
    auto static_sigma=hf(dm);
    double hf_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-before).count();
    before=std::chrono::steady_clock::now();
    if(p["scf_type"].as<scf_type>()==GW) {
      auto [gwowner,gw]=kernels::gw_kernel_factory::get_kernel(false,p,n,n,ns,NQ,ft,bz,S);
      for(size_t iteration=0;iteration<p["probe_iterations"].as<size_t>();++iteration) {
        auto iteration_before=std::chrono::steady_clock::now();
        gw(g,sigma);
        if(p["probe_iterations"].as<size_t>()>1 && !green::utils::context().global_rank)
          std::cout<<std::setprecision(17)<<"Native THC probe iteration "<<iteration<<" seconds="
            <<std::chrono::duration<double>(std::chrono::steady_clock::now()-iteration_before).count()<<std::endl;
      }
    } else if(p["scf_type"].as<scf_type>()==GF2) {
      gf2_solver gf2(p,ft,bz); gf2.solve(g,static_sigma,sigma);
    } else {
      sigma.fence(); if(!green::utils::context().node_rank) sigma.object().set_zero(); sigma.fence();
    }
    double dynamic_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-before).count();
    if(!green::utils::context().global_rank) {
      std::string output=p["probe_output"];
      if(std::filesystem::exists(output)) throw std::runtime_error("probe output already exists");
      green::h5pp::archive ar(output,"w");
      ar["Sigma1"]<<static_sigma; ar["Sigma_tau"]<<sigma.object(); ar["frozen_G"]<<g.object();
      ar["hf_seconds"]<<hf_seconds; ar["dynamic_seconds"]<<dynamic_seconds; ar.close();
      std::cout<<std::setprecision(17)<<"{\"hf_seconds\":"<<hf_seconds<<",\"dynamic_seconds\":"<<dynamic_seconds<<",\"representation\":\""<<p["interaction_representation"].as<std::string>()<<"\",\"mode\":\""<<p["thc_mode"].as<std::string>()<<"\"}"<<std::endl;
    }
  } catch(const std::exception& e) {std::cerr<<"THC probe: "<<e.what()<<std::endl; result=2;}
  MPI_Finalize(); return result;
}
