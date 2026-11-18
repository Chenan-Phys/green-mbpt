#include "green/mbpt/spatial_cavity.h"
#include <green/utils/mpi_utils.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace green::mbpt::spatial_cavity {
  namespace {
    void require(bool ok, const std::string& message) {
      if (!ok) throw std::runtime_error("Spatial cavity: " + message);
    }
    void hermitian(const MatrixXcd& a, const std::string& name) {
      require(a.allFinite() && (a-a.adjoint()).norm() < 1e-11*(1+a.norm()), name+" must be finite and Hermitian");
    }
    MatrixXd symmetric_real(const MatrixXcd& a, const std::string& name) {
      require(a.allFinite() && a.imag().norm() < 1e-7*(1+a.norm()), name+" has a nonphysical imaginary equal-time component");
      require((a-a.transpose()).norm() < 1e-7*(1+a.norm()), name+" is not symmetric at equal time");
      return .5*(a.real()+a.real().transpose()).eval();
    }
    dtensor<2> tensor_real(const MatrixXd& a) {
      dtensor<2> t(a.rows(), a.cols());
      std::copy(a.data(), a.data()+a.size(), t.data());
      return t;
    }
    std::vector<double> vector_real(const Eigen::VectorXd& a) { return {a.data(), a.data()+a.size()}; }
  }

  state& instance() { static state s; return s; }

  MatrixXcd state::density(const ztensor<5>& g) const {
    MatrixXcd d = MatrixXcd::Zero(nao, nao);
    const double prefactor = ns == 1 ? 2.0 : 1.0;
    for (size_t s=0; s<ns; ++s) d -= prefactor*matrix(g(g.shape()[0]-1, s, 0));
    return .5*(d+d.adjoint()).eval();
  }

  void state::observe_green(const ztensor<5>& g) {
    if(!enabled) return;
    green_residual=1.;
    if(previous_green.size()==g.size()) {
      green_residual=0.;
      for(size_t i=0;i<g.size();++i) green_residual=std::max(green_residual,std::abs(g.data()[i]-previous_green.data()[i]));
    }
    previous_green.resize(g.shape());
    previous_green << g;
  }

  void state::initialize(const params::params& p, const symmetry::brillouin_zone_utils& bz, const ztensor<4>& overlap) {
    *this = state{};
    h5pp::archive ar(p["input_file"]);
    if (!ar.has_group("QED")) {
      require(!ar.has_group("Leads"),"equilibrium reservoirs require the finite-system spatial QED interface");
      return;
    }
    int schema = 0, finite_system = 0;
    std::string model;
    ar["QED/schema"] >> schema;
    ar["QED/model"] >> model;
    ar["QED/finite_system"] >> finite_system;
    require(schema == 2 && model == "current_quadratic", "expected schema=2, model=current_quadratic");
    require(finite_system == 1 && bz.nk()==1 && bz.nq()==1 && bz.ink()==1 && bz.inq()==1,
            "only an explicit finite Gamma-point system is supported");
    require(utils::mpi_context::context().global_size == 1, "photon calculations currently require one MPI rank");
    ar["params/nao"] >> nao;
    size_t nso;
    ar["params/nso"] >> nso;
    ar["params/ns"] >> ns;
    ar["params/NQ"] >> coulomb_aux;
    require(nao==nso && (ns==1 || ns==2), "only scalar one/two-spin calculations are supported");
    require(p["kernel"].as<kernel_type>()==CPU, "only the CPU kernel is supported");
    const auto method=p["scf_type"].as<scf_type>();
    require(method==HF || method==GW || method==GF2, "expected HF, GW or GF2");
    require(!p["P_sp"].as<bool>() && !p["Sigma_sp"].as<bool>(), "double precision is required");
    require(bz.k_symmetry().tr_conj_list()[0]==0 && bz.q_symmetry().tr_conj_list()[0]==0,
            "antiunitary reduction is incompatible with this finite-system interface");
    MatrixXcd identity=MatrixXcd::Identity(nao,nao);
    for (size_t s=0;s<ns;++s) require((matrix(overlap(s,0))-identity).norm()<1e-11,
                                    "orthonormal orbitals are required; transform all operators first");
    MatrixXcd aux_u(coulomb_aux,coulomb_aux);
    bz.q_symmetry().q_sym_transform_p0(aux_u,0);
    require((aux_u-MatrixXcd::Identity(coulomb_aux,coulomb_aux)).norm()<1e-11,
            "identity auxiliary symmetry is required");
    std::vector<double> w;
    ar["QED/omega"] >> w;
    modes=w.size();
    require(modes>0 && modes<=4,"provide between one and four modes");
    omega=Eigen::Map<Eigen::VectorXd>(w.data(),modes);
    require(omega.allFinite() && omega.minCoeff()>0,"all bare photon frequencies must be positive");
    dtensor<4> jt;
    dtensor<5> kt;
    ar["QED/current"] >> jt;
    ar["QED/quadratic"] >> kt;
    require(jt.shape()==std::array<size_t,4>{modes,nao,nao,2},"current shape must be (mode,orb,orb,2)");
    require(kt.shape()==std::array<size_t,5>{modes,modes,nao,nao,2},"quadratic shape must be (mode,mode,orb,orb,2)");
    const auto* jp=reinterpret_cast<const std::complex<double>*>(jt.data());
    const auto* kp=reinterpret_cast<const std::complex<double>*>(kt.data());
    for(size_t a=0;a<modes;++a) {
      current.emplace_back(CMMatrixXcd(jp+a*nao*nao,nao,nao));
      hermitian(current.back(),"current");
      for(size_t b=0;b<modes;++b) {
        quadratic.emplace_back(CMMatrixXcd(kp+(a*modes+b)*nao*nao,nao,nao));
        hermitian(quadratic.back(),"quadratic");
      }
    }
    for(size_t a=0;a<modes;++a) for(size_t b=0;b<modes;++b)
      require((quadratic[a*modes+b]-quadratic[b*modes+a]).norm()<1e-11,"quadratic mode indices must be symmetric");
    for(size_t a=0;a<modes;++a) for(size_t b=a;b<modes;++b) {
      probes.emplace_back(a,b);
      probe_scale.push_back(std::sqrt(2/std::sqrt(omega(a)*omega(b))));
    }
    beta=p["BETA"];
    if (ar.has_group("Leads")) {
      std::string kind;
      ar["Leads/model"] >> kind;
      require(kind=="local_chain" && !p["const_density"].as<bool>(),
              "local-chain equilibrium reservoirs require fixed chemical potential");
      ar["Leads/sites"] >> lead_sites;
      ar["Leads/onsite"] >> lead_onsite;
      ar["Leads/hopping"] >> lead_hopping;
      ar["Leads/coupling"] >> lead_coupling;
      const size_t nl=lead_sites.size();
      require(nl && lead_onsite.size()==nl && lead_hopping.size()==nl && lead_coupling.size()==nl,
              "each terminal needs an orbital, onsite energy, chain hopping and tunnel amplitude");
      for(size_t a=0;a<nl;++a) {
        require(lead_sites[a]>=0 && size_t(lead_sites[a])<nao && std::isfinite(lead_onsite[a])
                && std::isfinite(lead_hopping[a]) && lead_hopping[a]>=0 && std::isfinite(lead_coupling[a]),
                "invalid local-chain reservoir parameters");
      }
    }
    energy_threshold=p["E_thr"];
    second_born=method==GF2;
    correlated=method==GW || second_born;
    q=Eigen::VectorXd::Zero(modes);
    covariance_q=MatrixXd::Identity(modes,modes);
    covariance_p=covariance_q;
    mixed_kq=MatrixXd::Zero(probes.size(),modes);
    enabled=true;
  }

  void state::embed_leads(MMatrixXcd inverse, std::complex<double> z) const {
    for(size_t a=0;a<lead_sites.size();++a) {
      const auto x=z-lead_onsite[a];
      const double t=lead_hopping[a];
      // Principal roots give the decaying branch on both Matsubara half axes.
      // A zero chain hopping leaves one explicit, discrete terminal level.
      const auto surface=t==0 ? 1./x : 2./(x+std::sqrt(x-2*t)*std::sqrt(x+2*t));
      inverse(lead_sites[a],lead_sites[a])-=lead_coupling[a]*lead_coupling[a]*surface;
    }
  }

  void state::add_static(const ztensor<4>& dm, ztensor<4>& sigma) {
    if(!enabled) return;
    old_q=q; old_sq=covariance_q; old_sp=covariance_p;
    old_kq=mixed_kq; old_mixed_energy=mixed_energy;
    const MatrixXcd previous_rho=rho;
    rho=MatrixXcd::Zero(nao,nao);
    // hf_solver passes a POSITIVE physical density, including the restricted
    // factor of two. Do not change its sign or count that spin factor again.
    for(size_t s=0;s<ns;++s) rho+=matrix(dm(s,0));
    // IR endpoint reconstruction has roundoff beyond the stricter input-
    // operator tolerance. Reject a material defect before symmetrizing it.
    require(rho.allFinite() && (rho-rho.adjoint()).norm()<1e-8*(1+rho.norm()),
            "density is not Hermitian within IR precision");
    rho=.5*(rho+rho.adjoint()).eval();
    density_residual=previous_rho.rows()==rho.rows() ? (rho-previous_rho).norm()/(1+rho.norm()) : 1.;
    kappa=MatrixXd::Zero(modes,modes);
    Eigen::VectorXd j(modes), z=Eigen::VectorXd::Zero(modes);
    for(size_t a=0;a<modes;++a) {
      j(a)=(current[a]*rho).trace().real();
      for(size_t b=0;b<modes;++b) kappa(a,b)=(quadratic[a*modes+b]*rho).trace().real();
    }
    for(size_t p=0;p<probes.size();++p) {
      auto [a,b]=probes[p];
      z(a)+=mixed_kq(p,b);
      if(a!=b) z(b)+=mixed_kq(p,a);
    }
    stiffness=omega.asDiagonal().toDenseMatrix()+4*kappa;
    Eigen::SelfAdjointEigenSolver<MatrixXd> normal(omega.cwiseSqrt().asDiagonal()*stiffness*omega.cwiseSqrt().asDiagonal());
    require(normal.info()==Eigen::Success && normal.eigenvalues().minCoeff()>0,"unstable quadratic photon Hamiltonian");
    frequencies=normal.eigenvalues().cwiseSqrt();
    Eigen::VectorXd coth(modes);
    for(size_t a=0;a<modes;++a) coth(a)=1/std::tanh(.5*beta*frequencies(a));
    auto u=normal.eigenvectors();
    bare_q=omega.cwiseSqrt().asDiagonal()*u*(coth.array()/frequencies.array()).matrix().asDiagonal()*u.transpose()*omega.cwiseSqrt().asDiagonal();
    bare_p=omega.cwiseSqrt().cwiseInverse().asDiagonal()*u*(coth.array()*frequencies.array()).matrix().asDiagonal()*u.transpose()*omega.cwiseSqrt().cwiseInverse().asDiagonal();
    q=stiffness.ldlt().solve(-2*(j+2*z));
    if(!correlated || !have_covariance) {
      covariance_q=bare_q; covariance_p=bare_p;
      mixed_energy=0; mixed_kq.setZero();
    }
    static_sigma=MatrixXcd::Zero(nao,nao);
    vertices.clear();
    for(size_t a=0;a<modes;++a) {
      static_sigma+=q(a)*current[a];
      MatrixXcd je=current[a];
      for(size_t b=0;b<modes;++b) {
        static_sigma+=(q(a)*q(b)+covariance_q(a,b))*quadratic[a*modes+b];
        je+=2*q(b)*quadratic[a*modes+b];
      }
      vertices.push_back(std::sqrt(2/omega(a))*je);
    }
    for(size_t p=0;p<probes.size();++p) {
      auto [a,b]=probes[p];
      vertices.push_back(probe_scale[p]*quadratic[a*modes+b]);
    }
    for(size_t s=0;s<ns;++s) matrix(sigma(s,0))+=static_sigma;
    if(!correlated) finish_update();
  }

  MatrixXd state::bare_covariance(double nu) const {
    MatrixXd inverse=2*kappa;
    for(size_t a=0;a<modes;++a) inverse(a,a)+=(nu*nu+omega(a)*omega(a))/(2*omega(a));
    return inverse.ldlt().solve(MatrixXd::Identity(modes,modes));
  }

  MatrixXcd state::interaction(double nu) const {
    size_t nq=coulomb_aux+extra_aux();
    MatrixXcd b=MatrixXcd::Zero(nq,nq);
    b.topLeftCorner(coulomb_aux,coulomb_aux).setIdentity();
    b.block(coulomb_aux,coulomb_aux,modes,modes)=
        -((omega/2).cwiseSqrt().asDiagonal()*bare_covariance(nu)*(omega/2).cwiseSqrt().asDiagonal()).cast<std::complex<double>>();
    // The remaining K vertices are response probes only; B=0, so they do not
    // introduce any two-photon exchange self-energy into this approximation.
    return b;
  }

  void state::update_photons(const std::vector<MatrixXcd>& chi, const grids::transformer_t& ft) {
    const size_t nw=chi.size(), nt=ft.sd().repn_fermi().nts();
    ztensor<4> dq(nw,1,modes,modes), dp(nw,1,modes,modes), xx(nw,1,1,1), zz(nw,1,probes.size(),modes);
    photon_w.resize(nw,1,modes,modes);
    const MatrixXd s=(omega/2).cwiseSqrt().asDiagonal();
    const MatrixXd wi=omega.cwiseInverse().asDiagonal();
    for(size_t n=0;n<nw;++n) {
      double nu=ft.wsample_bose()(n);
      const MatrixXcd c0=bare_covariance(nu).cast<std::complex<double>>();
      const MatrixXcd x=s.cast<std::complex<double>>()*chi[n].block(coulomb_aux,coulomb_aux,modes,modes)*s.cast<std::complex<double>>();
      const MatrixXcd delta=-c0*x*c0;
      matrix(dq(n,0))=delta;
      matrix(dp(n,0))=-nu*nu*wi.cast<std::complex<double>>()*delta*wi.cast<std::complex<double>>();
      matrix(photon_w(n,0))=c0+delta;
      xx(n,0,0,0)=(c0*x).trace();
      for(size_t p=0;p<probes.size();++p) {
        // <delta K delta Q> = Matsubara sum chi_(K,J') C0.
        matrix(zz(n,0)).row(p)=chi[n].block(coulomb_aux+modes+p,coulomb_aux,1,modes)*s.cast<std::complex<double>>()*c0/probe_scale[p];
      }
    }
    ztensor<4> tq(nt,1,modes,modes), tp(nt,1,modes,modes), tx(nt,1,1,1), tz(nt,1,probes.size(),modes);
    ft.w_b_to_tau_f(dq,tq,0,nt,true);
    ft.w_b_to_tau_f(dp,tp,0,nt,true);
    ft.w_b_to_tau_f(xx,tx,0,nt,true);
    ft.w_b_to_tau_f(zz,tz,0,nt,true);
    covariance_q=bare_q+symmetric_real(matrix(tq(0,0)),"Q covariance correction");
    covariance_p=bare_p+symmetric_real(matrix(tp(0,0)),"P covariance correction");
    require(Eigen::SelfAdjointEigenSolver<MatrixXd>(covariance_q).eigenvalues().minCoeff()>0 &&
            Eigen::SelfAdjointEigenSolver<MatrixXd>(covariance_p).eigenvalues().minCoeff()>0,"nonpositive photon covariance");
    require(std::abs(tx(0,0,0,0).imag())<1e-7,"nonreal mixed interaction energy");
    mixed_energy=tx(0,0,0,0).real();
    require(matrix(tz(0,0)).imag().norm()<1e-7,"nonreal mixed quadratic/current response");
    mixed_kq=matrix(tz(0,0)).real();
    have_covariance=true;
    finish_update();
  }

  void state::finish_update() {
    residual=std::max({(q-old_q).norm()/(1+q.norm()),
      (covariance_q-old_sq).norm()/(1+covariance_q.norm()),
      (covariance_p-old_sp).norm()/(1+covariance_p.norm()),
      (mixed_kq-old_kq).norm()/(1+mixed_kq.norm()),
      std::abs(mixed_energy-old_mixed_energy)/(1+std::abs(mixed_energy))});
  }

  double state::convergence(const ztensor<5>& g) const {
    if(!enabled) return 0;
    const double drho=std::max(density_residual,(density(g)-rho).norm()/(1+rho.norm()));
    return energy_threshold*std::max({residual/1e-8,drho/1e-7,green_residual/1e-9});
  }

  double state::photon_energy() const {
    return .25*(omega.array()*(q.array().square()+covariance_q.diagonal().array()+covariance_p.diagonal().array()-2)).sum();
  }

  double state::energy_correction(const ztensor<5>& g) const {
    if(!enabled) return 0;
    // The electronic GM expression contains only half of the one-body photon
    // tadpole and half of <delta J' delta Q>. Restore both and the photon H.
    return .5*(density(g)*static_sigma).trace().real()+.5*mixed_energy+photon_energy();
  }

  void state::dump(h5pp::archive& ar,const std::string& path,const ztensor<5>& g,double raw_energy) const {
    if(!enabled) return;
    ar[path+(lead_sites.empty()?"/Energy_total_QED":"/Energy_central_QED")] << raw_energy+energy_correction(g);
    if(!lead_sites.empty()) ar[path+"/Leads/terminal_count"] << int(lead_sites.size());
    ar[path+"/QED/energy_GM_correction"] << energy_correction(g);
    ar[path+"/QED/photon_energy"] << photon_energy();
    ar[path+"/QED/mixed_linear_energy"] << mixed_energy;
    ar[path+"/QED/field_residual"] << residual;
    ar[path+"/QED/density_residual"] << density_residual;
    ar[path+"/QED/green_residual"] << green_residual;
    ar[path+"/QED/q_mean"] << vector_real(q);
    ar[path+"/QED/bare_omega"] << vector_real(omega);
    ar[path+"/QED/tadpole_frequencies"] << vector_real(frequencies);
    ar[path+"/QED/covariance_Q"] << tensor_real(covariance_q);
    ar[path+"/QED/covariance_P"] << tensor_real(covariance_p);
    ar[path+"/QED/mixed_KQ"] << tensor_real(mixed_kq);
    ztensor<2> ss(nao,nao); matrix(ss)=static_sigma;
    ar[path+"/QED/static_sigma"] << ss;
    if(correlated && have_covariance) ar[path+"/QED/C_omega"] << photon_w;
  }
}
