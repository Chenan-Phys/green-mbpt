/* Molecular Pauli-Fierz extension; see docs/cavity.md for conventions. */
#ifndef GREEN_MBPT_CAVITY_H
#define GREEN_MBPT_CAVITY_H

#include <cmath>
#include <limits>
#include <green/h5pp/archive.h>
#include "common_defs.h"

namespace green::mbpt::cavity {
  struct configuration {
    bool enabled = false;
    double omega = 0.0, lambda = 0.0, nuclear_dipole = 0.0;
    double photon_energy_correction = 0.0, photon_variance = 1.0;
    double coherent_b = 0.0, bosonic_residual = 0.0;
    MatrixXcd dipole, second_moment;
    ztensor<4> chi_w, photon_delta_w;
  };
  // Initial molecular CPU implementation deliberately permits one MPI rank.
  inline configuration state;

  inline void initialize(const params::params& p, size_t nao, size_t nso, size_t nk) {
    state = configuration{};
    h5pp::archive ar(p["input_file"], "r");
    if (!ar.has_group("QED")) return;
    int schema = 0, molecular = 0;
    ar["QED/schema"] >> schema;
    ar["QED/molecular"] >> molecular;
    if (schema != 1 || molecular != 1 || nk != 1 || nao != nso)
      throw std::invalid_argument("QED schema 1 requires a scalar molecular input with one k point");
    if (utils::context().global_size != 1 || p["kernel"].as<kernel_type>() != CPU)
      throw std::invalid_argument("Molecular QED currently requires one MPI rank and the CPU kernel");
    if (p["scf_type"].as<scf_type>() != HF && p["scf_type"].as<scf_type>() != GW)
      throw std::invalid_argument("QED input currently supports HF and joint GW; electronic GF2 is not QED-GF2");
    if (p["q0_treatment"].as<sigma_q0_treatment_e>() != ignore_G0)
      throw std::invalid_argument("Molecular QED must use q0_treatment=ignore_G0");
    ar["QED/omega_hartree"] >> state.omega;
    ar["QED/lambda_au"] >> state.lambda;
    ar["QED/nuclear_dipole"] >> state.nuclear_dipole;
    dtensor<2> dipole, second;
    ar["QED/dipole"] >> dipole;
    ar["QED/second_moment"] >> second;
    if (dipole.shape()[0] != nao || dipole.shape()[1] != nao ||
        second.shape()[0] != nao || second.shape()[1] != nao)
      throw std::invalid_argument("QED operator dimensions do not match the AO basis");
    state.dipole = matrix(dipole).cast<std::complex<double>>();
    state.second_moment = matrix(second).cast<std::complex<double>>();
    if (!(state.omega > 0) || !std::isfinite(state.omega) || !std::isfinite(state.lambda) ||
        !state.dipole.allFinite() || !state.second_moment.allFinite() ||
        (state.dipole-state.dipole.adjoint()).norm() > 1e-10 ||
        (state.second_moment-state.second_moment.adjoint()).norm() > 1e-10)
      throw std::invalid_argument("Nonfinite, non-Hermitian or invalid QED parameters");
    state.enabled = true;
    const double beta = p["BETA"];
    const double nb = 1.0 / std::expm1(beta * state.omega);
    state.photon_variance = 1.0 + 2.0 * nb;
    state.photon_energy_correction = state.omega * nb;
    std::cout << "QED: single-mode Pauli-Fierz, full dipole DSE; "
              << "joint Coulomb-photon GW screening; omega=" << state.omega
              << " Ha, lambda=" << state.lambda << " au\n";
  }

  inline void add_static(const ztensor<4>& density, ztensor<4>& sigma) {
    if (!state.enabled) return;
    const size_t ns = density.shape()[0];
    const double per_spin = ns == 1 ? 0.5 : 1.0;
    const double l2 = state.lambda * state.lambda;
    double mu = state.nuclear_dipole;
    for (size_t s = 0; s < ns; ++s) {
      const MatrixXcd dm = matrix(density(s, 0));
      // The direct dipole Hartree term cancels the stationary coherent mean.
      matrix(sigma(s, 0)) += 0.5*l2*state.second_moment
        - l2*per_spin*(state.dipole*dm*state.dipole);
      mu += (state.dipole*dm).trace().real();
    }
    state.coherent_b = state.lambda*mu/std::sqrt(2.0*state.omega);
  }

  inline double static_energy_correction(const ztensor<5>& g) {
    if (!state.enabled) return 0.0;
    const size_t ns = g.shape()[1], last = g.shape()[0]-1;
    const double spin = ns == 1 ? 2.0 : 1.0;
    double value = 0.0;
    for (size_t s = 0; s < ns; ++s)
      value -= 0.25*state.lambda*state.lambda*spin*
        (state.second_moment*matrix(g(last,s,0))).trace().real();
    // Native GM weights a one-body term placed in Sigma_inf by one half.
    return value;
  }

  inline void dump(h5pp::archive& ar, const std::string& prefix, double total) {
    if (!state.enabled) return;
    ar[prefix+"/Energy_total_QED"] << total;
    ar[prefix+"/QED/Photon_energy_correction"] << state.photon_energy_correction;
    ar[prefix+"/QED/Photon_variance"] << state.photon_variance;
    ar[prefix+"/QED/Coherent_b"] << state.coherent_b;
    ar[prefix+"/QED/Bosonic_residual"] << state.bosonic_residual;
    ar["QED/omega_hartree"] << state.omega;
    ar["QED/lambda_au"] << state.lambda;
    if (state.chi_w.size()) {
      ar[prefix+"/QED/Chi_dipole_w"] << state.chi_w;
      ar[prefix+"/QED/Photon_delta_w"] << state.photon_delta_w;
    }
  }
}
#endif
