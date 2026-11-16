/* Spatial current/quadratic photon extension of stable GREEN v1.0.0.
 * Distributed under the repository MIT license. See docs/spatial_cavity.md.
 */
#ifndef GREEN_MBPT_SPATIAL_CAVITY_H
#define GREEN_MBPT_SPATIAL_CAVITY_H
#include "common_defs.h"
#include <green/grids/transformer_t.h>
#include <green/symmetry/symmetry.h>
#include <green/h5pp/archive.h>

namespace green::mbpt::spatial_cavity {
  // Process-local: enabled calculations currently require one MPI rank. A fresh
  // Dyson construction resets this object, including when input has no QED group.
  struct state {
    bool enabled = false, correlated = false, have_covariance = false;
    size_t modes = 0, nao = 0, ns = 0, coulomb_aux = 0;
    double beta = 0, energy_threshold = 0, residual = 0, density_residual = 0, green_residual = 0, mixed_energy = 0;
    Eigen::VectorXd omega, q, frequencies, old_q;
    MatrixXd kappa, stiffness, covariance_q, covariance_p, bare_q, bare_p;
    MatrixXd old_sq, old_sp, mixed_kq, old_kq;
    double old_mixed_energy = 0;
    MatrixXcd rho, static_sigma;
    std::vector<MatrixXcd> current, quadratic, vertices;
    std::vector<std::pair<size_t, size_t>> probes;
    std::vector<double> probe_scale;
    ztensor<4> photon_w;
    ztensor<5> previous_green;

    size_t extra_aux() const { return enabled ? modes + probes.size() : 0; }
    MatrixXcd density(const ztensor<5>& g) const;
    void observe_green(const ztensor<5>& g);
    void initialize(const params::params& p, const symmetry::brillouin_zone_utils& bz, const ztensor<4>& overlap);
    void add_static(const ztensor<4>& physical_dm, ztensor<4>& sigma);
    MatrixXd bare_covariance(double nu) const;
    MatrixXcd interaction(double nu) const;
    void update_photons(const std::vector<MatrixXcd>& responses, const grids::transformer_t& ft);
    void finish_update();
    double convergence(const ztensor<5>& g) const;
    double photon_energy() const;
    double energy_correction(const ztensor<5>& g) const;
    void dump(h5pp::archive& ar, const std::string& path, const ztensor<5>& g, double raw_energy) const;
  };
  state& instance();
}
#endif
