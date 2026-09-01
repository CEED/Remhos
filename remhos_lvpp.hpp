#ifndef REMHOS_LVPP_HPP
#define REMHOS_LVPP_HPP
#include "mfem.hpp"
#include "remap.hpp"
#include "legendre.hpp"

namespace mfem
{

inline real_t allreduce(MPI_Comm comm, real_t val, MPI_Op op)
{
   real_t recv;
   MPI_Allreduce(&val, &recv, 1, MPITypeMap<real_t>::mpi_type, op, comm);
   return recv;
}


class Dykstra
{
   MPI_Comm comm = MPI_COMM_NULL;
   StackedFunctional &constraints;
   StackedSharedFunctional *shared_constraints = nullptr;
   MassOperator &mass;
   const Vector &xmin;
   const Vector &xmax;
   Array<LegendreFunction*> legendre_funcs;
   Array<int> offsets;
   real_t tol;
   int max_iter;
   int max_linesearch = 30;
   real_t c1 = 1e-03; // Armijo condition constant

   bool enforce_sum_to_one = false;
   Array<int> sum_to_one_idx_start;
   int sum_to_one_block_size;

   // TODO: Remove this when the input format is finalized.
   bool duplicated_velocity = false;
   Array<int> velocity_idx_start;
   Array<int> velocity_related_constraints;
   Array<int> master_material_idx;
   int velocity_block_size = 0;
public:
   Dykstra(MPI_Comm comm, StackedFunctional &constraints, MassOperator &mass,
           Array<LegendreFunction*> &legendre_funcs_, Array<int> &offsets_,
           const Vector &xmin, const Vector &xmax, real_t tol=1e-10, int max_iter=1000)
      : comm(comm), constraints(constraints), mass(mass)
      , xmin(xmin), xmax(xmax)
      , legendre_funcs(legendre_funcs_), offsets(offsets_)
      , tol(tol), max_iter(max_iter)
   {
      shared_constraints = dynamic_cast<StackedSharedFunctional*>(&constraints);
   }
   void SetAbsTol(real_t tol) { this->tol = tol; }
   void SetMaxIter(int max_iter) { this->max_iter = max_iter; }
   void EnforceSumToOne(const Array<int> &idx_start, const int block_size)
   {
      enforce_sum_to_one = true;
      sum_to_one_idx_start = idx_start;
      sum_to_one_block_size = block_size;
   }

   // TODO: Remove this after the input format is finalized.
   void SetDuplicatedVelocity(Array<int> &velocity_starting_index,
                              Array<int> &velocity_related_constraints_index,
                              Array<int> &master_material_index,
                              int velocity_block_size)
   {
      duplicated_velocity = true;
      velocity_idx_start = velocity_starting_index;
      velocity_related_constraints = velocity_related_constraints_index;
      master_material_idx = master_material_index;
      this->velocity_block_size = velocity_block_size;
   }

   // Dykstra projection with Bregman divergence
   // At each iteration, we project onto the tangent plane of each constraint
   // psi_k = inv_sigmoid(Project_{k mod N}(sigmoid(psi_{k-1} + q_{k mod N})))
   // q_{k mod N} = psi_{k-1} + q_{k - N mod N} - psi_k
   // where Project_k is the projection onto the k-th constraint (tangent plane)
   void Project(Vector &projected_x);
private:
   void ProjectSumToOne(Vector &psi, Vector &qi);

   void Project(const Functional &con, Vector &psi, const Vector &grad,
                const real_t targ, Vector &psi_aux, Vector &projected_x);

   void MapLatent(const Vector &psi_,
                  const Vector &xmin_,
                  const Vector &xmax_,
                  Vector &x_);

   void MapPrimal(const Vector &x_,
                  const Vector &xmin_,
                  const Vector &xmax_,
                  Vector &psi_);

};
namespace remap
{
/// @brief Conservation functionals in the (eta, rho, p, v) variables, i.e. with
/// the specific internal energy e replaced by the pressure. This tree uses
/// p = rho*e, so gm1 is 1 here; it is kept as a parameter to document the
/// convention rather than hard-coding it.
/// The pointwise argument is u = [eta, rho, p, v_1, ..., v_dim].
///
/// These mirror remap::potential_f / remap::energy_f, using the pointwise
/// identity eta*rho*e = eta*p (this code uses p = rho*e). The internal energy becomes
/// bilinear in (eta, p) instead of trilinear in (eta, rho, e).

/// @brief int eta * p dx  (internal energy; p = rho*e so gm1 = 1)
inline real_t p_potential_f(const Vector &u, real_t gm1)
{ return u[0]*u[2] / gm1; }
inline void p_potential_df(const Vector &u, Vector &grad_u, real_t gm1)
{
   grad_u.SetSize(u.Size()); grad_u = 0.0;
   grad_u[0] = u[2] / gm1;
   grad_u[2] = u[0] / gm1;
}

/// @brief int eta * p + 0.5 * eta * rho * |v|^2 dx  (total energy)
inline real_t p_energy_f(const Vector &u, real_t gm1)
{
   MFEM_ASSERT(u.Size() > 3,
               "p_energy_f requires at least 4 components: [eta, rho, p, v1, ...]");
   real_t kinetic = 0.0;
   for (int i = 3; i < u.Size(); i++) { kinetic += u[i]*u[i]*0.5; }
   return u[0]*u[2]/gm1 + u[0]*u[1]*kinetic;
}
inline void p_energy_df(const Vector &u, Vector &grad_u, real_t gm1)
{
   MFEM_ASSERT(u.Size() > 3,
               "p_energy_df requires at least 4 components: [eta, rho, p, v1, ...]");
   grad_u.SetSize(u.Size()); grad_u = 0.0;
   real_t kinetic = 0.0;
   for (int i = 3; i < u.Size(); i++) { kinetic += u[i]*u[i]*0.5; }
   grad_u[0] = u[2]/gm1 + u[1]*kinetic;
   grad_u[1] = u[0]*kinetic;
   grad_u[2] = u[0]/gm1;
   const real_t mass = u[0]*u[1];
   for (int i = 3; i < u.Size(); i++) { grad_u[i] = mass*u[i]; }
}
} // namespace remap

/// @brief Report of the stage-2 energy box construction.
struct EnergyBoxReport
{
   real_t max_tighten = 0.0;  ///< largest shrinkage of the DMP box by pressure
   real_t max_clip    =
      0.0;  ///< largest gap when the two intervals do not overlap
   /// Largest distance the issued box reaches outside the DMP box (measured
   /// after widening and the non-negativity clamp), i.e. the energy-bound
   /// relaxation admitted in order to keep the pressure bounded.
   real_t max_dmp_excursion = 0.0;
   /// Largest distance the issued box reaches outside the *pressure* interval,
   /// i.e. the pressure-bound relaxation admitted (only the non-negativity
   /// clamp can produce one; expected to be ~0).
   real_t max_p_excursion = 0.0;
   int    n_empty     = 0;    ///< dofs where the intersection was empty
};

/// @brief Build the stage-2 energy box: the intersection of the energy DMP box
/// with the pressure box converted at fixed density, e = p / ((gamma-1)*rho),
/// where gamma-1 = 1 (p = rho*e).
///
/// The energy now lives at the quadrature points, next to rho and the pressure
/// box, so the bound is applied pointwise: p_min <= (gamma-1)*rho(x)*e(x) <=
/// p_max at every quadrature point, using that point's own density. This is
/// exact -- no dof-to-quad association is needed.
///
/// When the per-point intersection is empty the pressure bound wins, so e may
/// leave its DMP box.
EnergyBoxReport IntersectEnergyBoxWithPressure(MPI_Comm comm,
                                               const Vector &rho_q,
                                               const Vector &p_min_q,
                                               const Vector &p_max_q,
                                               const Vector &ind_q,
                                               real_t gm1,
                                               real_t ind_tol, real_t rho_tol_rel,
                                               Vector &e_min, Vector &e_max);

/// @brief Two-stage pressure-controlled Bregman remap.
///
/// Stage 1 projects (eta, rho, p, v), i.e. the specific internal energy is
/// replaced by the pressure p, which is then bounded by construction through
/// the Fermi-Dirac generator. The internal energy enters conservation as
/// int eta*p (gamma-1 = 1 here), the same quantity as int eta*rho*e but only
/// bilinear. No constraint ties (rho, e) to p, so the pressure box constrains a
/// primary variable and cannot make the feasible set empty (which it does when
/// P(rho, e) is bounded directly, since rho, e and 0.4*rho*e are then coupled).
///
/// Stage 2 freezes (eta, rho, v) and projects the energy alone onto the
/// energy-related global conservation law -- a single linear constraint plus a
/// box -- where the box is the intersection of the energy DMP box with the
/// pressure box converted at the frozen density.
///
/// Volume, mass and momentum depend only on (eta, rho, v) and so survive
/// stage 2 untouched; the energy law is enforced by stage 2 itself. All
/// conservation constraints therefore hold at the end.
class TwoStagePressureRemap
{
public:
   struct Options
   {
      real_t gamma_minus_one   = 1.0;
      real_t atol              = 1e-10;
      int    max_iter          = 100;
      /// Project e toward p/((gamma-1)*rho) from stage 1 rather than toward
      /// the interpolated energy.
      bool   e_target_from_pressure = true;
      /// Indicator threshold below which a point is treated as material-free,
      /// where the pressure bound is neither enforced nor reported.
      real_t ind_tol           = 1e-6;
      /// Density floor, RELATIVE to the largest density present, below which
      /// the pressure bound carries no information about the energy (p = rho*e
      /// is insensitive to e there) and the DMP box is kept.
      real_t rho_tol_rel       = 1e-6;
   };

   /// @param qspace         final-mesh quadrature space (eta, rho, e, p live here).
   /// @param pfes_v_scalar  final-mesh scalar velocity space (H1).
   /// @param mass_q,mass_h1  matching mass operators, supplied by the caller so
   ///        they are assembled once.
   TwoStagePressureRemap(QuadratureSpace &qspace,
                         ParFiniteElementSpace &pfes_v_scalar,
                         MassOperator &mass_q, MassOperator &mass_h1,
                         int num_materials, int dim, bool remap_v,
                         const Options &opts);

   /// @brief Run both stages.
   ///
   /// All (eta, rho, e, v_0 ... v_{dim-1}) vectors are global T-vectors with
   /// per-material blocks [size_qf, size_qf, size_e, size_v1 x dim].
   ///
   /// @param[in]  x_min,x_max  DMP boxes.
   /// @param[in]  p_min,p_max  pressure box at the quadrature points, per material.
   /// @param[in]  volume_0,mass_0  per-material conservation targets.
   /// @param[in]  energy_0     per-material energy target: the internal energy
   ///                          when remap_v is false, the total energy when true.
   /// @param[in]  moment_0     per-material momentum targets, size num_materials*dim.
   /// @param[in,out] x         interpolated state in, remapped state out.
   void Solve(const Vector &x_min, const Vector &x_max,
              const std::vector<Vector> &p_min,
              const std::vector<Vector> &p_max,
              const Vector &volume_0, const Vector &mass_0,
              const Vector &energy_0, const Vector &moment_0,
              Vector &x);

private:
   QuadratureSpace &qspace;
   ParFiniteElementSpace &pfes_v_scalar;
   MassOperator &mass_q, &mass_h1;
   const int num_materials, dim;
   const bool remap_v;
   const Options opts;

   // With the energy at the quadrature points, size_e == size_qf; it is kept
   // as a separate name only to mark the energy blocks.
   const int size_qf, size_e, size_v1;
   const int num_vars;               // per material: 3 + dim*remap_v
   int per_mat_e, per_mat_p;
   Array<int> space_idx_e, space_idx_p;
   Array<int> blk_e, blk_p;          // per-material block sizes
   std::vector<ParFiniteElementSpace*> fes;

   /// Stage 1: project (eta, rho, p, v). Returns the p-layout solution.
   void SolveStage1(const Vector &x_min, const Vector &x_max,
                    const std::vector<Vector> &p_min,
                    const std::vector<Vector> &p_max,
                    const Vector &volume_0, const Vector &mass_0,
                    const Vector &energy_0, const Vector &moment_0,
                    const Vector &x_interp, Vector &xp);

   /// Stage 2: project e with (eta, rho, v) frozen at their stage-1 values.
   void SolveStage2(const Vector &x_min, const Vector &x_max,
                    const std::vector<Vector> &p_min,
                    const std::vector<Vector> &p_max,
                    const Vector &energy_0, const Vector &xp,
                    const Vector &e_interp, Vector &x);
};

}

#endif // REMHOS_LVPP_HPP
