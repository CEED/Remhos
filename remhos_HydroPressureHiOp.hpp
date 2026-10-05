// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.

#ifndef MFEM_REMHOS_HYDRO_PRESSURE_HIOP
#define MFEM_REMHOS_HYDRO_PRESSURE_HIOP

#include "remhos_HiOp.hpp"

#include <memory>

namespace mfem
{

/// Monolithic hydro-remap optimization with pressure as the third unknown.
///
/// The local design-vector layout is
///
///     [indicator QF, density QF, pressure QF, velocity true dofs].
///
/// The equation of state follows the convention used by Remhos:
/// p = gamma*rho*e.  Consequently, the internal-energy part of the conserved
/// total energy is evaluated as rho*e = p/gamma.  This avoids division by rho
/// at empty quadrature points.
class RemhosHydroPressureHiOpProblem : public OptimizationProblem,
                                      public RemhosOptBase
{
private:
   enum class IntegralKind
   {
      Volume,
      Mass,
      Momentum,
      TotalEnergy
   };

   const Vector x_initial_;
   const Vector &pos_final_;
   QuadratureSpace &qspace_;
   ParFiniteElementSpace &vectorfespace_;
   QuadratureFunction p_target_;

   const int size_qf_;
   const int size_gf_vec_true_;

   Array<int> offsets_;
   Array<int> quadrature_offsets_;

   int spatial_dim_ = 0;
   int num_elements_ = 0;
   Mesh *mesh_ = nullptr;

   QuadratureFunction ind_0_;
   QuadratureFunction rho_0_;

   mutable std::unique_ptr<HypreParMatrix> velocity_mass_;

   class TotalEnergyGradVIntegrator : public LinearFormIntegrator
   {
   public:
      TotalEnergyGradVIntegrator(const QuadratureFunction &ind,
                                 const QuadratureFunction &rho,
                                 const ParGridFunction &velocity);

      void AssembleRHSElementVect(const FiniteElement &el,
                                  ElementTransformation &T,
                                  Vector &elvect) override;

   private:
      const QuadratureFunction *ind_;
      const QuadratureFunction *rho_;
      const ParGridFunction *velocity_;
   };

   class MomentumGradVIntegrator : public LinearFormIntegrator
   {
   public:
      MomentumGradVIntegrator(const QuadratureFunction &ind,
                              const QuadratureFunction &rho,
                              int component);

      void AssembleRHSElementVect(const FiniteElement &el,
                                  ElementTransformation &T,
                                  Vector &elvect) override;

   private:
      const QuadratureFunction *ind_;
      const QuadratureFunction *rho_;
      int component_;
   };

   class VelocityDifferenceIntegrator : public LinearFormIntegrator
   {
   public:
      VelocityDifferenceIntegrator(const ParGridFunction &velocity,
                                   const ParGridFunction &velocity_target,
                                   const QuadratureFunction &rule_source);

      void AssembleRHSElementVect(const FiniteElement &el,
                                  ElementTransformation &T,
                                  Vector &elvect) override;

   private:
      const ParGridFunction *velocity_;
      const ParGridFunction *velocity_target_;
      const QuadratureFunction *rule_source_;
   };

   void ExpandDesign(const Vector &x, Vector &full_design) const;

   double IntegrateConservedQuantity(const QuadratureFunction &ind,
                                     const QuadratureFunction &rho,
                                     const QuadratureFunction &pressure,
                                     const ParGridFunction &velocity,
                                     IntegralKind kind,
                                     int component = 0) const;

   double IntegrateVelocityDifference(const ParGridFunction &velocity,
                                      const ParGridFunction &velocity_target) const;

   double IntegrateVelocityGradientDifference(
      const ParGridFunction &velocity,
      const ParGridFunction &velocity_target) const;

public:
   real_t w_1 = 1e1;    ///< Indicator objective weight.
   real_t w_2 = 1e1;    ///< Density objective weight.
   real_t w_p = 1e1;    ///< Pressure objective weight.
   real_t w_4 = 1e1;    ///< Velocity L2 objective weight.
   real_t w_4_H1 = 1e-1;

   double gamma = 1.0;

   RemhosHydroPressureHiOpProblem(
      QuadratureSpace &qspace,
      ParFiniteElementSpace &vectorfespace,
      const Vector &pos_final,
      const Vector &initial_design,
      const QuadratureFunction &pressure_target,
      int num_design_variables,
      const Vector &xmin,
      const Vector &xmax,
      double initial_volume,
      double initial_mass,
      const Vector &initial_momentum,
      double initial_total_energy,
      int num_constraints,
      bool use_H1_seminorm,
      const Array<int> &optimization_indices,
      double gamma_,
      bool is_L2 = true,
      bool subproblem_ = false);

   void setWeightedSpaceType(
      hiop::hiopInterfaceBase::WeightedSpaceType weighted_space)
   {
      SetWeightedSpaceTypeBase(weighted_space);
   }

   hiop::hiopInterfaceBase::WeightedSpaceType getWeightedSpaceType() const override
   {
      return GetWeightedSpaceTypeBase();
   }

   real_t CalcObjective(const Vector &x) const override;

   void CalcObjectiveGrad(const Vector &x, Vector &grad) const override;

   void CalcObjectiveM(std::vector<Vector> &diag_mass,
                       std::vector<HypreParMatrix *> &matrices) const override;

   void CalcConstraint(int constraint_number,
                       const Vector &x,
                       Vector &constraint_values) const override;

   void CalcConstraintGrad(int constraint_number,
                           const Vector &x,
                           Vector &grad) const override;
};

} // namespace mfem

#endif // MFEM_REMHOS_HYDRO_PRESSURE_HIOP
