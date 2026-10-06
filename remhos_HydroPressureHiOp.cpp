// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.

#include "remhos_HydroPressureHiOp.hpp"

namespace mfem
{

RemhosHydroPressureHiOpProblem::RemhosHydroPressureHiOpProblem(
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
   bool is_L2,
   bool subproblem_)
   : OptimizationProblem(num_design_variables, nullptr, nullptr),
     RemhosOptBase(num_constraints, num_design_variables,
                   optimization_indices, subproblem_),
     x_initial_(initial_design),
     pos_final_(pos_final),
     qspace_(qspace),
     vectorfespace_(vectorfespace),
     p_target_(&qspace),
     size_qf_(qspace.GetSize()),
     size_gf_vec_true_(vectorfespace.GetTrueVSize()),
     offsets_(5),
     ind_0_(&qspace),
     rho_0_(&qspace),
     gamma(gamma_)
{
   std::cout<<"Remap with preassure design variable"<<std::endl;
   MFEM_VERIFY(gamma > 0.0, "The pressure equation-of-state factor must be positive.");

   mesh_ = qspace_.GetMesh();
   MFEM_VERIFY(mesh_ == vectorfespace_.GetMesh(),
               "The pressure and velocity spaces must use the same mesh.");

   spatial_dim_ = vectorfespace_.GetVDim();
   MFEM_VERIFY(spatial_dim_ == mesh_->SpaceDimension(),
               "The velocity vector dimension must match the mesh space dimension.");
   MFEM_VERIFY(initial_momentum.Size() == spatial_dim_,
               "The target momentum has the wrong dimension.");
   MFEM_VERIFY(num_constraints == 3 + spatial_dim_,
               "Expected volume, mass, momentum, and total-energy constraints.");
   MFEM_VERIFY(pressure_target.Size() == size_qf_,
               "The pressure target has the wrong size.");

   offsets_[0] = 0;
   offsets_[1] = offsets_[0] + size_qf_;
   offsets_[2] = offsets_[1] + size_qf_;
   offsets_[3] = offsets_[2] + size_qf_;
   offsets_[4] = offsets_[3] + size_gf_vec_true_;

   MFEM_VERIFY(x_initial_.Size() == offsets_[4],
               "Invalid [indicator,density,pressure,velocity] design size.");
   if (subproblem)
   {
      MFEM_VERIFY(optimization_indices.Size() == num_design_variables,
                  "Optimization subset size does not match the design size.");
      for (int i = 0; i < optimization_indices.Size(); i++)
      {
         MFEM_VERIFY(optimization_indices[i] >= 0 &&
                     optimization_indices[i] < offsets_[4],
                     "Optimization subset index is out of range.");
      }
   }
   else
   {
      MFEM_VERIFY(num_design_variables == offsets_[4],
                  "The full design size does not match the supplied spaces.");
   }

   targetVol = initial_volume;
   targetMass = initial_mass;
   targetMomentum = initial_momentum;
   targetEnergy = initial_total_energy;
   isL2_ = is_L2;
   this->numConstraints = num_constraints;
   w_4_H1 = use_H1_seminorm ? 1e-1 : 0.0;

   // CalcConstraint returns residuals relative to the requested conserved
   // quantities, so every equality constraint has zero as its right-hand side.
   massvec = 0.0;
   SetEqualityConstraint(massvec);
   SetSolutionBounds(xmin, xmax);

   num_elements_ = mesh_->GetNE();
   quadrature_offsets_.SetSize(num_elements_ + 1);
   quadrature_offsets_[0] = 0;
   for (int e = 0; e < num_elements_; e++)
   {
      quadrature_offsets_[e + 1] =
         quadrature_offsets_[e] + qspace_.GetElementIntRule(e).GetNPoints();
   }
   MFEM_VERIFY(quadrature_offsets_.Last() == size_qf_,
               "Quadrature-space size does not match its element rules.");

   Vector ind_initial(const_cast<real_t *>(x_initial_.GetData()), size_qf_);
   Vector rho_initial(const_cast<real_t *>(x_initial_.GetData()) + size_qf_,
                      size_qf_);
   ind_0_ = ind_initial;
   rho_0_ = rho_initial;
   p_target_ = static_cast<const Vector &>(pressure_target);
}

void RemhosHydroPressureHiOpProblem::ExpandDesign(
   const Vector &x, Vector &full_design) const
{
   full_design.SetSize(offsets_[4]);
   if (subproblem)
   {
      MFEM_VERIFY(x.Size() == optProbInd.Size(),
                  "Optimization-subset vector has the wrong size.");
      full_design = x_initial_;
      full_design.SetSubVector(optProbInd, x);
   }
   else
   {
      MFEM_VERIFY(x.Size() == offsets_[4], "Design vector has the wrong size.");
      full_design = x;
   }
}

real_t RemhosHydroPressureHiOpProblem::CalcObjective(const Vector &x) const
{
   Vector full_design;
   ExpandDesign(x, full_design);

   QuadratureFunction ind(&qspace_, full_design.GetData());
   QuadratureFunction rho(&qspace_, full_design.GetData() + size_qf_);
   QuadratureFunction pressure(&qspace_, full_design.GetData() + 2*size_qf_);

   QuadratureFunction ind_diff(&qspace_);
   QuadratureFunction rho_diff(&qspace_);
   QuadratureFunction pressure_diff(&qspace_);
   subtract(ind, ind_0_, ind_diff);
   subtract(rho, rho_0_, rho_diff);
   subtract(pressure, p_target_, pressure_diff);

   ParGridFunction velocity(&vectorfespace_);
   Vector velocity_true(full_design.GetData() + 3*size_qf_,
                        size_gf_vec_true_);
   velocity.SetFromTrueDofs(velocity_true);

   ParGridFunction velocity_target(&vectorfespace_);
   Vector velocity_target_true(
      const_cast<real_t *>(x_initial_.GetData()) + 3*size_qf_,
      size_gf_vec_true_);
   velocity_target.SetFromTrueDofs(velocity_target_true);

   real_t ind_norm = 0.0;
   real_t rho_norm = 0.0;
   real_t pressure_norm = 0.0;

   if (isL2_)
   {
      for (int e = 0; e < num_elements_; e++)
      {
         IsoparametricTransformation transformation;
         mesh_->GetElementTransformation(e, pos_final_, &transformation);
         const IntegrationRule &rule = qspace_.GetElementIntRule(e);
         const int start = quadrature_offsets_[e];

         for (int q = 0; q < rule.GetNPoints(); q++)
         {
            const IntegrationPoint &ip = rule.IntPoint(q);
            transformation.SetIntPoint(&ip);
            const real_t weight = transformation.Weight()*ip.weight;
            const int i = start + q;
            ind_norm += 0.5*weight*ind_diff[i]*ind_diff[i];
            rho_norm += 0.5*weight*rho_diff[i]*rho_diff[i];
            pressure_norm +=
               0.5*weight*pressure_diff[i]*pressure_diff[i];
         }
      }
   }
   else
   {
      ind_norm = 0.5*ind_diff.Norml2()*ind_diff.Norml2();
      rho_norm = 0.5*rho_diff.Norml2()*rho_diff.Norml2();
      pressure_norm =
         0.5*pressure_diff.Norml2()*pressure_diff.Norml2();
   }

   MPI_Allreduce(MPI_IN_PLACE, &ind_norm, 1, MPI_DOUBLE, MPI_SUM,
                 vectorfespace_.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &rho_norm, 1, MPI_DOUBLE, MPI_SUM,
                 vectorfespace_.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &pressure_norm, 1, MPI_DOUBLE, MPI_SUM,
                 vectorfespace_.GetComm());

   const real_t velocity_norm =
      IntegrateVelocityDifference(velocity, velocity_target);
   const real_t velocity_H1_norm =
      IntegrateVelocityGradientDifference(velocity, velocity_target);

   return w_1*ind_norm + w_2*rho_norm + w_p*pressure_norm +
          w_4*velocity_norm + w_4_H1*velocity_H1_norm;
}

void RemhosHydroPressureHiOpProblem::CalcObjectiveGrad(
   const Vector &x, Vector &grad) const
{
   Vector full_design;
   ExpandDesign(x, full_design);

   QuadratureFunction ind(&qspace_, full_design.GetData());
   QuadratureFunction rho(&qspace_, full_design.GetData() + size_qf_);
   QuadratureFunction pressure(&qspace_, full_design.GetData() + 2*size_qf_);

   QuadratureFunction ind_diff(&qspace_);
   QuadratureFunction rho_diff(&qspace_);
   QuadratureFunction pressure_diff(&qspace_);
   subtract(ind, ind_0_, ind_diff);
   subtract(rho, rho_0_, rho_diff);
   subtract(pressure, p_target_, pressure_diff);

   ParGridFunction velocity(&vectorfespace_);
   Vector velocity_true(full_design.GetData() + 3*size_qf_,
                        size_gf_vec_true_);
   velocity.SetFromTrueDofs(velocity_true);

   ParGridFunction velocity_target(&vectorfespace_);
   Vector velocity_target_true(
      const_cast<real_t *>(x_initial_.GetData()) + 3*size_qf_,
      size_gf_vec_true_);
   velocity_target.SetFromTrueDofs(velocity_target_true);

   if (isL2_)
   {
      for (int e = 0; e < num_elements_; e++)
      {
         IsoparametricTransformation transformation;
         mesh_->GetElementTransformation(e, pos_final_, &transformation);
         const IntegrationRule &rule = qspace_.GetElementIntRule(e);
         const int start = quadrature_offsets_[e];

         for (int q = 0; q < rule.GetNPoints(); q++)
         {
            const IntegrationPoint &ip = rule.IntPoint(q);
            transformation.SetIntPoint(&ip);
            const real_t weight = transformation.Weight()*ip.weight;
            const int i = start + q;
            ind_diff[i] *= weight;
            rho_diff[i] *= weight;
            pressure_diff[i] *= weight;
         }
      }
   }

   ParLinearForm velocity_form(&vectorfespace_);
   velocity_form.AddDomainIntegrator(
      new VelocityDifferenceIntegrator(velocity, velocity_target, ind));
   velocity_form.Assemble();
   Vector velocity_grad(size_gf_vec_true_);
   velocity_form.ParallelAssemble(velocity_grad);

   ParLinearForm velocity_H1_form(&vectorfespace_);
   VectorGradientDifferenceCoefficient gradient_difference(velocity,
                                                           velocity_target);
   velocity_H1_form.AddDomainIntegrator(
      new VectorDomainLFH1semiNormIntegrator(gradient_difference));
   velocity_H1_form.Assemble();
   Vector velocity_H1_grad(size_gf_vec_true_);
   velocity_H1_form.ParallelAssemble(velocity_H1_grad);

   ind_diff *= w_1;
   rho_diff *= w_2;
   pressure_diff *= w_p;
   velocity_grad *= w_4;
   velocity_H1_grad *= w_4_H1;
   velocity_grad += velocity_H1_grad;

   BlockVector full_grad(offsets_);
   full_grad.GetBlock(0) = ind_diff;
   full_grad.GetBlock(1) = rho_diff;
   full_grad.GetBlock(2) = pressure_diff;
   full_grad.GetBlock(3) = velocity_grad;

   if (subproblem)
   {
      full_grad.GetSubVector(optProbInd, grad);
   }
   else
   {
      grad = full_grad;
   }
}

void RemhosHydroPressureHiOpProblem::CalcObjectiveM(
   std::vector<Vector> &diag_mass,
   std::vector<HypreParMatrix *> &matrices) const
{
   MFEM_VERIFY(!subproblem,
               "The weighted-space operator is not available for a subset problem.");

   diag_mass.resize(3);
   for (int block = 0; block < 3; block++)
   {
      diag_mass[block].SetSize(size_qf_);
      diag_mass[block] = 1.0;
   }

   if (isL2_)
   {
      for (int e = 0; e < num_elements_; e++)
      {
         IsoparametricTransformation transformation;
         mesh_->GetElementTransformation(e, pos_final_, &transformation);
         const IntegrationRule &rule = qspace_.GetElementIntRule(e);
         const int start = quadrature_offsets_[e];

         for (int q = 0; q < rule.GetNPoints(); q++)
         {
            const IntegrationPoint &ip = rule.IntPoint(q);
            transformation.SetIntPoint(&ip);
            const real_t weight = transformation.Weight()*ip.weight;
            for (int block = 0; block < 3; block++)
            {
               diag_mass[block][start + q] = weight;
            }
         }
      }
   }

   if (!velocity_mass_)
   {
      ParBilinearForm mass_form(&vectorfespace_);
      mass_form.AddDomainIntegrator(new VectorMassIntegrator());
      mass_form.Assemble();
      mass_form.Finalize();
      velocity_mass_.reset(mass_form.ParallelAssemble());
   }

   matrices.resize(1);
   matrices[0] = velocity_mass_.get();
}

void RemhosHydroPressureHiOpProblem::CalcConstraint(
   int constraint_number, const Vector &x, Vector &constraint_values) const
{
   Vector full_design;
   ExpandDesign(x, full_design);

   QuadratureFunction ind(&qspace_, full_design.GetData());
   QuadratureFunction rho(&qspace_, full_design.GetData() + size_qf_);
   QuadratureFunction pressure(&qspace_, full_design.GetData() + 2*size_qf_);
   ParGridFunction velocity(&vectorfespace_);
   Vector velocity_true(full_design.GetData() + 3*size_qf_,
                        size_gf_vec_true_);
   velocity.SetFromTrueDofs(velocity_true);

   if (constraint_number == 0)
   {
      constraint_values[0] =
         IntegrateConservedQuantity(ind, rho, pressure, velocity,
                                    IntegralKind::Volume) - targetVol;
   }
   else if (constraint_number == 1)
   {
      constraint_values[1] =
         IntegrateConservedQuantity(ind, rho, pressure, velocity,
                                    IntegralKind::Mass) - targetMass;
   }
   else if (constraint_number >= 2 &&
            constraint_number < 2 + spatial_dim_)
   {
      const int component = constraint_number - 2;
      constraint_values[constraint_number] =
         IntegrateConservedQuantity(ind, rho, pressure, velocity,
                                    IntegralKind::Momentum, component) -
         targetMomentum[component];
   }
   else if (constraint_number == 2 + spatial_dim_)
   {
      constraint_values[constraint_number] =
         IntegrateConservedQuantity(ind, rho, pressure, velocity,
                                    IntegralKind::TotalEnergy) - targetEnergy;
   }
   else
   {
      MFEM_ABORT("Constraint index does not exist.");
   }
}

void RemhosHydroPressureHiOpProblem::CalcConstraintGrad(
   int constraint_number, const Vector &x, Vector &grad) const
{
   Vector full_design;
   ExpandDesign(x, full_design);

   QuadratureFunction ind(&qspace_, full_design.GetData());
   QuadratureFunction rho(&qspace_, full_design.GetData() + size_qf_);
   QuadratureFunction pressure(&qspace_, full_design.GetData() + 2*size_qf_);
   ParGridFunction velocity(&vectorfespace_);
   Vector velocity_true(full_design.GetData() + 3*size_qf_,
                        size_gf_vec_true_);
   velocity.SetFromTrueDofs(velocity_true);

   QuadratureFunction ind_grad(&qspace_);
   QuadratureFunction rho_grad(&qspace_);
   QuadratureFunction pressure_grad(&qspace_);
   ind_grad = 0.0;
   rho_grad = 0.0;
   pressure_grad = 0.0;
   Vector velocity_grad(size_gf_vec_true_);
   velocity_grad = 0.0;

   const bool momentum_constraint =
      constraint_number >= 2 && constraint_number < 2 + spatial_dim_;
   const bool energy_constraint = constraint_number == 2 + spatial_dim_;
   MFEM_VERIFY(constraint_number == 0 || constraint_number == 1 ||
               momentum_constraint || energy_constraint,
               "Constraint index does not exist.");

   for (int e = 0; e < num_elements_; e++)
   {
      IsoparametricTransformation transformation;
      mesh_->GetElementTransformation(e, pos_final_, &transformation);
      const IntegrationRule &rule = qspace_.GetElementIntRule(e);
      const int start = quadrature_offsets_[e];

      for (int q = 0; q < rule.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = rule.IntPoint(q);
         transformation.SetIntPoint(&ip);
         const real_t weight = transformation.Weight()*ip.weight;
         const int i = start + q;
         const real_t ind_value = ind[i];
         const real_t rho_value = rho[i];

         if (constraint_number == 0)
         {
            ind_grad[i] = weight;
         }
         else if (constraint_number == 1)
         {
            ind_grad[i] = weight*rho_value;
            rho_grad[i] = weight*ind_value;
         }
         else
         {
            Vector velocity_value;
            velocity.GetVectorValue(transformation, ip, velocity_value);

            if (momentum_constraint)
            {
               const int component = constraint_number - 2;
               ind_grad[i] =
                  weight*rho_value*velocity_value[component];
               rho_grad[i] =
                  weight*ind_value*velocity_value[component];
            }
            else
            {
               const real_t velocity_squared =
                  velocity_value*velocity_value;
               ind_grad[i] = weight*(pressure[i]/gamma +
                                     0.5*rho_value*velocity_squared);
               rho_grad[i] =
                  weight*ind_value*0.5*velocity_squared;
               pressure_grad[i] = weight*ind_value/gamma;
            }
         }
      }
   }

   if (momentum_constraint)
   {
      ParLinearForm velocity_form(&vectorfespace_);
      velocity_form.AddDomainIntegrator(
         new MomentumGradVIntegrator(ind, rho, constraint_number - 2));
      velocity_form.Assemble();
      velocity_form.ParallelAssemble(velocity_grad);
   }
   else if (energy_constraint)
   {
      ParLinearForm velocity_form(&vectorfespace_);
      velocity_form.AddDomainIntegrator(
         new TotalEnergyGradVIntegrator(ind, rho, velocity));
      velocity_form.Assemble();
      velocity_form.ParallelAssemble(velocity_grad);
   }

   BlockVector full_grad(offsets_);
   full_grad.GetBlock(0) = ind_grad;
   full_grad.GetBlock(1) = rho_grad;
   full_grad.GetBlock(2) = pressure_grad;
   full_grad.GetBlock(3) = velocity_grad;

   if (subproblem)
   {
      full_grad.GetSubVector(optProbInd, grad);
   }
   else
   {
      grad = full_grad;
   }
}

double RemhosHydroPressureHiOpProblem::IntegrateConservedQuantity(
   const QuadratureFunction &ind,
   const QuadratureFunction &rho,
   const QuadratureFunction &pressure,
   const ParGridFunction &velocity,
   IntegralKind kind,
   int component) const
{
   MFEM_VERIFY(kind != IntegralKind::Momentum ||
               (component >= 0 && component < spatial_dim_),
               "Momentum component is out of range.");

   double integral = 0.0;
   for (int e = 0; e < num_elements_; e++)
   {
      IsoparametricTransformation transformation;
      mesh_->GetElementTransformation(e, pos_final_, &transformation);
      const IntegrationRule &rule = qspace_.GetElementIntRule(e);
      const int num_points = rule.GetNPoints();

      Vector ind_values;
      Vector rho_values;
      Vector pressure_values;
      DenseMatrix velocity_values;
      ind.GetValues(e, ind_values);
      rho.GetValues(e, rho_values);
      pressure.GetValues(e, pressure_values);
      if (kind == IntegralKind::Momentum || kind == IntegralKind::TotalEnergy)
      {
         velocity.GetVectorValues(transformation, rule, velocity_values);
      }

      for (int q = 0; q < num_points; q++)
      {
         const IntegrationPoint &ip = rule.IntPoint(q);
         transformation.SetIntPoint(&ip);
         const real_t weight = transformation.Weight()*ip.weight;
         real_t integrand = 0.0;

         switch (kind)
         {
            case IntegralKind::Volume:
               integrand = ind_values[q];
               break;
            case IntegralKind::Mass:
               integrand = ind_values[q]*rho_values[q];
               break;
            case IntegralKind::Momentum:
               integrand = ind_values[q]*rho_values[q]*
                           velocity_values(component, q);
               break;
            case IntegralKind::TotalEnergy:
            {
               real_t velocity_squared = 0.0;
               for (int d = 0; d < spatial_dim_; d++)
               {
                  velocity_squared +=
                     velocity_values(d, q)*velocity_values(d, q);
               }
               integrand = ind_values[q]*(pressure_values[q]/gamma +
                           0.5*rho_values[q]*velocity_squared);
               break;
            }
         }
         integral += weight*integrand;
      }
   }

   MPI_Allreduce(MPI_IN_PLACE, &integral, 1, MPI_DOUBLE, MPI_SUM,
                 vectorfespace_.GetComm());
   return integral;
}

double RemhosHydroPressureHiOpProblem::IntegrateVelocityDifference(
   const ParGridFunction &velocity,
   const ParGridFunction &velocity_target) const
{
   double integral = 0.0;
   for (int e = 0; e < num_elements_; e++)
   {
      IsoparametricTransformation transformation;
      mesh_->GetElementTransformation(e, pos_final_, &transformation);
      const IntegrationRule &rule = qspace_.GetElementIntRule(e);
      DenseMatrix velocity_values;
      DenseMatrix target_values;
      velocity.GetVectorValues(transformation, rule, velocity_values);
      velocity_target.GetVectorValues(transformation, rule, target_values);
      velocity_values -= target_values;

      for (int q = 0; q < rule.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = rule.IntPoint(q);
         transformation.SetIntPoint(&ip);
         real_t norm_squared = 0.0;
         for (int d = 0; d < spatial_dim_; d++)
         {
            norm_squared += velocity_values(d, q)*velocity_values(d, q);
         }
         integral += 0.5*transformation.Weight()*ip.weight*norm_squared;
      }
   }

   MPI_Allreduce(MPI_IN_PLACE, &integral, 1, MPI_DOUBLE, MPI_SUM,
                 vectorfespace_.GetComm());
   return integral;
}

double RemhosHydroPressureHiOpProblem::IntegrateVelocityGradientDifference(
   const ParGridFunction &velocity,
   const ParGridFunction &velocity_target) const
{
   double integral = 0.0;
   for (int e = 0; e < num_elements_; e++)
   {
      IsoparametricTransformation transformation;
      mesh_->GetElementTransformation(e, pos_final_, &transformation);
      const IntegrationRule &rule = qspace_.GetElementIntRule(e);

      for (int q = 0; q < rule.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = rule.IntPoint(q);
         transformation.SetIntPoint(&ip);
         DenseMatrix velocity_gradient;
         DenseMatrix target_gradient;
         velocity.GetVectorGradient(transformation, velocity_gradient);
         velocity_target.GetVectorGradient(transformation, target_gradient);
         velocity_gradient -= target_gradient;
         integral += 0.5*transformation.Weight()*ip.weight*
                     velocity_gradient.FNorm2();
      }
   }

   MPI_Allreduce(MPI_IN_PLACE, &integral, 1, MPI_DOUBLE, MPI_SUM,
                 vectorfespace_.GetComm());
   return integral;
}

RemhosHydroPressureHiOpProblem::TotalEnergyGradVIntegrator::
TotalEnergyGradVIntegrator(const QuadratureFunction &ind,
                           const QuadratureFunction &rho,
                           const ParGridFunction &velocity)
   : ind_(&ind), rho_(&rho), velocity_(&velocity)
{ }

void RemhosHydroPressureHiOpProblem::TotalEnergyGradVIntegrator::
AssembleRHSElementVect(const FiniteElement &el,
                       ElementTransformation &transformation,
                       Vector &elvect)
{
   const int dof = el.GetDof();
   const int vdim = velocity_->VectorDim();
   const int element = transformation.ElementNo;
   Vector shape(dof);
   Vector velocity_value(vdim);
   elvect.SetSize(dof*vdim);
   elvect = 0.0;

   const IntegrationRule &rule = ind_->GetSpace()->GetIntRule(element);
   Vector ind_values;
   Vector rho_values;
   ind_->GetValues(element, ind_values);
   rho_->GetValues(element, rho_values);

   for (int q = 0; q < rule.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = rule.IntPoint(q);
      transformation.SetIntPoint(&ip);
      const real_t weight = ip.weight*transformation.Weight();
      velocity_->GetVectorValue(element, ip, velocity_value);
      el.CalcShape(ip, shape);

      for (int d = 0; d < vdim; d++)
      {
         Vector component(elvect.GetData() + d*dof, dof);
         component.Add(weight*ind_values[q]*rho_values[q]*velocity_value[d],
                       shape);
      }
   }
}

RemhosHydroPressureHiOpProblem::MomentumGradVIntegrator::
MomentumGradVIntegrator(const QuadratureFunction &ind,
                        const QuadratureFunction &rho,
                        int component)
   : ind_(&ind), rho_(&rho), component_(component)
{ }

void RemhosHydroPressureHiOpProblem::MomentumGradVIntegrator::
AssembleRHSElementVect(const FiniteElement &el,
                       ElementTransformation &transformation,
                       Vector &elvect)
{
   const int dof = el.GetDof();
   const int vdim = rho_->GetSpace()->GetMesh()->SpaceDimension();
   const int element = transformation.ElementNo;
   MFEM_VERIFY(component_ >= 0 && component_ < vdim,
               "Momentum component is out of range.");

   Vector shape(dof);
   elvect.SetSize(dof*vdim);
   elvect = 0.0;

   const IntegrationRule &rule = ind_->GetSpace()->GetIntRule(element);
   Vector ind_values;
   Vector rho_values;
   ind_->GetValues(element, ind_values);
   rho_->GetValues(element, rho_values);

   for (int q = 0; q < rule.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = rule.IntPoint(q);
      transformation.SetIntPoint(&ip);
      const real_t weight = ip.weight*transformation.Weight();
      el.CalcShape(ip, shape);
      Vector component(elvect.GetData() + component_*dof, dof);
      component.Add(weight*ind_values[q]*rho_values[q], shape);
   }
}

RemhosHydroPressureHiOpProblem::VelocityDifferenceIntegrator::
VelocityDifferenceIntegrator(const ParGridFunction &velocity,
                             const ParGridFunction &velocity_target,
                             const QuadratureFunction &rule_source)
   : velocity_(&velocity),
     velocity_target_(&velocity_target),
     rule_source_(&rule_source)
{ }

void RemhosHydroPressureHiOpProblem::VelocityDifferenceIntegrator::
AssembleRHSElementVect(const FiniteElement &el,
                       ElementTransformation &transformation,
                       Vector &elvect)
{
   const int dof = el.GetDof();
   const int vdim = velocity_->VectorDim();
   const int element = transformation.ElementNo;
   Vector shape(dof);
   Vector velocity_value(vdim);
   Vector target_value(vdim);
   elvect.SetSize(dof*vdim);
   elvect = 0.0;

   const IntegrationRule &rule = rule_source_->GetSpace()->GetIntRule(element);
   for (int q = 0; q < rule.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = rule.IntPoint(q);
      transformation.SetIntPoint(&ip);
      velocity_->GetVectorValue(element, ip, velocity_value);
      velocity_target_->GetVectorValue(element, ip, target_value);
      const real_t weight = ip.weight*transformation.Weight();
      el.CalcShape(ip, shape);

      for (int d = 0; d < vdim; d++)
      {
         Vector component(elvect.GetData() + d*dof, dof);
         component.Add(weight*(velocity_value[d] - target_value[d]), shape);
      }
   }
}

} // namespace mfem
