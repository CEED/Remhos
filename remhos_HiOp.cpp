// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include "remhos_HiOp.hpp"
#include "remhos_tools.hpp"

using namespace std;

namespace mfem
{
void GetOptimizationSubsetInd(
      const mfem::Vector & xmin, const mfem::Vector & xmax, mfem::Array<int> & optInd)
{
      int n = xmin.Size();

      int SizeOptSubset = GetSizeOptimizationSubset( xmin, xmax);

      optInd.SetSize(SizeOptSubset);
      int counter = 0;
      double eps = 1e-10;

      mfem::Vector diff(xmin);
      diff -=xmax;

      for( int Ik = 0; Ik < n; Ik++)
      {
         if (std::abs(diff[Ik]) > eps)
         {
            optInd[counter] = Ik;
            counter ++;
         }
      }
}

int GetSizeOptimizationSubset(const Vector &xmin, const Vector &xmax)
{
      const int n = xmin.Size();
      const double eps = 1e-10;

      Vector diff(xmin);
      diff -= xmax;

      int counter = 0;
      for (int Ik = 0; Ik < n; Ik++)
      {
         if (std::abs(diff[Ik]) > eps) { counter++; }
      }

      return counter;
}

VectorGradientDifferenceCoefficient::VectorGradientDifferenceCoefficient(
   const mfem::ParGridFunction &v, const mfem::ParGridFunction &v_ref)
   : MatrixCoefficient(v.VectorDim(), v.ParFESpace()->GetParMesh()->Dimension()),
     v_(&v), v_ref_(&v_ref)
{
   MFEM_VERIFY(v.VectorDim() == v_ref.VectorDim(),
               "Vector dimensions must match.");
   MFEM_VERIFY(v.ParFESpace()->GetParMesh()->Dimension() ==
               v_ref.ParFESpace()->GetParMesh()->Dimension(),
               "Spatial dimensions must match.");
}

void VectorGradientDifferenceCoefficient::Eval(
   DenseMatrix &K, ElementTransformation &T, const IntegrationPoint &ip)
{
   DenseMatrix K_ref;
   T.SetIntPoint(&ip);
   v_->GetVectorGradient(T, K);
   v_ref_->GetVectorGradient(T, K_ref);
   K -= K_ref;
}

VectorDomainLFH1semiNormIntegrator::VectorDomainLFH1semiNormIntegrator(
   mfem::MatrixCoefficient &Q)
   : Q_(&Q)
{ }

void VectorDomainLFH1semiNormIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &T, Vector &elvect)
{
   const int dof = el.GetDof();
   const int dim = el.GetDim();
   const int vdim = Q_->GetHeight();

   MFEM_VERIFY(Q_->GetWidth() == dim,
               "Matrix coefficient width must match the mesh dimension.");

   DenseMatrix dshape(dof, dim);
   DenseMatrix q_val(vdim, dim);

   elvect.SetSize(dof * vdim);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      const int order = T.OrderW() + 2 * el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   for (int q = 0; q < ir->GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir->IntPoint(q);
      T.SetIntPoint(&ip);

      const real_t w = ip.weight * T.Weight();
      el.CalcPhysDShape(T, dshape);
      Q_->Eval(q_val, T, ip);

      for (int c = 0; c < vdim; c++)
      {
         Vector elvect_comp(elvect.GetData() + c * dof, dof);
         for (int j = 0; j < dof; j++)
         {
            real_t contribution = 0.0;
            for (int d = 0; d < dim; d++)
            {
               contribution += q_val(c, d) * dshape(j, d);
            }
            elvect_comp(j) += w * contribution;
         }
      }
   }
}


RemhosIndRhoEHiOpProblem::EnergyGradIntegrator::EnergyGradIntegrator(
  const mfem::QuadratureFunction &ind, const mfem::QuadratureFunction &rho)
  : ind_(&ind), rho_(&rho) 
  {}

void RemhosIndRhoEHiOpProblem::EnergyGradIntegrator::AssembleRHSElementVect(
  const FiniteElement &el, ElementTransformation &T, Vector &elvect)
{
  // grab sizes
  int dof = el.GetDof();
  int dim = el.GetDim();
  int eleIndex = T.ElementNo;

  // initialize storage
  Vector N(dof);

  // output vector
  elvect.SetSize(dof);
  elvect = 0.0;

  const IntegrationRule *ir = &(ind_->GetSpace()->GetIntRule(eleIndex));
  const int nqp = ir->GetNPoints();

  Vector ind_vals(nqp), rho_vals(nqp);
  ind_->GetValues(eleIndex, ind_vals);
  rho_->GetValues(eleIndex, rho_vals);

  // loop over integration points
  for (int i = 0; i < ir->GetNPoints(); i++)
  {
    // set current integration point
    const IntegrationPoint &ip = ir->IntPoint(i);
    T.SetIntPoint(&ip);

    // evaluate gaussian integration weight
    double w = ip.weight * T.Weight();

    el.CalcShape(ip, N);

    elvect.Add(w * ind_vals[i] * rho_vals[i] , N);
  }
}

RemhosHydroHiOpProblem::totalEnergyGradEIntegrator::totalEnergyGradEIntegrator(
  const mfem::QuadratureFunction &ind, const mfem::QuadratureFunction &rho)
  : ind_(&ind), rho_(&rho) 
  {}

void RemhosHydroHiOpProblem::totalEnergyGradEIntegrator::AssembleRHSElementVect(
  const FiniteElement &el, ElementTransformation &T, Vector &elvect)
{
  // grab sizes
  int dof = el.GetDof();
  int dim = el.GetDim();
  int eleIndex = T.ElementNo;

  // initialize storage
  Vector N(dof);

  // output vector
  elvect.SetSize(dof);
  elvect = 0.0;

  const IntegrationRule *ir = &(ind_->GetSpace()->GetIntRule(eleIndex));
  const int nqp = ir->GetNPoints();

  Vector ind_vals(nqp), rho_vals(nqp);
  ind_->GetValues(eleIndex, ind_vals);
  rho_->GetValues(eleIndex, rho_vals);

  // loop over integration points
  for (int i = 0; i < ir->GetNPoints(); i++)
  {
    // set current integration point
    const IntegrationPoint &ip = ir->IntPoint(i);
    T.SetIntPoint(&ip);

    // evaluate gaussian integration weight
    double w = ip.weight * T.Weight();

    el.CalcShape(ip, N);

    elvect.Add(w * ind_vals[i] * rho_vals[i] , N);
  }
}

RemhosHydroHiOpProblem::totalEnergyGradVIntegrator::totalEnergyGradVIntegrator(
  const mfem::QuadratureFunction &ind, const mfem::QuadratureFunction &rho, const mfem::ParGridFunction &vel)
  : ind_(&ind), rho_(&rho), vel_(&vel)
  {}

void RemhosHydroHiOpProblem::totalEnergyGradVIntegrator::AssembleRHSElementVect(
  const FiniteElement &el, ElementTransformation &T, Vector &elvect)
{
  // grab sizes
  int dof = el.GetDof();
  int dim = el.GetDim();
  int eleIndex = T.ElementNo;

  // initialize storage
  Vector N(dof);
  Vector velGP(dim);

  // output vector
  elvect.SetSize(dof*dim);
  elvect = 0.0;

  const IntegrationRule *ir = &(ind_->GetSpace()->GetIntRule(eleIndex));
  const int nqp = ir->GetNPoints();

  Vector ind_vals(nqp), rho_vals(nqp);
  ind_->GetValues(eleIndex, ind_vals);
  rho_->GetValues(eleIndex, rho_vals);

  // loop over integration points
  for (int i = 0; i < ir->GetNPoints(); i++)
  {
    // set current integration point
    const IntegrationPoint &ip = ir->IntPoint(i);
    T.SetIntPoint(&ip);

    // evaluate gaussian integration weight
    double w = ip.weight * T.Weight();
    vel_->GetVectorValue(eleIndex, ip, velGP);

    el.CalcShape(ip, N);

    for (int d = 0; d < dim; d++)
    {
      Vector elvect_temp(elvect.GetData() + d*dof, dof);
      elvect_temp.Add( w * ind_vals[i] * rho_vals[i] * velGP(d), N);
    }
  }
}

RemhosHydroHiOpProblem::momentumGradVIntegrator::momentumGradVIntegrator(
  const mfem::QuadratureFunction &ind, const mfem::QuadratureFunction &rho, const int dim)
  : ind_(&ind), rho_(&rho), considerdDim_(dim)
  {}

void RemhosHydroHiOpProblem::momentumGradVIntegrator::AssembleRHSElementVect(
  const FiniteElement &el, ElementTransformation &T, Vector &elvect)
{
  // grab sizes
  int dof = el.GetDof();
  int dim = el.GetDim();
  int eleIndex = T.ElementNo;

  // initialize storage
  Vector N(dof);

  // output vector
  elvect.SetSize(dof*dim);
  elvect = 0.0;

  const IntegrationRule *ir = &(ind_->GetSpace()->GetIntRule(eleIndex));
  const int nqp = ir->GetNPoints();

  Vector ind_vals(nqp), rho_vals(nqp);
  ind_->GetValues(eleIndex, ind_vals);
  rho_->GetValues(eleIndex, rho_vals);

  // loop over integration points
  for (int i = 0; i < ir->GetNPoints(); i++)
  {
    // set current integration point
    const IntegrationPoint &ip = ir->IntPoint(i);
    T.SetIntPoint(&ip);

    // evaluate gaussian integration weight
    double w = ip.weight * T.Weight();
    el.CalcShape(ip, N);

    Vector elvect_temp(elvect.GetData() + considerdDim_*dof, dof);
    elvect_temp.Add( w * ind_vals[i] * rho_vals[i], N);
  }
}

RemhosHydroHiOpProblem::VDiffIntegrator::VDiffIntegrator(
  const mfem::ParGridFunction &v, const mfem::ParGridFunction &v_0, const mfem::QuadratureFunction &ind)
  : v_(&v), v_0_(&v_0), ind_(&ind)
  {}

void RemhosHydroHiOpProblem::VDiffIntegrator::AssembleRHSElementVect(
  const FiniteElement &el, ElementTransformation &T, Vector &elvect)
{
  // grab sizes
  int dof = el.GetDof();
  int dim = el.GetDim();
  int eleIndex = T.ElementNo;

  // initialize storage
  Vector N(dof);
  Vector velGP(dim);
  Vector vel0GP(dim);

  // output vector
  elvect.SetSize(dof*dim);
  elvect = 0.0;

  const IntegrationRule *ir = &(ind_->GetSpace()->GetIntRule(eleIndex));
  const int nqp = ir->GetNPoints();

  // loop over integration points
  for (int i = 0; i < ir->GetNPoints(); i++)
  {
    // set current integration point
    const IntegrationPoint &ip = ir->IntPoint(i);
    T.SetIntPoint(&ip);

    v_->GetVectorValue(eleIndex, ip, velGP);
    v_0_->GetVectorValue(eleIndex, ip, vel0GP);

    // evaluate gaussian integration weight
    double w = ip.weight * T.Weight();
    el.CalcShape(ip, N);

    for( int d = 0; d<dim; d++)
    {
      real_t v_diff_comp = velGP[d] - vel0GP[d];
      Vector elvect_temp(elvect.GetData() + d*dof, dof);
      elvect_temp.Add( w * v_diff_comp, N);
    }
  }
}

PressureDiffGradEIntegrator::PressureDiffGradEIntegrator(
    mfem::QuadratureFunction &rho, const mfem::QuadratureFunction &p0, mfem::ParGridFunction &e)
  : rho_(&rho), p_0_(&p0), e_(&e)
  {}

void PressureDiffGradEIntegrator::AssembleRHSElementVect(
  const FiniteElement &el, ElementTransformation &T, Vector &elvect)
{
  // grab sizes
  int dof = el.GetDof();
  int dim = el.GetDim();
  int eleIndex = T.ElementNo;

  // initialize storage
  Vector N(dof);

  // output vector
  elvect.SetSize(dof);
  elvect = 0.0;

  const IntegrationRule *ir = &(rho_->GetSpace()->GetIntRule(eleIndex));
  const int nqp = ir->GetNPoints();

  Vector p0_vals(nqp), rho_vals(nqp), e_vals(nqp);
  p_0_->GetValues(eleIndex, p0_vals); 
  rho_->GetValues(eleIndex, rho_vals);
  e_->GetValues(T, *ir, e_vals);

  // loop over integration points
  for (int i = 0; i < ir->GetNPoints(); i++)
  {
    // set current integration point
    const IntegrationPoint &ip = ir->IntPoint(i);
    T.SetIntPoint(&ip);

    // evaluate gaussian integration weight
    double w = ip.weight * T.Weight();
    double pressureDiff = rho_vals(i) * e_vals(i) - p0_vals(i);

    //std::cout<<pressureDiff<<std::endl;
    el.CalcShape(ip, N);

    elvect.Add(w * pressureDiff * rho_vals[i], N);
  }
}

RemhosIndRhoVOpProblem::momentumGradVIntegrator::momentumGradVIntegrator(
  const mfem::QuadratureFunction &ind, const mfem::QuadratureFunction &rho, const int dim)
  : ind_(&ind), rho_(&rho), considerdDim_(dim)
  {}

void RemhosIndRhoVOpProblem::momentumGradVIntegrator::AssembleRHSElementVect(
  const FiniteElement &el, ElementTransformation &T, Vector &elvect)
{
  // grab sizes
  int dof = el.GetDof();
  int dim = el.GetDim();
  int eleIndex = T.ElementNo;

  // initialize storage
  Vector N(dof);

  // output vector
  elvect.SetSize(dof*dim);
  elvect = 0.0;

  const IntegrationRule *ir = &(ind_->GetSpace()->GetIntRule(eleIndex));
  const int nqp = ir->GetNPoints();

  Vector ind_vals(nqp), rho_vals(nqp);
  ind_->GetValues(eleIndex, ind_vals);
  rho_->GetValues(eleIndex, rho_vals);

  // loop over integration points
  for (int i = 0; i < ir->GetNPoints(); i++)
  {
    // set current integration point
    const IntegrationPoint &ip = ir->IntPoint(i);
    T.SetIntPoint(&ip);

    // evaluate gaussian integration weight
    double w = ip.weight * T.Weight();
    el.CalcShape(ip, N);

    Vector elvect_temp(elvect.GetData() + considerdDim_*dof, dof);
    elvect_temp.Add( w * ind_vals[i] * rho_vals[i], N);
  }
}

RemhosIndRhoVOpProblem::VDiffIntegrator::VDiffIntegrator(
  const mfem::ParGridFunction &v, const mfem::ParGridFunction &v_0, const mfem::QuadratureFunction &ind)
  : v_(&v), v_0_(&v_0), ind_(&ind)
  {}

void RemhosIndRhoVOpProblem::VDiffIntegrator::AssembleRHSElementVect(
  const FiniteElement &el, ElementTransformation &T, Vector &elvect)
{
  // grab sizes
  int dof = el.GetDof();
  int dim = el.GetDim();
  int eleIndex = T.ElementNo;

  // initialize storage
  Vector N(dof);
  Vector velGP(dim);
  Vector vel0GP(dim);

  // output vector
  elvect.SetSize(dof*dim);
  elvect = 0.0;

  const IntegrationRule *ir = &(ind_->GetSpace()->GetIntRule(eleIndex));
  const int nqp = ir->GetNPoints();

  // loop over integration points
  for (int i = 0; i < ir->GetNPoints(); i++)
  {
    // set current integration point
    const IntegrationPoint &ip = ir->IntPoint(i);
    T.SetIntPoint(&ip);

    v_->GetVectorValue(eleIndex, ip, velGP);
    v_0_->GetVectorValue(eleIndex, ip, vel0GP);

    // evaluate gaussian integration weight
    double w = ip.weight * T.Weight();
    el.CalcShape(ip, N);

    for( int d = 0; d<dim; d++)
    {
      real_t v_diff_comp = velGP[d] - vel0GP[d];
      Vector elvect_temp(elvect.GetData() + d*dof, dof);
      elvect_temp.Add( w * v_diff_comp, N);
    }
  }
}

RemhosEOpProblem::totalEnergyGradEIntegrator::totalEnergyGradEIntegrator(
  const mfem::QuadratureFunction &ind, const mfem::QuadratureFunction &rho)
  : ind_(&ind), rho_(&rho) 
  {}

void RemhosEOpProblem::totalEnergyGradEIntegrator::AssembleRHSElementVect(
  const FiniteElement &el, ElementTransformation &T, Vector &elvect)
{
  // grab sizes
  int dof = el.GetDof();
  int dim = el.GetDim();
  int eleIndex = T.ElementNo;

  // initialize storage
  Vector N(dof);

  // output vector
  elvect.SetSize(dof);
  elvect = 0.0;

  const IntegrationRule *ir = &(ind_->GetSpace()->GetIntRule(eleIndex));
  const int nqp = ir->GetNPoints();

  Vector ind_vals(nqp), rho_vals(nqp);
  ind_->GetValues(eleIndex, ind_vals);
  rho_->GetValues(eleIndex, rho_vals);

  // loop over integration points
  for (int i = 0; i < ir->GetNPoints(); i++)
  {
    // set current integration point
    const IntegrationPoint &ip = ir->IntPoint(i);
    T.SetIntPoint(&ip);

    // evaluate gaussian integration weight
    double w = ip.weight * T.Weight();

    el.CalcShape(ip, N);

    elvect.Add(w * ind_vals[i] * rho_vals[i] , N);
  }
}



} // namespace mfem
