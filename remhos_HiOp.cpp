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



} // namespace mfem
