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

#ifndef MFEM_REMHOS_HiOp
#define MFEM_REMHOS_HiOp

#include "mfem.hpp"

namespace mfem
{

class RhemosHiOpProblem : public OptimizationProblem
{
private:
   const ParGridFunction x_initial;
   ParFiniteElementSpace & fespace;
   const Vector & designVar;
   Vector d_lo, d_hi, massvec;

   double targetMass;
   double H1semiNormweight_;

public:
   RhemosHiOpProblem(ParFiniteElementSpace &space,
                     const ParGridFunction &u_initial,
                     const Vector &design_Var,
                     const Vector &xmin, 
                     const Vector &xmax, 
                     double initalmass,
                     int numConstraints_,
                     double H1semiNormweight)
      : OptimizationProblem(design_Var.Size(), NULL, NULL),
        x_initial(u_initial), fespace(space), designVar(design_Var),
        d_lo(numConstraints_), d_hi(numConstraints_), massvec(numConstraints_), targetMass(initalmass), H1semiNormweight_(H1semiNormweight)
   {

      numConstraints = numConstraints_;
      SetEqualityConstraint(massvec);
      // SetInequalityConstraint(d_lo, d_hi);

      SetSolutionBounds(xmin, xmax);
   }

   virtual double CalcObjective(const Vector &x) const
   {
      ParGridFunction x_diff(&fespace); x_diff = 0.0;
      ParGridFunction x_interpolated(&fespace); x_interpolated.SetFromTrueDofs(x);
      subtract( x_interpolated, x_initial, x_diff);

      ParLinearForm dQdeta(&fespace);
      GridFunctionCoefficient x_diff_coeff(&x_diff);
      ProductCoefficient x_diff_coeffsquared(x_diff_coeff, x_diff_coeff);
      ProductCoefficient half_x_diff_coeffsquared(0.5, x_diff_coeffsquared);

	   	GradientGridFunctionCoefficient GradientCoeff_new(&x_interpolated);
      	GradientGridFunctionCoefficient GradientCoeff_old(&x_initial);
      VectorSumCoefficient new_minus_old_coeff(GradientCoeff_new, GradientCoeff_old, 1.0, -1.0);

      InnerProductCoefficient innerProductCoeff(new_minus_old_coeff,new_minus_old_coeff);

      ProductCoefficient H1SemiNormCoeff(H1semiNormweight_, innerProductCoeff);

      auto *lfi_1 = new DomainLFIntegrator(half_x_diff_coeffsquared);
      auto *lfi_2 = new DomainLFIntegrator(H1SemiNormCoeff);
      dQdeta.AddDomainIntegrator(lfi_1);
      dQdeta.AddDomainIntegrator(lfi_2);
      dQdeta.Assemble();

      ::mfem::ParGridFunction oneGridFunction(&fespace);
      oneGridFunction = 1.0;

      double val = dQdeta(oneGridFunction);

      return val;
   }

   virtual void CalcObjectiveGrad(const Vector &x, Vector &grad) const
   {
      ParGridFunction x_diff(&fespace); x_diff = 0.0;
      ParGridFunction x_interpolated(&fespace); x_interpolated.SetFromTrueDofs(x);
      subtract(1.0, x_interpolated, x_initial, x_diff);

      ParLinearForm dQdeta(&fespace);
      GridFunctionCoefficient x_diff_coeff(&x_diff);

	   	GradientGridFunctionCoefficient GradientCoeff_new(&x_interpolated);
      	GradientGridFunctionCoefficient GradientCoeff_old(&x_initial);
      VectorSumCoefficient new_minus_old_coeff(GradientCoeff_new, GradientCoeff_old, 1.0, -1.0);

       	ScalarVectorProductCoefficient a_timesB(H1semiNormweight_*2.0, new_minus_old_coeff );


      auto *lfi_1 = new DomainLFIntegrator(x_diff_coeff);
      auto *lfi_2 = new DomainLFGradIntegrator(a_timesB);

      dQdeta.AddDomainIntegrator(lfi_1);
      dQdeta.AddDomainIntegrator(lfi_2);
      dQdeta.Assemble();
      dQdeta.ParallelAssemble(grad);
   }

   virtual void CalcConstraintGrad(const int constNumber, const Vector &x, Vector &grad) const
   {
      if( constNumber == 0)
      {
         ParLinearForm dConstdeta(&fespace); dConstdeta = 0.0;
         ConstantCoefficient dConst_coeff(1.0);
         auto *constrlfi = new DomainLFIntegrator(dConst_coeff);

         dConstdeta.AddDomainIntegrator(constrlfi);
         dConstdeta.Assemble();
         dConstdeta.ParallelAssemble(grad);
      }
      else if( constNumber == 1)
      {
         ParGridFunction x_interpolated(&fespace); x_interpolated.SetFromTrueDofs(x);   

         ParLinearForm dConstdeta(&fespace); dConstdeta = 0.0;
         GridFunctionCoefficient dGF_coeff(&x_interpolated);
         auto *constrlfi = new DomainLFIntegrator(dGF_coeff);

         dConstdeta.AddDomainIntegrator(constrlfi);
         dConstdeta.Assemble();
         dConstdeta.ParallelAssemble(grad);

         grad *= 2.0;
      }
   }

   virtual void CalcConstraint(const int constNumber, const Vector &x, Vector &constVal) const
   {
      if( constNumber == 0)
      {
         ParGridFunction x_interpolated(&fespace); x_interpolated.SetFromTrueDofs(x);    
         Vector * pos = fespace.GetParMesh()->GetNodes();

         double mass_s = calculateMass(*pos, x_interpolated);
         constVal[0] = mass_s - targetMass;
      }
      else if( constNumber == 1)
      {
         ParGridFunction x_interpolated(&fespace); x_interpolated.SetFromTrueDofs(x);    
         Vector * pos = fespace.GetParMesh()->GetNodes();

         double mass_s = calculateMass(*pos, x_interpolated);
         constVal[0] = std::pow(mass_s, 2.0) - std::pow(targetMass, 2.0);         
      }
   }

private:

   double calculateMass( const Vector &pos, const ParGridFunction &g) const
   {
      double mass = 0.0;
      const int NE = g.ParFESpace()->GetNE();
      for (int e = 0; e < NE; e++)
      {
         auto el = g.ParFESpace()->GetFE(e);
         auto ir = IntRules.Get(el->GetGeomType(), el->GetOrder() + 2);
         IsoparametricTransformation Tr;
         // Must be w.r.t. the given positions.
         g.ParFESpace()->GetParMesh()->GetElementTransformation(e, pos, &Tr);

         Vector g_vals(ir.GetNPoints());
         g.GetValues(Tr, ir, g_vals);

         for (int q = 0; q < ir.GetNPoints(); q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            Tr.SetIntPoint(&ip);
            mass += Tr.Weight() * ip.weight * g_vals(q);
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, &mass, 1, MPI_DOUBLE, MPI_SUM,
                  g.ParFESpace()->GetComm());
      return mass;
   }
   
};


} // namespace mfem

#endif // MFEM_REMHOS_GHiOp
