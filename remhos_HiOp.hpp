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

void GetOptimizationSubsetInd(
      const mfem::Vector & xmin, const mfem::Vector & xmax, mfem::Array<int> & optInd);

int GetSizeOptimizationSubset(const Vector &xmin, const Vector &xmax);

class PressureDiffGradEIntegrator : public mfem::LinearFormIntegrator
{
public:
   PressureDiffGradEIntegrator(mfem::QuadratureFunction &rho, const mfem::QuadratureFunction &p0, mfem::ParGridFunction &e);
   ~PressureDiffGradEIntegrator(){};
   void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvect);
private:

   mfem::QuadratureFunction *rho_;
   const mfem::QuadratureFunction *p_0_;
   mfem::ParGridFunction    *e_;
};


class RemhosHiOpProblem : public OptimizationProblem
{
private:
   const ParGridFunction x_initial;
   ParFiniteElementSpace & fespace;
   const int NumDesVar_;
   Vector d_lo, d_hi, massvec;

   double targetMass;
   double H1SemiNormWeight = 0.0;
   mfem::Array<int> optProbInd;
   bool subproblem = false;

   mutable ParBilinearForm * mass_form =nullptr;

   mutable hiop::hiopInterfaceBase::WeightedSpaceType weightedSpace_ = 
         hiop::hiopInterfaceBase::WeightedSpaceType::Euclidean;

public:
   RemhosHiOpProblem(ParFiniteElementSpace &space,
                     const ParGridFunction &u_initial,
                     const int &numDesVar,
                     const Vector &xmin, 
                     const Vector &xmax, 
                     double initalmass,
                     int numConstraints_,
                     bool use_H1_semi,
                     const mfem::Array<int> & optProbInd_,
                     bool sub =false)
      : OptimizationProblem(numDesVar, NULL, NULL),
        x_initial(u_initial), fespace(space), NumDesVar_(numDesVar),
        d_lo(numConstraints_), d_hi(numConstraints_), massvec(numConstraints_),
        targetMass(initalmass), optProbInd(optProbInd_), subproblem(sub)
   {
      numConstraints = numConstraints_;
      SetEqualityConstraint(massvec);
      // SetInequalityConstraint(d_lo, d_hi);

      SetSolutionBounds(xmin, xmax);

      if (use_H1_semi)
      {
         double dx = space.GetMesh()->GetElementSize(0, 0);
         MPI_Allreduce(MPI_IN_PLACE, &dx, 1, MPI_DOUBLE,
                       MPI_MIN, space.GetComm());
         H1SemiNormWeight = dx * dx;
      }
   }

   void setWeightedSpaceType( hiop::hiopInterfaceBase::WeightedSpaceType weightedSpace)
   {
      weightedSpace_ = weightedSpace;
   }

   virtual hiop::hiopInterfaceBase::WeightedSpaceType getWeightedSpaceType() const override
   {
      return weightedSpace_; 
   }

   real_t CalcObjective(const Vector &x) const override
   {

      ParGridFunction x_interpolated(&fespace); 

      if(subproblem)
      {
         x_interpolated = x_initial;
         x_interpolated.SetSubVector(optProbInd,x);
      }
      else
      {
         x_interpolated.SetFromTrueDofs(x);
      }

      ParGridFunction x_diff(&fespace); x_diff = 0.0;
      subtract( x_interpolated, x_initial, x_diff);

      // L2 norm (0.5*(u_1-u_0)^2)
      ParLinearForm dQdeta(&fespace);
      GridFunctionCoefficient x_diff_coeff(&x_diff);
      ProductCoefficient x_diff_coeffsquared(x_diff_coeff, x_diff_coeff);
      ProductCoefficient half_x_diff_coeffsquared(0.5, x_diff_coeffsquared);

      // H1-semi norm ((\nable u_1 - \nabla u_0)^2)
	   GradientGridFunctionCoefficient GradientCoeff_new(&x_interpolated);
      GradientGridFunctionCoefficient GradientCoeff_old(&x_initial);
      VectorSumCoefficient new_minus_old_coeff(GradientCoeff_new, GradientCoeff_old, 1.0, -1.0);

      InnerProductCoefficient innerProductCoeff(new_minus_old_coeff,new_minus_old_coeff);
      ProductCoefficient H1SemiNormCoeff(H1SemiNormWeight, innerProductCoeff);

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

   void CalcObjectiveGrad(const Vector &x, Vector &grad) const override
   {

      ParGridFunction x_interpolated(&fespace); 
      if(subproblem)
      {
         x_interpolated = x_initial;
         x_interpolated.SetSubVector(optProbInd,x);
      }
      else
      {
         x_interpolated.SetFromTrueDofs(x);
      }

      ParGridFunction x_diff(&fespace); x_diff = 0.0;
      subtract(1.0, x_interpolated, x_initial, x_diff);

      ParLinearForm dQdeta(&fespace);
      GridFunctionCoefficient x_diff_coeff(&x_diff);

      GradientGridFunctionCoefficient GradientCoeff_new(&x_interpolated);
      GradientGridFunctionCoefficient GradientCoeff_old(&x_initial);
      VectorSumCoefficient new_minus_old_coeff(GradientCoeff_new, GradientCoeff_old, 1.0, -1.0);

      ScalarVectorProductCoefficient a_timesB(H1SemiNormWeight*2.0, new_minus_old_coeff );

      mfem::Vector tempGrad(x_interpolated.Size());

      auto *lfi_1 = new DomainLFIntegrator(x_diff_coeff);
      auto *lfi_2 = new DomainLFGradIntegrator(a_timesB);

      dQdeta.AddDomainIntegrator(lfi_1);
      dQdeta.AddDomainIntegrator(lfi_2);

      dQdeta.Assemble();
      dQdeta.ParallelAssemble(tempGrad);

      if(subproblem)
      {
         tempGrad.GetSubVector(optProbInd,grad);
      }
      else
      {
         grad = tempGrad;
      } 
   }

   virtual void CalcObjectiveM(  std::vector<mfem::Vector> & diagMass, std::vector<HypreParMatrix *> & M_) const override
   {
      if(subproblem)
      {
         mfem_error("CalcObjectiveHessian not implemented for subproblem option");
      }

      M_.resize(1);

      delete(mass_form);
      mass_form = new ParBilinearForm(&fespace);
      auto *blfi = new MassIntegrator();
      mass_form->AddDomainIntegrator(blfi);
      mass_form->Assemble();
      mass_form->Finalize();

      M_[0] = mass_form->ParallelAssemble();
   }
   
   void CalcConstraintGrad(const int constNumber,
                           const Vector &x, Vector &grad) const override
   {
      ParGridFunction x_interpolated(&fespace); 
      if(subproblem)
      {
         x_interpolated = x_initial;
         x_interpolated.SetSubVector(optProbInd,x);
      }
      else
      {
         x_interpolated.SetFromTrueDofs(x);
      }

      mfem::Vector tempGrad(x_interpolated.Size());

      if( constNumber == 0)
      {
         ParLinearForm dConstdeta(&fespace); dConstdeta = 0.0;
         ConstantCoefficient dConst_coeff(1.0);
         auto *constrlfi = new DomainLFIntegrator(dConst_coeff);

         dConstdeta.AddDomainIntegrator(constrlfi);
         dConstdeta.Assemble();
         dConstdeta.ParallelAssemble(tempGrad);
      }
      else if( constNumber == 1)
      {
         ParLinearForm dConstdeta(&fespace); dConstdeta = 0.0;
         GridFunctionCoefficient dGF_coeff(&x_interpolated);
         auto *constrlfi = new DomainLFIntegrator(dGF_coeff);

         dConstdeta.AddDomainIntegrator(constrlfi);
         dConstdeta.Assemble();
         dConstdeta.ParallelAssemble(tempGrad);

         grad *= 2.0;
      }
      if(subproblem){ tempGrad.GetSubVector(optProbInd,grad); }
      else { grad = tempGrad; } 
   }

   void CalcConstraint(const int constNumber,
                       const Vector &x, Vector &constVal) const override
   {
      ParGridFunction x_interpolated(&fespace); 
      if(subproblem)
      {
         x_interpolated = x_initial;
         x_interpolated.SetSubVector(optProbInd,x);;
      }
      else
      {
         x_interpolated.SetFromTrueDofs(x);
      }

      if( constNumber == 0)
      {
         Vector * pos = fespace.GetParMesh()->GetNodes();

         double mass_s = calculateMass(*pos, x_interpolated);
         constVal[0] = mass_s - targetMass;
      }
      else if( constNumber == 1)
      { 
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

class RemhosQuadHiOpProblem : public OptimizationProblem
{
private:
   const QuadratureFunction x_initial;
   const Vector pos_final;
   QuadratureSpace & qspace;
   const QuadratureFunction & designVar;
   Vector d_lo, d_hi, massvec;

   double targetMass;
   double H1SemiNormWeight = 0.0;
   bool isL2_ = true;

public:
   RemhosQuadHiOpProblem(QuadratureSpace &space,
                     const Vector &pos_final_,
                     const QuadratureFunction &u_initial,
                     const QuadratureFunction &design_Var,
                     const Vector &xmin, 
                     const Vector &xmax, 
                     double initalmass,
                     int numConstraints_,
                     bool use_H1_semi,
                     bool isL2 = true)
      : OptimizationProblem(design_Var.Size(), NULL, NULL),
        x_initial(u_initial), pos_final(pos_final_), qspace(space), designVar(design_Var),
        d_lo(numConstraints_), d_hi(numConstraints_), massvec(numConstraints_),
        targetMass(initalmass), isL2_(isL2)
   {

      numConstraints = numConstraints_;
      SetEqualityConstraint(massvec);
      // SetInequalityConstraint(d_lo, d_hi);

      SetSolutionBounds(xmin, xmax);

      if (use_H1_semi)
      {
         double dx = space.GetMesh()->GetElementSize(0, 0);
         MPI_Allreduce(MPI_IN_PLACE, &dx, 1, MPI_DOUBLE,
                       MPI_MIN, MPI_COMM_WORLD);
         H1SemiNormWeight = dx * dx;
      }
   }

   virtual hiop::hiopInterfaceBase::WeightedSpaceType getWeightedSpaceType() const override
   {
      return hiop::hiopInterfaceBase::WeightedSpaceType::Euclidean; 
   }

virtual double CalcObjective(const Vector &x) const override
   {
      QuadratureFunction x_diff(&qspace); x_diff = 0.0;
      subtract( x, x_initial, x_diff);

      real_t normSq = 0.0;

      if(isL2_)
      {
         auto mesh = qspace.GetMesh();
         const int NE = mesh->GetNE();

         Array<int> offset(NE+1);
         offset[0] = 0;

         for (int e = 0; e < NE; e++)
         {            
            const IntegrationRule &ir = qspace.GetElementIntRule(e);
            const int nqp = ir.GetNPoints();

            offset[e+1] = offset[e] + nqp;
         }

         for (int e = 0; e < NE; e++)
         {
            const int s_offset = offset[e];

            IsoparametricTransformation Tr;
            mesh->GetElementTransformation(e, pos_final, &Tr);

            const IntegrationRule &ir = qspace.GetElementIntRule(e);
            const int nqp = ir.GetNPoints();

            for (int q = 0; q < nqp; q++)
            {
               const IntegrationPoint &ip = ir.IntPoint(q);
               Tr.SetIntPoint(&ip);
               real_t w = Tr.Weight() * ip.weight;

               normSq += 0.5* w *x_diff[s_offset+q] * x_diff[s_offset+q];
            }
         }
      }
      else{
         normSq = 0.5*x_diff.Norml2() * x_diff.Norml2();

      }

      MPI_Allreduce(MPI_IN_PLACE, &normSq, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);

      return normSq;
   }

   virtual void CalcObjectiveGrad(const Vector &x, Vector &grad) const override
   {
      QuadratureFunction x_diff(&qspace); x_diff = 0.0;
      subtract( x, x_initial, x_diff);

      if(isL2_)
      {
         auto mesh = qspace.GetMesh();
         const int NE = mesh->GetNE();

         Array<int> offset(NE+1);
         offset[0] = 0;

         for (int e = 0; e < NE; e++)
         {            
            const IntegrationRule &ir = qspace.GetElementIntRule(e);
            const int nqp = ir.GetNPoints();

            offset[e+1] = offset[e] + nqp;
         }

         for (int e = 0; e < NE; e++)
         {
            const int s_offset = offset[e];

            IsoparametricTransformation Tr;
            mesh->GetElementTransformation(e, pos_final, &Tr);

            const IntegrationRule &ir = qspace.GetElementIntRule(e);
            const int nqp = ir.GetNPoints();

            for (int q = 0; q < nqp; q++)
            {
               const IntegrationPoint &ip = ir.IntPoint(q);
               Tr.SetIntPoint(&ip);
               real_t w = Tr.Weight() * ip.weight;

               x_diff[s_offset+q] = w * x_diff[s_offset+q];
            }
         }
      }

      grad = x_diff;
   }

   virtual void CalcConstraintGrad(const int constNumber, const Vector &x, Vector &grad) const override
   {
      if( constNumber == 0)
      {
         grad = 0.0;
         auto mesh = qspace.GetMesh();
         const int NE = mesh->GetNE();

         Array<int> offset(NE+1);
         offset[0] = 0;

         for (int e = 0; e < NE; e++)
         {            
            const IntegrationRule &ir = qspace.GetElementIntRule(e);
            const int nqp = ir.GetNPoints();

            offset[e+1] = offset[e] + nqp;
         }

         for (int e = 0; e < NE; e++)
         {
            const int s_offset = offset[e];

            IsoparametricTransformation Tr;
            mesh->GetElementTransformation(e, pos_final, &Tr);

            const IntegrationRule &ir = qspace.GetElementIntRule(e);
            const int nqp = ir.GetNPoints();

            for (int q = 0; q < nqp; q++)
            {
               const IntegrationPoint &ip = ir.IntPoint(q);
               Tr.SetIntPoint(&ip);

               grad[s_offset+q] = Tr.Weight() * ip.weight;

            }
         }
      }

   }

   virtual void CalcConstraint(const int constNumber, const Vector &x, Vector &constVal) const override
   {
      if( constNumber == 0)
      {
         QuadratureFunction u_desing(qspace);
         u_desing =  x;

         double mass_s = Integrate(pos_final, &u_desing, nullptr, nullptr);

         constVal[0] = mass_s - targetMass;
      }
   }

private:

double Integrate(const Vector &pos,
                 const QuadratureFunction *q1,
                 const QuadratureFunction *q2,
                 const ParGridFunction *g1) const
{
   MFEM_VERIFY(q1 || q2 || g1, "At least one function must be specified.");

   const QuadratureSpace *qspace = nullptr;
   if (q1) { qspace = dynamic_cast<const QuadratureSpace *>(q1->GetSpace()); }
   if (q2) { qspace = dynamic_cast<const QuadratureSpace *>(q2->GetSpace()); }

   auto mesh = (qspace) ? qspace->GetMesh() : g1->ParFESpace()->GetMesh();
   const int NE = mesh->GetNE();
   double integral = 0.0;
   for (int e = 0; e < NE; e++)
   {
      const IntegrationRule &ir =
          (qspace) ? qspace->GetElementIntRule(e)
                   : IntRules.Get(g1->ParFESpace()->GetFE(e)->GetGeomType(), 7);
      const int nqp = ir.GetNPoints();

      // Transformation w.r.t. the given mesh positions.
      IsoparametricTransformation Tr;
      mesh->GetElementTransformation(e, pos, &Tr);

      Vector q1_vals(nqp), q2_vals(nqp), g1_vals(nqp);
      if (q1) { q1->GetValues(e, q1_vals); } else { q1_vals = 1.0; }
      if (q2) { q2->GetValues(e, q2_vals); } else { q2_vals = 1.0; }
      if (g1) { g1->GetValues(Tr, ir, g1_vals); } else { g1_vals = 1.0; }

      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         integral += Tr.Weight() * ip.weight *
                     q1_vals(q) * q2_vals(q) * g1_vals(q);
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &integral, 1, MPI_DOUBLE, MPI_SUM,
                 MPI_COMM_WORLD);
   return integral;
}
   
};

class RemhosIndRhoEHiOpProblem : public OptimizationProblem
{
private:
   const Vector x_initial;
   mfem::QuadratureFunction p_initial_;
   const Vector &pos_final;
   QuadratureSpace & qspace_;
   ParFiniteElementSpace & fespace_;
   const int    numDesVar_;
   Vector d_lo, d_hi, massvec;

   double targetVol;
   double targetMass;
   double targetEnergy;
   double H1SemiNormWeight = 0.0;
   bool isL2_ = true;

   const int size_qf;
   const int size_gf;

   Array<int> offset_;

   real_t w_1 = 1e0;
   real_t w_2 = 1.0;
   real_t w_3 = 1.0;
   real_t w_p = 1e3;

   mfem::Array<int> optProbInd;
   bool subproblem = false;
   bool pressureOpt = false;

   mutable ParBilinearForm * mass_form =nullptr;

   mutable hiop::hiopInterfaceBase::WeightedSpaceType weightedSpace_ = 
         hiop::hiopInterfaceBase::WeightedSpaceType::Euclidean;

   class EnergyGradIntegrator : public mfem::LinearFormIntegrator
   {
   public:
      EnergyGradIntegrator(const mfem::QuadratureFunction &ind, const mfem::QuadratureFunction &rho);
      ~EnergyGradIntegrator(){};
      void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvect);
   private:

      const mfem::QuadratureFunction *ind_;
      const mfem::QuadratureFunction *rho_;
   };

public:
   RemhosIndRhoEHiOpProblem(QuadratureSpace       & qspace,
                            ParFiniteElementSpace & fespace,
                            const Vector          & pos_final_,
                            const Vector          & u_initial,
                            mfem::QuadratureFunction & p_initial,
                            const int             & numDesVar,
                            const Vector          & xmin, 
                            const Vector          & xmax, 
                            const double          & initalvol,
                            const double          & initalmass,
                            const double          & initalenergy,
                            const int             & numConstraints_,
                            const bool            & use_H1_semi,
                            const mfem::Array<int> & optProbInd_,
                            const bool            & isL2 = true,
                            const bool            & sub =false,
                            const bool            & pOpt = false)
      : OptimizationProblem(numDesVar, NULL, NULL),
        x_initial(u_initial), p_initial_(p_initial), pos_final(pos_final_), qspace_(qspace), fespace_(fespace), numDesVar_(numDesVar),
        d_lo(numConstraints_), d_hi(numConstraints_), massvec(numConstraints_),
        targetVol(initalvol), targetMass(initalmass), targetEnergy(initalenergy), isL2_(isL2),
        size_qf(qspace.GetSize()), size_gf(fespace.GetNDofs()), offset_(4), optProbInd(optProbInd_), subproblem(sub),
        pressureOpt(pOpt)
   {
      numConstraints = numConstraints_;
      SetEqualityConstraint(massvec);
      // SetInequalityConstraint(d_lo, d_hi);

      SetSolutionBounds(xmin, xmax);

      offset_[0] = 0;
      offset_[1] = offset_[0] + size_qf ;
      offset_[2] = offset_[1] + size_qf;
      offset_[3] = offset_[2] + size_gf;
   }

   void setWeightedSpaceType( hiop::hiopInterfaceBase::WeightedSpaceType weightedSpace)
   {
      weightedSpace_ = weightedSpace;
   }

   virtual hiop::hiopInterfaceBase::WeightedSpaceType getWeightedSpaceType() const override
   {
      return weightedSpace_; 
   }

   double CalcObjective(const Vector &x) const override
   {
      Vector x_interpolated(offset_[3]);  

      if(subproblem)
      {
         x_interpolated = x_initial;
         x_interpolated.SetSubVector(optProbInd,x);
      }
      else
      {
         x_interpolated = x;
      }

      QuadratureFunction ind(&qspace_, x_interpolated.GetData());
      QuadratureFunction rho(&qspace_, x_interpolated.GetData() + size_qf);
      ParGridFunction    energy  (&fespace_, x_interpolated.GetData() + 2*size_qf);
   
      QuadratureFunction ind_0(&qspace_, x_initial.GetData());
      QuadratureFunction rho_0(&qspace_, x_initial.GetData() + size_qf);
      ParGridFunction    e_0  (&fespace_, x_initial.GetData() + 2*size_qf);

      QuadratureFunction ind_diff(&qspace_);
      QuadratureFunction roh_diff(&qspace_);
      ParGridFunction    e_diff(&fespace_);

      subtract( ind, ind_0, ind_diff);
      subtract( rho, rho_0, roh_diff);
      subtract( energy, e_0, e_diff);

      //-------------------------------------------------------------------

      real_t normindSq = 0.0;
      real_t normrohSq = 0.0;
      
      if(isL2_)
      {
         auto mesh = qspace_.GetMesh();
         const int NE = mesh->GetNE();

         Array<int> offset(NE+1);
         offset[0] = 0;

         for (int e = 0; e < NE; e++)
         {
            const IntegrationRule &ir = qspace_.GetElementIntRule(e);
            const int nqp = ir.GetNPoints();

            offset[e+1] = offset[e] + nqp;
         }

         for (int e = 0; e < NE; e++)
         {
            const int s_offset = offset[e];

            IsoparametricTransformation Tr;
            mesh->GetElementTransformation(e, pos_final, &Tr);

            const IntegrationRule &ir = qspace_.GetElementIntRule(e);
            const int nqp = ir.GetNPoints();

            for (int q = 0; q < nqp; q++)
            {
               const IntegrationPoint &ip = ir.IntPoint(q);
               Tr.SetIntPoint(&ip);
               real_t w = Tr.Weight() * ip.weight;

               normindSq += 0.5* w *ind_diff[s_offset+q] * ind_diff[s_offset+q];
               normrohSq += 0.5* w *roh_diff[s_offset+q] * roh_diff[s_offset+q];
            }
         }
      }
      else{
         normindSq = 0.5 * ind_diff.Norml2() * ind_diff.Norml2();
         normrohSq = 0.5 * roh_diff.Norml2() * roh_diff.Norml2();
      }

      MPI_Allreduce(MPI_IN_PLACE, &normindSq, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

      MPI_Allreduce(MPI_IN_PLACE, &normrohSq, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

      //-------------------------------------------------------------------

      // L2 norm (0.5*(u_1-u_0)^2)
      ParLinearForm dQdeta(&fespace_);
      GridFunctionCoefficient e_diff_coeff(&e_diff);
      ProductCoefficient e_diff_coeffsquared(e_diff_coeff, e_diff_coeff);
      ProductCoefficient half_e_diff_coeffsquared(0.5, e_diff_coeffsquared);

      auto *lfi_1 = new DomainLFIntegrator(half_e_diff_coeffsquared);
      dQdeta.AddDomainIntegrator(lfi_1);
      dQdeta.Assemble();

      ::mfem::ParGridFunction oneGridFunction(&fespace_);
      oneGridFunction = 1.0;

      real_t val = dQdeta(oneGridFunction);

      //std::cout<<"ind: "<< normindSq<<" | rho: "<< normrohSq<<"| energy: "<<val <<std::endl;
      // std::cout<<"ind: "<< normindSq<<std::endl;
      // std::cout<<"rho: "<< normrohSq<<std::endl;
      // std::cout<<"e: "<< val<<std::endl;

      real_t pDiffSq = 0.0;
      if(pressureOpt)
      {
         pDiffSq = IntegratePressureDiff(pos_final, &p_initial_, &rho, &energy);
      } 

      return w_1*normindSq + w_2*normrohSq +w_3* val + w_p*pDiffSq;
   }

   void CalcObjectiveGrad(const Vector &x, Vector &grad) const  override
   {
      Vector x_interpolated(offset_[3]);  

      if(subproblem)
      {
         x_interpolated = x_initial;
         x_interpolated.SetSubVector(optProbInd,x);
      }
      else
      {
         x_interpolated = x;
      }

      QuadratureFunction ind(&qspace_, x_interpolated.GetData());
      QuadratureFunction rho(&qspace_, x_interpolated.GetData() + size_qf);
      ParGridFunction    energy  (&fespace_, x_interpolated.GetData() + 2*size_qf);

      BlockVector ind_rho_e_grad(offset_);

      QuadratureFunction ind_0(&qspace_, x_initial.GetData());
      QuadratureFunction rho_0(&qspace_, x_initial.GetData() + size_qf);
      ParGridFunction    e_0  (&fespace_, x_initial.GetData() + 2*size_qf);

      QuadratureFunction ind_diff(&qspace_);
      QuadratureFunction roh_diff(&qspace_);
      ParGridFunction    e_diff(&fespace_);

      subtract( ind, ind_0, ind_diff);
      subtract( rho, rho_0, roh_diff);
      subtract( energy, e_0, e_diff);

      QuadratureFunction pGradRho(&qspace_); pGradRho = 0.0;

      //------------------------------------------------------------------------

      if(isL2_)
      {
         auto mesh = qspace_.GetMesh();
         const int NE = mesh->GetNE();

         Array<int> offset(NE+1);
         offset[0] = 0;

         for (int e = 0; e < NE; e++)
         {
            const IntegrationRule &ir = qspace_.GetElementIntRule(e);
            const int nqp = ir.GetNPoints();

            offset[e+1] = offset[e] + nqp;
         }

         for (int e = 0; e < NE; e++)
         {
            const int s_offset = offset[e];

            IsoparametricTransformation Tr;
            mesh->GetElementTransformation(e, pos_final, &Tr);

            const IntegrationRule &ir = qspace_.GetElementIntRule(e);
            const int nqp = ir.GetNPoints();

            for (int q = 0; q < nqp; q++)
            {
               const IntegrationPoint &ip = ir.IntPoint(q);
               Tr.SetIntPoint(&ip);
               real_t w = Tr.Weight() * ip.weight;

               ind_diff[s_offset+q] *= w;
               roh_diff[s_offset+q] *= w;

               if(pressureOpt)
               {
                  Vector p0_vals(nqp), rho_vals(nqp), e_vals(nqp);
                  rho.GetValues(e, rho_vals); 
                  p_initial_.GetValues(e, p0_vals);
                  energy.GetValues(Tr, ir, e_vals);

                  double pressureDiff = 0.4 *rho_vals(q) * e_vals(q) - p0_vals(q);

                  pGradRho[s_offset+q] = 0.4*w * pressureDiff * e_vals(q);
               }
            }
         }
      }

      //------------------------------------------------------------------------

      ParLinearForm dQdeta(&fespace_);
      ParGridFunction    e_grad(&fespace_);
      ParGridFunction    p_e_grad(&fespace_); p_e_grad = 0.0;
      GridFunctionCoefficient e_diff_coeff(&e_diff);

      auto *lfi_1 = new DomainLFIntegrator(e_diff_coeff);

      dQdeta.AddDomainIntegrator(lfi_1);
      dQdeta.Assemble();
      dQdeta.ParallelAssemble(e_grad);

      ind_diff *= w_1;
      roh_diff *= w_2;
      e_grad   *= w_3;

      if(pressureOpt)
      {
         ParLinearForm pressureGradELF(&fespace_);
         mfem::LinearFormIntegrator *lfi_1 =
             new mfem::PressureDiffGradEIntegrator( rho, p_initial_, energy);

         pressureGradELF.AddDomainIntegrator(lfi_1);
         pressureGradELF.Assemble();
         pressureGradELF.ParallelAssemble(p_e_grad);

         p_e_grad *= w_p;
         pGradRho *= w_p;

         e_grad += p_e_grad;
         roh_diff += pGradRho;
      }

      ind_rho_e_grad.GetBlock(0) = ind_diff;
      ind_rho_e_grad.GetBlock(1) = roh_diff;
      ind_rho_e_grad.GetBlock(2) = e_grad;

      //       ind_rho_e_grad.GetBlock(1) = pGradRho;
      // ind_rho_e_grad.GetBlock(2) = p_e_grad;

      if(subproblem)
      {
         ind_rho_e_grad.GetSubVector(optProbInd,grad);
      }
      else
      {
         grad = ind_rho_e_grad;
      } 
   }

virtual void CalcObjectiveM(  std::vector<mfem::Vector> & diagMass, std::vector<HypreParMatrix *> & M_) const override
   {
      if(subproblem)
      {
         mfem_error("CalcObjectiveHessian not implemented for subproblem option");
      }

      diagMass.resize(2);
      M_.resize(1);

      QuadratureFunction ind_w(&qspace_); ind_w = 1.0;
      QuadratureFunction roh_w(&qspace_); roh_w = 1.0;
      ParGridFunction    e_diff(&fespace_);

      diagMass[0].SetSize(ind_w.Size());
      diagMass[1].SetSize(ind_w.Size());
      //------------------------------------------------------------------------

      if(isL2_)
      {
         auto mesh = qspace_.GetMesh();
         const int NE = mesh->GetNE();

         Array<int> offset(NE+1);
         offset[0] = 0;

         for (int e = 0; e < NE; e++)
         {
            const IntegrationRule &ir = qspace_.GetElementIntRule(e);
            const int nqp = ir.GetNPoints();

            offset[e+1] = offset[e] + nqp;
         }

         for (int e = 0; e < NE; e++)
         {
            const int s_offset = offset[e];

            IsoparametricTransformation Tr;
            mesh->GetElementTransformation(e, pos_final, &Tr);

            const IntegrationRule &ir = qspace_.GetElementIntRule(e);
            const int nqp = ir.GetNPoints();

            for (int q = 0; q < nqp; q++)
            {
               const IntegrationPoint &ip = ir.IntPoint(q);
               Tr.SetIntPoint(&ip);
               real_t w = Tr.Weight() * ip.weight;

               ind_w[s_offset+q] *= w;
               roh_w[s_offset+q] *= w;
            }
         }
      }
      diagMass[0] = ind_w;
      diagMass[1] = roh_w;

      delete(mass_form);
      mass_form = new ParBilinearForm(&fespace_);
      auto *blfi = new MassIntegrator();
      mass_form->AddDomainIntegrator(blfi);
      mass_form->Assemble();
      mass_form->Finalize();

      M_[0] = mass_form->ParallelAssemble();
   }

   void CalcConstraintGrad(const int constNumber,
                           const Vector &x, Vector &grad) const override
   {
      Vector x_interpolated(offset_[3]);  

      if(subproblem)
      {
         x_interpolated = x_initial;
         x_interpolated.SetSubVector(optProbInd,x);
      }
      else
      {
         x_interpolated = x;
      }

      QuadratureFunction ind(&qspace_, x_interpolated.GetData());
      QuadratureFunction rho(&qspace_, x_interpolated.GetData() + size_qf);
      ParGridFunction    energy  (&fespace_, x_interpolated.GetData() + 2*size_qf);

      BlockVector ind_rho_e_grad(offset_);
      QuadratureFunction ind_grad(&qspace_); ind_grad= 0.0;
      QuadratureFunction rho_grad(&qspace_); rho_grad= 0.0;
      ParGridFunction    e_grad(&fespace_);  e_grad= 0.0;

      grad = 0.0;

      auto mesh = qspace_.GetMesh();
      const int NE = mesh->GetNE();

      Array<int> offsetGP(NE+1);
      offsetGP[0] = 0;

      for (int e = 0; e < NE; e++)
      {            
         const IntegrationRule &ir = qspace_.GetElementIntRule(e);
         const int nqp = ir.GetNPoints();

         offsetGP[e+1] = offsetGP[e] + nqp;
      }

      IsoparametricTransformation Tr;

      for (int e = 0; e < NE; e++)
      {
         mesh->GetElementTransformation(e, pos_final, &Tr);

         const IntegrationRule &ir = qspace_.GetElementIntRule(e);
         const int nqp = ir.GetNPoints();

         for (int q = 0; q < nqp; q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            Tr.SetIntPoint(&ip);

            double w = Tr.Weight() * ip.weight;

            if( constNumber == 0)
            {
               ind_grad[offsetGP[e]+q] = w;
            }
            else if( constNumber == 1)
            {
               ind_grad[offsetGP[e]+q] = w * rho[offsetGP[e]+q];
               rho_grad[offsetGP[e]+q] = w * ind[offsetGP[e]+q];
            }
            else if( constNumber == 2)
            {
               double e_val = energy.GetValue(Tr, ip);
 
               ind_grad[offsetGP[e]+q] = w * rho[offsetGP[e]+q] * e_val;
               rho_grad[offsetGP[e]+q] = w * ind[offsetGP[e]+q] * e_val;
            }
            else{mfem_error("Constraint index does not exist.");}
         }
      }

      if( constNumber == 2)
      {
         ParLinearForm energyGradLF(&fespace_);
         mfem::LinearFormIntegrator *lfi_1 =
             new mfem::RemhosIndRhoEHiOpProblem::EnergyGradIntegrator(ind, rho);

         energyGradLF.AddDomainIntegrator(lfi_1);
         energyGradLF.Assemble();
         energyGradLF.ParallelAssemble(e_grad);
      }

      ind_rho_e_grad.GetBlock(0) = ind_grad;
      ind_rho_e_grad.GetBlock(1) = rho_grad;
      ind_rho_e_grad.GetBlock(2) = e_grad;

      if(subproblem)
      {
         ind_rho_e_grad.GetSubVector(optProbInd,grad);
      }
      else
      {
         grad = ind_rho_e_grad;
      } 
};

void CalcConstraint(const int constNumber,
                    const Vector &x, Vector &constVal) const override
{
      Vector x_interpolated(offset_[3]);  

      if(subproblem)
      {
         x_interpolated = x_initial;
         x_interpolated.SetSubVector(optProbInd,x);
      }
      else
      {
         x_interpolated = x;
      }

      QuadratureFunction ind(&qspace_, x_interpolated.GetData());
      QuadratureFunction rho(&qspace_, x_interpolated.GetData() + size_qf);
      ParGridFunction    energy  (&fespace_, x_interpolated.GetData() + 2*size_qf);

      if( constNumber == 0 )
      {
         double vol_s = Integrate(pos_final, &ind, nullptr, nullptr);

         constVal[0] = vol_s - targetVol;
      }
      else if( constNumber == 1)
      {
         double mass_s = Integrate(pos_final, &ind, &rho, nullptr);

         constVal[1] = mass_s - targetMass;
      }
      else if( constNumber == 2)
      {
         double energy_s = Integrate(pos_final, &ind, &rho, &energy);

         constVal[2] = energy_s - targetEnergy;
      }
      else{mfem_error("Constraint index does not exist.");}
   };

private:

double Integrate(const Vector &pos,
                 const QuadratureFunction *q1,
                 const QuadratureFunction *q2,
                 const ParGridFunction *g1) const
{
   MFEM_VERIFY(q1 || q2 || g1, "At least one function must be specified.");

   const QuadratureSpace *qspace = nullptr;
   if (q1) { qspace = dynamic_cast<const QuadratureSpace *>(q1->GetSpace()); }
   if (q2) { qspace = dynamic_cast<const QuadratureSpace *>(q2->GetSpace()); }

   auto mesh = (qspace) ? qspace->GetMesh() : g1->ParFESpace()->GetMesh();
   const int NE = mesh->GetNE();
   double integral = 0.0;
   for (int e = 0; e < NE; e++)
   {
      const IntegrationRule &ir =
          (qspace) ? qspace->GetElementIntRule(e)
                   : IntRules.Get(g1->ParFESpace()->GetFE(e)->GetGeomType(), 7);
      const int nqp = ir.GetNPoints();

      // Transformation w.r.t. the given mesh positions.
      IsoparametricTransformation Tr;
      mesh->GetElementTransformation(e, pos, &Tr);

      Vector q1_vals(nqp), q2_vals(nqp), g1_vals(nqp);
      if (q1) { q1->GetValues(e, q1_vals); } else { q1_vals = 1.0; }
      if (q2) { q2->GetValues(e, q2_vals); } else { q2_vals = 1.0; }
      if (g1) { g1->GetValues(Tr, ir, g1_vals); } else { g1_vals = 1.0; }

      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         integral += Tr.Weight() * ip.weight *
                     q1_vals(q) * q2_vals(q) * g1_vals(q);
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &integral, 1, MPI_DOUBLE, MPI_SUM,
                 MPI_COMM_WORLD);
   return integral;
}

double IntegratePressureDiff(const Vector &pos,
                           const QuadratureFunction *p0_,
                           const QuadratureFunction *rho_,
                           const ParGridFunction *e_) const
{
   MFEM_VERIFY(rho_ && p0_ && e_, "All function must be specified.");

   const QuadratureSpace *qspace = nullptr;
   if (rho_) { qspace = dynamic_cast<const QuadratureSpace *>(rho_->GetSpace()); }

   auto mesh = qspace->GetMesh();
   const int NE = mesh->GetNE();
   double integral = 0.0;
   for (int e = 0; e < NE; e++)
   {
      const IntegrationRule &ir = qspace->GetElementIntRule(e);
      const int nqp = ir.GetNPoints();

      // Transformation w.r.t. the given mesh positions.
      IsoparametricTransformation Tr;
      mesh->GetElementTransformation(e, pos, &Tr);

      Vector p0_vals(nqp), rho_vals(nqp), e_vals(nqp);
      p0_->GetValues(e, p0_vals); 
      rho_->GetValues(e, rho_vals);
      e_->GetValues(Tr, ir, e_vals);

      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         double pressureDiff = 0.4 * rho_vals(q) * e_vals(q) - p0_vals(q);
         integral += 0.5 * Tr.Weight() * ip.weight * pressureDiff * pressureDiff;
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &integral, 1, MPI_DOUBLE, MPI_SUM,
                 MPI_COMM_WORLD);
   return integral;
}
   
};

class RemhosHydroHiOpProblem : public OptimizationProblem
{
private:
   const Vector x_initial;
   const Vector &pos_final;
   QuadratureSpace & qspace_;
   ParFiniteElementSpace & scalarfespace_;
   ParFiniteElementSpace & vectorfespace_;
   const int    numDesVar_;
   Vector d_lo, d_hi, massvec;

   mfem::QuadratureFunction p_initial_;

   double targetVol;
   double targetMass;
   Vector targetMomentum;
   double targetEnergy;
   double H1SemiNormWeight = 0.0;
   bool isL2_ = true;

   const int size_qf;
   const int size_gf;
   const int size_gf_vec;
   const int size_gf_vec_true;

   Array<int> offset_;
   Array<int> offsetGP;

   real_t w_1 = 1e1;
   real_t w_2 = 1e1;
   real_t w_3 = 1e1;
   real_t w_4 = 1e1;
   real_t w_p = 1e3;

   mfem::Array<int> optProbInd;
   bool subproblem = false;
   bool pressureOpt = false;

   mutable ParBilinearForm * mass_form =nullptr;

   int spatialDim = 0;
   int NE_ = 0;

   Mesh * mesh_ = nullptr;

   mutable hiop::hiopInterfaceBase::WeightedSpaceType weightedSpace_ = 
         hiop::hiopInterfaceBase::WeightedSpaceType::Euclidean;

   QuadratureFunction ind_0;
   QuadratureFunction rho_0;
   ParGridFunction e_0;
   //ParGridFunction v_0;

class totalEnergyGradEIntegrator : public mfem::LinearFormIntegrator {
public:
  totalEnergyGradEIntegrator(const mfem::QuadratureFunction &ind, const mfem::QuadratureFunction &rho);
  ~totalEnergyGradEIntegrator(){};
  void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvect);
private:

  const mfem::QuadratureFunction *ind_;
  const mfem::QuadratureFunction *rho_;
};

class totalEnergyGradVIntegrator : public mfem::LinearFormIntegrator {
public:
  totalEnergyGradVIntegrator(const mfem::QuadratureFunction &ind, const mfem::QuadratureFunction &rho, const mfem::ParGridFunction &vel);
  ~totalEnergyGradVIntegrator(){};
  void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvect);
private:

  const mfem::QuadratureFunction *ind_;
  const mfem::QuadratureFunction *rho_;
  const mfem::ParGridFunction *vel_;
};

class momentumGradVIntegrator : public mfem::LinearFormIntegrator {
public:
  momentumGradVIntegrator(const mfem::QuadratureFunction &ind, const mfem::QuadratureFunction &rho, const int dim);
  ~momentumGradVIntegrator(){};
  void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvect);
private:

  const mfem::QuadratureFunction *ind_;
  const mfem::QuadratureFunction *rho_;
  int considerdDim_;
};

class VDiffIntegrator : public mfem::LinearFormIntegrator {
public:
  VDiffIntegrator(const mfem::ParGridFunction &v, const mfem::ParGridFunction &v_0,const mfem::QuadratureFunction &ind);
  ~VDiffIntegrator(){};
  void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvect);
private:

  const mfem::ParGridFunction *v_;
  const mfem::ParGridFunction *v_0_;
  const mfem::QuadratureFunction *ind_;
};



public:
   RemhosHydroHiOpProblem(  QuadratureSpace        & qspace,
                           ParFiniteElementSpace & scalarfespace,
                           ParFiniteElementSpace & vectorfespace,
                            const Vector          & pos_final_,
                            const Vector          & u_initial,
                            mfem::QuadratureFunction & p_initial,
                            const int             & numDesVar,
                            const Vector          & xmin, 
                            const Vector          & xmax, 
                            const double          & initalvol,
                            const double          & initalmass,
                            const Vector          & initalmomentum,
                            const double          & initalenergy,
                            const int             & numConstraints_,
                            const bool            & use_H1_semi,
                            const mfem::Array<int> & optProbInd_,
                            const bool            & isL2 = true,
                            const bool            & sub =false,
                            const bool            & pOpt = false)
      : OptimizationProblem(numDesVar, NULL, NULL),
        x_initial(u_initial), p_initial_(p_initial), pos_final(pos_final_), qspace_(qspace), scalarfespace_(scalarfespace), vectorfespace_(vectorfespace), numDesVar_(numDesVar),
        d_lo(numConstraints_), d_hi(numConstraints_), massvec(numConstraints_),
        targetVol(initalvol), targetMass(initalmass), targetMomentum(initalmomentum), targetEnergy(initalenergy), isL2_(isL2),
        size_qf(qspace.GetSize()), size_gf(scalarfespace.GetVSize()), size_gf_vec(vectorfespace.GetVSize()),
        size_gf_vec_true(vectorfespace.GetTrueVSize()), offset_(5), optProbInd(optProbInd_), subproblem(sub), pressureOpt(pOpt)
   {
      numConstraints = numConstraints_;
      SetEqualityConstraint(massvec);
      // SetInequalityConstraint(d_lo, d_hi);

      SetSolutionBounds(xmin, xmax);
     
      spatialDim = scalarfespace_.GetMesh()->SpaceDimension ();

      offset_[0] = 0;
      offset_[1] = offset_[0] + size_qf;
      offset_[2] = offset_[1] + size_qf;
      offset_[3] = offset_[2] + size_gf;
      offset_[4] = offset_[3] + size_gf_vec_true;

      mesh_ = qspace_.GetMesh();
      NE_ = mesh_->GetNE();

      offsetGP.SetSize(NE_+1);
      offsetGP[0] = 0;

      for (int e = 0; e < NE_; e++)
      {            
         const IntegrationRule &ir = qspace_.GetElementIntRule(e);
         const int nqp = ir.GetNPoints();

         offsetGP[e+1] = offsetGP[e] + nqp;
      }

      ind_0 = QuadratureFunction(&qspace_, x_initial.GetData());
      rho_0 = QuadratureFunction(&qspace_, x_initial.GetData() + size_qf);
      e_0.MakeRef(&scalarfespace_, x_initial.GetData() + 2*size_qf);
      // v_0   = ParGridFunction(&vectorfespace_);

      // Vector v_0_true(x_initial.GetData() + 2*size_qf + size_gf, size_gf_vec_true);
      // v_0.SetFromTrueDofs(v_0_true);
   }

   void setWeightedSpaceType( hiop::hiopInterfaceBase::WeightedSpaceType weightedSpace)
   {
      weightedSpace_ = weightedSpace;
   }

   virtual hiop::hiopInterfaceBase::WeightedSpaceType getWeightedSpaceType() const override
   {
      return weightedSpace_; 
   }

   double CalcObjective(const Vector &x) const override
   {
      Vector x_interpolated(offset_[4]);  

      if(subproblem)
      {
         x_interpolated = x_initial;
         x_interpolated.SetSubVector(optProbInd,x);
      }
      else
      {
         x_interpolated = x;
      }

      QuadratureFunction ind(&qspace_, x_interpolated.GetData());
      QuadratureFunction rho(&qspace_, x_interpolated.GetData() + size_qf);
      ParGridFunction    energy  (&scalarfespace_, x_interpolated.GetData() + 2*size_qf);
      
      QuadratureFunction ind_diff(&qspace_);
      QuadratureFunction roh_diff(&qspace_);
      ParGridFunction    e_diff(&scalarfespace_);
      ParGridFunction    v_diff(&vectorfespace_);

      ParGridFunction    velocity  (&vectorfespace_);
      Vector vel_true(x_interpolated.GetData() + 2*size_qf + size_gf, size_gf_vec_true);
      velocity.SetFromTrueDofs(vel_true);

      ParGridFunction v_0(&vectorfespace_);
      Vector v_0_true(x_initial.GetData() + 2*size_qf + size_gf, size_gf_vec_true);
      v_0.SetFromTrueDofs(v_0_true);

      subtract( ind     , ind_0, ind_diff);
      subtract( rho     , rho_0, roh_diff);
      subtract( energy  , e_0  , e_diff);
      subtract( velocity, v_0  , v_diff);

      //-------------------------------------------------------------------

      real_t normindSq = 0.0;
      real_t normrohSq = 0.0;
      real_t normESq = 0.0;
      real_t normVSq = 0.0;
      real_t pDiffSq = 0.0;
      
      if(isL2_)
      {
         for (int e = 0; e < NE_; e++)
         {
            const int s_offset = offsetGP[e];

            IsoparametricTransformation Tr;
            mesh_->GetElementTransformation(e, pos_final, &Tr);

            const IntegrationRule &ir = qspace_.GetElementIntRule(e);
            const int nqp = ir.GetNPoints();

            for (int q = 0; q < nqp; q++)
            {
               const IntegrationPoint &ip = ir.IntPoint(q);
               Tr.SetIntPoint(&ip);
               real_t w = Tr.Weight() * ip.weight;

               normindSq += 0.5* w *ind_diff[s_offset+q] * ind_diff[s_offset+q];
               normrohSq += 0.5* w *roh_diff[s_offset+q] * roh_diff[s_offset+q];
            }
         }
      }
      else{
         normindSq = 0.5 * ind_diff.Norml2() * ind_diff.Norml2();
         normrohSq = 0.5 * roh_diff.Norml2() * roh_diff.Norml2();
      }

      MPI_Allreduce(MPI_IN_PLACE, &normindSq, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

      MPI_Allreduce(MPI_IN_PLACE, &normrohSq, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

      normESq = Integrate_e_minus_e0(pos_final, &ind, &energy, &e_0);

      normVSq = Integrate_v_minus_v0(pos_final, &ind, &velocity, &v_0);

      if(pressureOpt)
      {
         pDiffSq = IntegratePressureDiff(pos_final, &p_initial_, &rho, &energy);
      } 

      return w_1*normindSq + w_2*normrohSq + w_3* normESq + w_4* normVSq + w_p*pDiffSq;
   }

   void CalcObjectiveGrad(const Vector &x, Vector &grad) const  override
   {
      Vector x_interpolated(offset_[4]);  

      if(subproblem)
      {
         x_interpolated = x_initial;
         x_interpolated.SetSubVector(optProbInd,x);
      }
      else
      {
         x_interpolated = x;
      }

      QuadratureFunction ind(&qspace_, x_interpolated.GetData());
      QuadratureFunction rho(&qspace_, x_interpolated.GetData() + size_qf);
      ParGridFunction    energy  (&scalarfespace_, x_interpolated.GetData() + 2*size_qf);
     
      QuadratureFunction ind_diff(&qspace_);
      QuadratureFunction roh_diff(&qspace_);
      ParGridFunction    e_diff(&scalarfespace_);
      ParGridFunction    v_diff(&vectorfespace_);

      ParGridFunction    velocity  (&vectorfespace_);
      Vector vel_true(x_interpolated.GetData() + 2*size_qf + size_gf, size_gf_vec_true);
      velocity.SetFromTrueDofs(vel_true);

      ParGridFunction v_0(&vectorfespace_);
      Vector v_0_true(x_initial.GetData() + 2*size_qf + size_gf, size_gf_vec_true);
      v_0.SetFromTrueDofs(v_0_true);

      subtract( ind     , ind_0, ind_diff);
      subtract( rho     , rho_0, roh_diff);
      subtract( energy  , e_0  , e_diff);
      subtract( velocity, v_0  , v_diff);

      QuadratureFunction pGradRho(&qspace_); pGradRho = 0.0;

      BlockVector ind_rho_e_v_grad(offset_);
      ind_rho_e_v_grad = 0.0;

      //------------------------------------------------------------------------

      if(isL2_)
      {
         for (int e = 0; e < NE_; e++)
         {
            const int s_offset = offsetGP[e];

            IsoparametricTransformation Tr;
            mesh_->GetElementTransformation(e, pos_final, &Tr);

            const IntegrationRule &ir = qspace_.GetElementIntRule(e);
            const int nqp = ir.GetNPoints();

            for (int q = 0; q < nqp; q++)
            {
               const IntegrationPoint &ip = ir.IntPoint(q);
               Tr.SetIntPoint(&ip);
               real_t w = Tr.Weight() * ip.weight;

               ind_diff[s_offset+q] *= w;
               roh_diff[s_offset+q] *= w;

               if(pressureOpt)
               {
                  Vector p0_vals(nqp), rho_vals(nqp), e_vals(nqp);
                  rho.GetValues(e, rho_vals); 
                  p_initial_.GetValues(e, p0_vals);
                  energy.GetValues(Tr, ir, e_vals);

                  double pressureDiff = 0.4 *rho_vals(q) * e_vals(q) - p0_vals(q);

                  pGradRho[s_offset+q] = 0.4*w * pressureDiff * e_vals(q);
               }
            }
         }
      }

      //------------------------------------------------------------------------

      ParLinearForm dQdeta(&scalarfespace_);
      ParLinearForm dQdv(&vectorfespace_);
      ParGridFunction    e_grad(&scalarfespace_); e_grad = 0.0;
      ParGridFunction    p_e_grad(&scalarfespace_); p_e_grad = 0.0;
      Vector v_grad_true(size_gf_vec_true); v_grad_true = 0.0;
      GridFunctionCoefficient e_diff_coeff(&e_diff);
      VectorGridFunctionCoefficient v_diff_coeff(&v_diff);

      auto *lfi_1 = new DomainLFIntegrator(e_diff_coeff);
      //auto *lfi_2 = new	VectorDomainLFIntegrator (v_diff_coeff);
      auto *lfi_2 = new VDiffIntegrator (velocity, v_0, ind);

      dQdeta.AddDomainIntegrator(lfi_1);
      dQdeta.Assemble();
      dQdeta.ParallelAssemble(e_grad);

      dQdv.AddDomainIntegrator(lfi_2);
      dQdv.Assemble();
      dQdv.ParallelAssemble(v_grad_true);

      ind_diff *= w_1;
      roh_diff *= w_2;
      e_grad   *= w_3;
      v_grad_true   *= w_4;

      if(pressureOpt)
      {
         ParLinearForm pressureGradELF(&scalarfespace_);
         mfem::LinearFormIntegrator *lfi_1 =
             new mfem::PressureDiffGradEIntegrator( rho, p_initial_, energy);

         pressureGradELF.AddDomainIntegrator(lfi_1);
         pressureGradELF.Assemble();
         pressureGradELF.ParallelAssemble(p_e_grad);

         p_e_grad *= w_p;
         pGradRho *= w_p;

         e_grad += p_e_grad;
         roh_diff += pGradRho;
      }

      ind_rho_e_v_grad.GetBlock(0) = ind_diff;
      ind_rho_e_v_grad.GetBlock(1) = roh_diff;
      ind_rho_e_v_grad.GetBlock(2) = e_grad;
      ind_rho_e_v_grad.GetBlock(3) = v_grad_true;

      if(subproblem)
      {
         ind_rho_e_v_grad.GetSubVector(optProbInd,grad);
      }
      else
      {
         grad = ind_rho_e_v_grad;
      } 
   }

virtual void CalcObjectiveM(  std::vector<mfem::Vector> & diagMass, std::vector<HypreParMatrix *> & M_) const override
   {
      if(subproblem)
      {
         mfem_error("CalcObjectiveHessian not implemented for subproblem option");
      }

      diagMass.resize(2);
      M_.resize(1);

      QuadratureFunction ind_w(&qspace_); ind_w = 1.0;
      QuadratureFunction roh_w(&qspace_); roh_w = 1.0;
      ParGridFunction    e_diff(&scalarfespace_);

      diagMass[0].SetSize(ind_w.Size());
      diagMass[1].SetSize(ind_w.Size());
      //------------------------------------------------------------------------

      if(isL2_)
      {
         for (int e = 0; e < NE_; e++)
         {
            const int s_offset = offsetGP[e];

            IsoparametricTransformation Tr;
            mesh_->GetElementTransformation(e, pos_final, &Tr);

            const IntegrationRule &ir = qspace_.GetElementIntRule(e);
            const int nqp = ir.GetNPoints();

            for (int q = 0; q < nqp; q++)
            {
               const IntegrationPoint &ip = ir.IntPoint(q);
               Tr.SetIntPoint(&ip);
               real_t w = Tr.Weight() * ip.weight;

               ind_w[s_offset+q] *= w;
               roh_w[s_offset+q] *= w;
            }
         }
      }
      diagMass[0] = ind_w;
      diagMass[1] = roh_w;

      delete(mass_form);
      mass_form = new ParBilinearForm(&scalarfespace_);
      auto *blfi = new MassIntegrator();
      mass_form->AddDomainIntegrator(blfi);
      mass_form->Assemble();
      mass_form->Finalize();

      M_[0] = mass_form->ParallelAssemble();

      delete mass_form;
   }

   void CalcConstraintGrad(const int constNumber,
                           const Vector &x, Vector &grad) const override
   {
      Vector x_interpolated(offset_[4]);  

      if(subproblem)
      {
         x_interpolated = x_initial;
         x_interpolated.SetSubVector(optProbInd,x);
      }
      else
      {
         x_interpolated = x;
      }

      QuadratureFunction ind(&qspace_, x_interpolated.GetData());
      QuadratureFunction rho(&qspace_, x_interpolated.GetData() + size_qf);
      ParGridFunction    energy  (&scalarfespace_, x_interpolated.GetData() + 2*size_qf);
      
      ParGridFunction    vel  (&vectorfespace_);
      Vector vel_true(x_interpolated.GetData() + 2*size_qf + size_gf, size_gf_vec_true);
      vel.SetFromTrueDofs(vel_true);

      BlockVector ind_rho_e_v_grad(offset_);
      QuadratureFunction ind_grad(&qspace_); ind_grad= 0.0;
      QuadratureFunction rho_grad(&qspace_); rho_grad= 0.0;
      ParGridFunction    e_grad(&scalarfespace_);  e_grad= 0.0;
      Vector v_grad_true(size_gf_vec_true); v_grad_true = 0.0;

      grad = 0.0;

      IsoparametricTransformation Tr;

      for (int e = 0; e < NE_; e++)
      {
         mesh_->GetElementTransformation(e, pos_final, &Tr);

         const IntegrationRule &ir = qspace_.GetElementIntRule(e);
         const int nqp = ir.GetNPoints();

         for (int q = 0; q < nqp; q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            Tr.SetIntPoint(&ip);
            mfem::Vector vel_GP;

            double w = Tr.Weight() * ip.weight;
            double ind_GP = ind[offsetGP[e]+q];
            double rho_GP = rho[offsetGP[e]+q];
            double e_GP = energy.GetValue(Tr, ip);
            vel.GetVectorValue (Tr, ip, vel_GP);

            if( constNumber == 0)
            {
               ind_grad[offsetGP[e]+q] = w;
            }
            else if( constNumber == 1)
            {
               ind_grad[offsetGP[e]+q] = w * rho_GP;
               rho_grad[offsetGP[e]+q] = w * ind_GP;
            }
            else if( ( spatialDim == 2 && ( constNumber == 2 || constNumber == 3 )) ||
                     ( spatialDim == 3 && ( constNumber == 2 || constNumber == 3 || constNumber == 4 )))
            {
               int d = 0;
               if      (constNumber == 2) { d = 0; }
               else if (constNumber == 3) { d = 1; }
               else if (constNumber == 4) { d = 2; }

               ind_grad[offsetGP[e]+q] = w * rho_GP * vel_GP[d];
               rho_grad[offsetGP[e]+q] = w * ind_GP * vel_GP[d];
            }
            else if( ( spatialDim == 2 && constNumber == 4) ||
                      ( spatialDim == 3 && constNumber == 5))
            {
               double velSq = vel_GP * vel_GP;
 
               ind_grad[offsetGP[e]+q] = w * rho_GP * ( e_GP + 0.5 * velSq );
               rho_grad[offsetGP[e]+q] = w * ind_GP * ( e_GP + 0.5 * velSq );

            }
            else{mfem_error("Constraint index does not exist.");}
         }
      }

      if( ( spatialDim == 2 && ( constNumber == 2 || constNumber == 3 )) ||
          ( spatialDim == 3 && ( constNumber == 2 || constNumber == 3 || constNumber == 4 )))
      {

         int d = 0;
         if      (constNumber == 2) { d = 0; }
         else if (constNumber == 3) { d = 1; }
         else if (constNumber == 4) { d = 2; }

         ParLinearForm momentumGradVLF(&vectorfespace_);
         mfem::LinearFormIntegrator *lfi_vel =
             new mfem::RemhosHydroHiOpProblem::momentumGradVIntegrator(ind, rho, d);

         momentumGradVLF.AddDomainIntegrator(lfi_vel);
         momentumGradVLF.Assemble();
         momentumGradVLF.ParallelAssemble(v_grad_true);
      }
      else if( ( spatialDim == 2 && constNumber == 4) ||
               ( spatialDim == 3 && constNumber == 5))
      {
         ParLinearForm energyGradELF(&scalarfespace_);
         mfem::LinearFormIntegrator *lfi_e =
             new mfem::RemhosHydroHiOpProblem::totalEnergyGradEIntegrator(ind, rho);

         energyGradELF.AddDomainIntegrator(lfi_e);
         energyGradELF.Assemble();
         energyGradELF.ParallelAssemble(e_grad);
      
         //-----------------------------------------------------------------------

         ParLinearForm energyGradVLF(&vectorfespace_);
         mfem::LinearFormIntegrator *lfi_vel =
             new mfem::RemhosHydroHiOpProblem::totalEnergyGradVIntegrator(ind, rho, vel);

         energyGradVLF.AddDomainIntegrator(lfi_vel);
         energyGradVLF.Assemble();
         energyGradVLF.ParallelAssemble(v_grad_true);
      }

      ind_rho_e_v_grad.GetBlock(0) = ind_grad;
      ind_rho_e_v_grad.GetBlock(1) = rho_grad;
      ind_rho_e_v_grad.GetBlock(2) = e_grad;
      ind_rho_e_v_grad.GetBlock(3) = v_grad_true;

      if(subproblem)
      {
         ind_rho_e_v_grad.GetSubVector(optProbInd,grad);
      }
      else
      {
         grad = ind_rho_e_v_grad;
      } 
};

void CalcConstraint(const int constNumber,
                    const Vector &x, Vector &constVal) const override
{
      Vector x_interpolated(offset_[4]);  

      if(subproblem)
      {
         x_interpolated = x_initial;
         x_interpolated.SetSubVector(optProbInd,x);
      }
      else
      {
         x_interpolated = x;
      }

      QuadratureFunction ind(&qspace_, x_interpolated.GetData());
      QuadratureFunction rho(&qspace_, x_interpolated.GetData() + size_qf);
      ParGridFunction    energy  (&scalarfespace_, x_interpolated.GetData() + 2*size_qf);
     
      ParGridFunction    vel  (&vectorfespace_);
      Vector vel_true(x_interpolated.GetData() + 2*size_qf + size_gf, size_gf_vec_true);
      vel.SetFromTrueDofs(vel_true);

      if( constNumber == 0 )
      {
         double vol_s = Integrate(pos_final, &ind, nullptr, nullptr, nullptr);

         constVal[0] = vol_s - targetVol;

         //std::cout<<"Const 1: "<< constVal[0]<<std::endl;
      }
      else if( constNumber == 1)
      {
         double mass_s = Integrate(pos_final, &ind, &rho, nullptr, nullptr);

         constVal[1] = mass_s - targetMass;

         //std::cout<<"Const 2: "<< constVal[1]<<std::endl;
      }
      else if( ( spatialDim == 2 && ( constNumber == 2 || constNumber == 3 )) ||
               ( spatialDim == 3 && ( constNumber == 2 || constNumber == 3 || constNumber == 4 )))
      {
         int d = 0;

         if      (constNumber == 2) { d = 0; }
         else if (constNumber == 3) { d = 1; }
         else if (constNumber == 4) { d = 2; }

         double momentum_s = Integrate(pos_final,
                              &ind, &rho, nullptr, &vel, d);

         constVal[2+d] = momentum_s - targetMomentum[d];

         //std::cout<<"Const "<< 3+d<<": " << constVal[2+d]<<std::endl;
      }
      else if( ( spatialDim == 2 && constNumber == 4) ||
               ( spatialDim == 3 && constNumber == 5))
      {
         double tot_energy = Integrate(pos_final,
                                     &ind, &rho, &energy, &vel);

         constVal[2+spatialDim] = tot_energy - targetEnergy;
         //std::cout<<"Const "<< 3+spatialDim<<": " << constVal[2+spatialDim]<<std::endl;
      }

      else{mfem_error("Constraint index does not exist.");}
   };

private:

double Integrate(const Vector &pos,
                 const QuadratureFunction *ind,
                 const QuadratureFunction *rho,
                 const ParGridFunction *e,
                 const ParGridFunction *v, int comp = 0) const
{
   MFEM_VERIFY(ind || rho || e, "At least one function must be specified.");

   const QuadratureSpace *qspace = nullptr;
   if (ind) { qspace = dynamic_cast<const QuadratureSpace *>(ind->GetSpace()); }
   if (rho) { qspace = dynamic_cast<const QuadratureSpace *>(rho->GetSpace()); }

   auto mesh = (qspace) ? qspace->GetMesh() : e->ParFESpace()->GetMesh();
   const int NE = mesh->GetNE(), dim = mesh->Dimension();
   double integral = 0.0;
   for (int j = 0; j < NE; j++)
   {
      const IntegrationRule &ir =
         (qspace) ? qspace->GetElementIntRule(j)
                  : IntRules.Get(e->ParFESpace()->GetFE(j)->GetGeomType(), 7);
      const int nqp = ir.GetNPoints();

      // Transformation w.r.t. the given mesh positions.
      IsoparametricTransformation Tr;
      mesh->GetElementTransformation(j, pos, &Tr);

      Vector ind_vals(nqp), rho_vals(nqp), e_vals(nqp);
      DenseMatrix v_vals(dim, nqp);
      if (ind) { ind->GetValues(j, ind_vals); }
      else { ind_vals = 1.0; }
      if (rho) { rho->GetValues(j, rho_vals); }
      else { rho_vals = 1.0; }
      if (e) { e->GetValues(Tr, ir, e_vals); }
      else { e_vals = 1.0; }
      if (v) { v->GetVectorValues(Tr, ir, v_vals); }
      else { v_vals = 0.0; }

      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         real_t vv = 0.0;
         for (int d = 0; d < dim; d++) { vv += v_vals(d, q) * v_vals(d, q); }
         if (v != nullptr && e == nullptr)
         {
            // Momentum case.
            integral += Tr.Weight() * ip.weight *
                        ind_vals(q) * rho_vals(q) * v_vals(comp, q);
         }
         else
         {
            // Volume / mass / internal energy / total energy cases.
            integral += Tr.Weight() * ip.weight *
                        (ind_vals(q) * rho_vals(q) * e_vals(q) +
                         0.5 * ind_vals(q) * rho_vals(q) * vv);
         }
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &integral, 1, MPI_DOUBLE, MPI_SUM,
                 MPI_COMM_WORLD);
   return integral;
}

double Integrate_v_minus_v0(const Vector &pos,
                            const QuadratureFunction *ind,
                            const ParGridFunction *v,
                            const ParGridFunction *v_0) const
{
   MFEM_VERIFY(ind, "At least one function must be specified.");

   const QuadratureSpace *qspace = nullptr;
   if (ind) { qspace = dynamic_cast<const QuadratureSpace *>(ind->GetSpace()); }

   auto mesh = qspace->GetMesh();
   const int NE = mesh->GetNE(), dim = mesh->Dimension();
   double integral = 0.0;
   for (int j = 0; j < NE; j++)
   {
      const IntegrationRule &ir = qspace->GetElementIntRule(j);
      const int nqp = ir.GetNPoints();

      // Transformation w.r.t. the given mesh positions.
      IsoparametricTransformation Tr;
      mesh->GetElementTransformation(j, pos, &Tr);

      Vector ind_vals(nqp);
      DenseMatrix v_vals(dim, nqp);
      DenseMatrix v_vals_0(dim, nqp);
      
      v->GetVectorValues(Tr, ir, v_vals);
      v_0->GetVectorValues(Tr, ir, v_vals_0);

      v_vals -=v_vals_0;

      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         real_t vv = 0.0;
         for (int d = 0; d < dim; d++) { vv += v_vals(d, q) * v_vals(d, q); }

         // Volume / mass / internal energy / total energy cases.
         integral += Tr.Weight() * ip.weight * 0.5 * vv;

      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &integral, 1, MPI_DOUBLE, MPI_SUM,
                 MPI_COMM_WORLD);
   return integral;
}

double Integrate_e_minus_e0(const Vector &pos,
                            const QuadratureFunction *ind,
                            const ParGridFunction *e,
                            const ParGridFunction *e_0) const
{
   MFEM_VERIFY(ind, "At least one function must be specified.");

   const QuadratureSpace *qspace = nullptr;
   if (ind) { qspace = dynamic_cast<const QuadratureSpace *>(ind->GetSpace()); }

   auto mesh = qspace->GetMesh();
   const int NE = mesh->GetNE(), dim = mesh->Dimension();
   double integral = 0.0;
   for (int j = 0; j < NE; j++)
   {
      const IntegrationRule &ir = qspace->GetElementIntRule(j);
      const int nqp = ir.GetNPoints();

      // Transformation w.r.t. the given mesh positions.
      IsoparametricTransformation Tr;
      mesh->GetElementTransformation(j, pos, &Tr);

      Vector ind_vals(nqp);
      Vector e_vals(nqp);
      Vector e_vals_0(nqp);
     
      e->GetValues(Tr, ir, e_vals);
      e_0->GetValues(Tr, ir, e_vals_0);
      e_vals -=e_vals_0;

      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         real_t ee = e_vals[q] * e_vals[q];

         // Volume / mass / internal energy / total energy cases.
         integral += Tr.Weight() * ip.weight * 0.5 * ee;

      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &integral, 1, MPI_DOUBLE, MPI_SUM,
                 MPI_COMM_WORLD);
   return integral;
}

double IntegratePressureDiff(const Vector &pos,
                           const QuadratureFunction *p0_,
                           const QuadratureFunction *rho_,
                           const ParGridFunction *e_) const
{
   MFEM_VERIFY(rho_ && p0_ && e_, "All function must be specified.");

   const QuadratureSpace *qspace = nullptr;
   if (rho_) { qspace = dynamic_cast<const QuadratureSpace *>(rho_->GetSpace()); }

   auto mesh = qspace->GetMesh();
   const int NE = mesh->GetNE();
   double integral = 0.0;
   for (int e = 0; e < NE; e++)
   {
      const IntegrationRule &ir = qspace->GetElementIntRule(e);
      const int nqp = ir.GetNPoints();

      // Transformation w.r.t. the given mesh positions.
      IsoparametricTransformation Tr;
      mesh->GetElementTransformation(e, pos, &Tr);

      Vector p0_vals(nqp), rho_vals(nqp), e_vals(nqp);
      p0_->GetValues(e, p0_vals); 
      rho_->GetValues(e, rho_vals);
      e_->GetValues(Tr, ir, e_vals);

      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         double pressureDiff = 0.4 * rho_vals(q) * e_vals(q) - p0_vals(q);
         integral += 0.5 * Tr.Weight() * ip.weight * pressureDiff * pressureDiff;
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &integral, 1, MPI_DOUBLE, MPI_SUM,
                 MPI_COMM_WORLD);
   return integral;
}
};


} // namespace mfem

#endif // MFEM_REMHOS_GHiOp
