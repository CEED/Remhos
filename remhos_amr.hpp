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

#ifndef MFEM_REMHOS_AMR
#define MFEM_REMHOS_AMR

#include "remhos.hpp"

namespace mfem
{

namespace amr
{

enum estimator: int { custom = 0, jjt, zz, l2zz, kelly};

class EstimatorIntegrator: public DiffusionIntegrator
{
public:
   enum class mode { diffusion, one, two };

private:
   int NE, e2;
   ParMesh &pmesh;
   const mode flux_mode;
   const int max_level;
   const double jjt_threshold;
   ConstantCoefficient one {1.0};

public:
   EstimatorIntegrator(ParMesh &pmesh,
                       const int max_level,
                       const double jjt_threshold,
                       const mode flux_mode = mode::diffusion);

   void Reset();

   double ComputeFluxEnergy(const FiniteElement &fluxelem,
                            ElementTransformation &Trans,
                            Vector &flux, Vector *d_energy = NULL);

   virtual void ComputeElementFlux(const FiniteElement &el,
                                   ElementTransformation &Trans,
                                   Vector &u,
                                   const FiniteElement &fluxelem,
                                   Vector &flux,
                                   bool with_coef = false);
private:
   void ComputeElementFlux1(const FiniteElement &el,
                            ElementTransformation &Trans,
                            const Vector &u,
                            const FiniteElement &fluxelem,
                            Vector &flux);

   void ComputeElementFlux2(const int e,
                            const FiniteElement &el,
                            ElementTransformation &Trans,
                            const FiniteElement &fluxelem,
                            Vector &flux);
};

// AMR operator
class Operator
{
   ParMesh &pmesh;
   ParGridFunction &u;
   ParFiniteElementSpace &pfes;

   const int myid, dim, sdim;

   H1_FECollection fec;
   L2_FECollection flux_fec;
   ParFiniteElementSpace flux_fes;

   RT_FECollection *smooth_flux_fec = nullptr;
   ErrorEstimator *estimator = nullptr;
   ThresholdRefiner *refiner = nullptr;
   ThresholdDerefiner *derefiner = nullptr;
   amr::EstimatorIntegrator *integ = nullptr;

   socketstream amr_vis[4];
   const char *host = "localhost";
   const int port = 19916;
   int Wx = 400, Wy = 400;
   const int Ww = 640, Wh = 640;
   const char *keys = "gAmRj";

   const struct Options
   {
      int order, mesh_order;
      int estimator;
      double ref_threshold;
      double deref_threshold;
      double jjt_ref_threshold;
      double jjt_deref_threshold;
      int max_level;
      int nc_limit;
   } opt;

   bool mesh_refined;
   bool mesh_derefined;

   Array<Refinement> refs;
   Vector derefs;

public:
   Operator(ParFiniteElementSpace &pfes,
            ParFiniteElementSpace &mesh_pfes,
            ParMesh &pmesh,
            ParGridFunction &sol,
            int order, int mesh_order,
            int estimator,
            double ref_t,
            double deref_t,
            double jac_ref_t,
            double jac_deref_t,
            int max_level,
            int nc_limit);

   ~Operator();

   void Reset();

   void Apply();

   bool Refined();
   bool DeRefined();

   void Update(AdvectionOperator&,
               ODESolver *ode_solver,
               BlockVector &S,
               Array<int> &offset,
               LowOrderMethod &lom,
               ParMesh *subcell_mesh,
               ParFiniteElementSpace *pfes_sub,
               ParGridFunction *xsub,
               ParGridFunction &v_sub_gf,
               Vector &lumpedM,
               const double mass0_u,
               ParGridFunction &inflow_gf,
               FunctionCoefficient &inflow);

private:
   void AMRUpdateEstimatorCustom();
   void AMRUpdateEstimatorJJt();
   void AMRUpdateEstimatorZZKelly();

   void AMRUpdate(BlockVector &S,
                  Array<int> &offset,
                  LowOrderMethod &lom,
                  ParMesh *subcell_mesh,
                  ParFiniteElementSpace *pfes_sub,
                  ParGridFunction *xsub,
                  ParGridFunction &v_sub_gf);
};

struct Mass : mfem::Operator
{
   //ParFiniteElementSpace fes;
   const int vsize;
   const SparseMatrix *R_ptr;
   DenseMatrix *R;
   ParBilinearForm m;
   OperatorHandle M;
   //std::unique_ptr<Solver> prec;
   CGSolver cg;
   Array<int> empty;

   mutable Vector z1, z2;

   Mass(ParFiniteElementSpace &fes_, bool pa=false);

   virtual void Mult(const Vector &u, Vector &Mu) const override;

   void Solve(const Vector &Mu, Vector &u) const;
};

struct AMR_P : mfem::Operator
{
   Mass &M_refine, &M_coarse;
   const mfem::Operator &P;
   RAPOperator rap;
   CGSolver cg;

   mutable Vector z1, z2;

   AMR_P(Mass &M_refine, Mass &M_coarse,
         const mfem::Operator &P);

   void Mult(const Vector &x, Vector &y) const override;
};

} // namespace amr

}  // namespace mfem

#endif  // MFEM_REMHOS_AMR
