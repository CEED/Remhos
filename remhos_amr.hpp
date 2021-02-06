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

enum Estimator: int { ZZ = 0, L2ZZ, JJt, Custom };

struct Options
{
   const int estimator;
   const int order, mesh_order;
   const int max_level, nc_limit;
   const double ref_threshold, deref_threshold;
};

class EstimatorIntegrator;

class Operator
{
   ParFiniteElementSpace &pfes;
   ParMesh &pmesh;
   ParGridFunction &u;
   const int myid, dim, sdim;
   const Options &opt;

   FiniteElementCollection *h1_fec = nullptr;
   FiniteElementCollection *l2_fec = nullptr;
   FiniteElementCollection *flux_fec = nullptr;
   ParFiniteElementSpace *flux_fes = nullptr;

   RT_FECollection *smooth_flux_fec = nullptr;
   ErrorEstimator *estimator = nullptr;
   ThresholdRefiner *refiner = nullptr;
   ThresholdDerefiner *derefiner = nullptr;
   EstimatorIntegrator *integ = nullptr;

   socketstream amr_vis[4];
   const char *host = "localhost";
   const int port = 19916;
   const int Wx = 400, Wy = 400;
   const int Ww = 640, Wh = 640;
   const char *keys = "gAmRj";

   bool mesh_refined = false;
   bool mesh_derefined = false;
   Array<Refinement> refs;
   Vector derefs;

public:
   Operator(ParFiniteElementSpace&, ParMesh&, ParGridFunction&, const Options&);

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
   void ApplyZZ();
   void ApplyJJt();
   void ApplyCustom();
   void UpdateAndRebalance(BlockVector &S,
                           Array<int> &offset,
                           LowOrderMethod &lom,
                           ParMesh *subcell_mesh,
                           ParFiniteElementSpace *pfes_sub,
                           ParGridFunction *xsub,
                           ParGridFunction &v_sub_gf);
};

class EstimatorIntegrator: public DiffusionIntegrator
{
   enum class FluxMode { diffusion, one, two };

private:
   int NE, e2;
   ParMesh &pmesh;
   const FluxMode mode;
   const Options &opt;
   ConstantCoefficient one {1.0};

public:
   EstimatorIntegrator(ParMesh&, const Options&,
                       const FluxMode = FluxMode::diffusion);

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

/// DGMass
struct DGMass : mfem::Operator
{
   mutable Array<double> M, Minv;
   mutable Array<int> ipiv;
   Array<int> vec_offsets, M_offsets;
   int nel, vdim;

   DGMass(ParFiniteElementSpace &fes);
   virtual void Mult(const Vector &u, Vector &Mu) const override;
   void Solve(Vector &b) const;
   void Solve(const Vector &Mu, Vector &u) const;
};

/// Mass
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

/// AMR_P
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

/// LORMixedMass
struct LORMixedMass : mfem::Operator
{
   ParFiniteElementSpace &fes_ho, &fes_lor;
   Table ho2lor;
   mutable Array<double> M_mixed;
   Array<int> offsets;

   LORMixedMass(ParFiniteElementSpace &fes_ho_,
                ParFiniteElementSpace &fes_lor_);
   virtual void Mult(const Vector &x, Vector &y) const override;
   virtual void MultTranspose(const Vector &x, Vector &y) const override;
};

/// TransferR
struct TransferR : mfem::Operator
{
   LORMixedMass mass_mixed;
   DGMass mass_lor;
   mutable Vector Minvx;

   TransferR(ParFiniteElementSpace &fes_ho,
             ParFiniteElementSpace &fes_lor);
   virtual void Mult(const Vector &x, Vector &y) const override;
   virtual void MultTranspose(const Vector &x, Vector &y) const override;
};

/// PtRtMRPOperator
struct PtRtMRPOperator : mfem::Operator
{
   const mfem::Operator &P, &R, &M;
   mutable Vector z1, z2, z3;
   PtRtMRPOperator(const mfem::Operator &P_,
                   const mfem::Operator &R_,
                   const mfem::Operator &M_);
   virtual void Mult(const Vector &x, Vector &y) const override;
};

/// TransferP
struct TransferP : mfem::Operator
{
   //TransferR R;
   const mfem::Operator &R;
   const mfem::Operator &P;
   DGMass &mass_lor;
   PtRtMRPOperator PtRtMRP;
   HypreParMatrix *M_ho;
   std::unique_ptr<Solver> prec;
   CGSolver cg;
   Array<int> empty;

   mutable Vector Y, Mx, RtMx, PtRtMx;

   TransferP(ParFiniteElementSpace &fes_ho,
             const mfem::Operator *R,
             //Mass mass_lor,
             DGMass &dg_mass_lor,
             ParFiniteElementSpace &fes_lor);
   ~TransferP();
   virtual void Mult(const Vector &x, Vector &y) const override;
};

/// L2Projection
struct L2Projection
{
   ParFiniteElementSpace &fes;
   ParBilinearForm m;
   ParLinearForm b;
   OperatorHandle M;
   std::unique_ptr<Solver> prec;
   CGSolver cg;
   Array<int> empty;
   L2Projection(ParFiniteElementSpace &fes_);
   void Project(Coefficient &coeff, ParGridFunction &u);
};

} // namespace amr

}  // namespace mfem

#endif  // MFEM_REMHOS_AMR
