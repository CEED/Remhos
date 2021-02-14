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

enum Estimator: int { ZZ = 0, L2ZZ, JJt, Custom, DRL4AMR };

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
   double derefs_max;

   // JJt
   Mesh quad_JJt;
   const int lorder_JJt = 1; // LOR space order

public:
   Operator(ParFiniteElementSpace&, ParMesh&, ParGridFunction&, const Options&);

   ~Operator();

   void Reset();
   void Apply(Array<Refinement> = {});
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
               const double mass0_u,
               ParGridFunction &inflow_gf,
               FunctionCoefficient &inflow);

private:
   void ApplyZZ();
   void ApplyJJt();
   void ApplyCustom();
   void ApplyDRL4AMR(Array<Refinement>&);
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

} // namespace amr

}  // namespace mfem

#endif  // MFEM_REMHOS_AMR
