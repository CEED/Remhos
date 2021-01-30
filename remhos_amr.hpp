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

enum estimator: int { custom = 0, zz, kelly};

class EstimatorIntegrator: public DiffusionIntegrator
{
   enum class mode { diffusion, one, two };

   int NE, e;
   ParMesh &pmesh;
   const mode flux_mode;
   const int max_level;
   const double jac_threshold;
   ConstantCoefficient one {1.0};

public:
   EstimatorIntegrator(ParMesh &pmesh,
                       const int max_level,
                       const double jac_threshold,
                       const mode flux_mode = mode::two);

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
   ParGridFunction &x, &xsub, &sol;
   ParFiniteElementSpace &pfes, &mesh_pfes;

   const int myid, dim, sdim;
   const int order = 3; // should be computed

   L2_FECollection flux_fec;
   ParFiniteElementSpace flux_fes;

   RT_FECollection *smooth_flux_fec = nullptr;
   ErrorEstimator *estimator = nullptr;
   ThresholdRefiner *refiner = nullptr;
   ThresholdDerefiner *derefiner = nullptr;
   amr::EstimatorIntegrator *integ = nullptr;

   const struct Options
   {
      int estimator;
      double ref_threshold;
      double jac_threshold;
      double deref_threshold;
      int max_level;
      int nc_limit;
   } opt;

public:
   Operator(ParFiniteElementSpace &pfes,
            ParFiniteElementSpace &mesh_pfes,
            ParMesh &pmesh,
            ParGridFunction &x,
            ParGridFunction &xsub,
            ParGridFunction &sol,
            int estimator,
            double ref_t, double jac_t, double deref_t,
            int max_level, int nc_limit);

   ~Operator();

   void Reset();

   void Update(AdvectionOperator&,
               ODESolver *ode_solver,
               BlockVector &S,
               Array<int> &offset,
               ParGridFunction &u);

private:
   void AMRUpdate(BlockVector &S,
                  Array<int> &offset,
                  ParGridFunction &u);
};

} // namespace amr

}  // namespace mfem

#endif  // MFEM_REMHOS_AMR
