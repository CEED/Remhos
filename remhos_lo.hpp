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

#ifndef MFEM_REMHOS_LO
#define MFEM_REMHOS_LO

#include "mfem.hpp"

namespace mfem
{

// Low-Order Solver.
class LOSolver
{
protected:
   ParFiniteElementSpace &pfes;

public:
   LOSolver(ParFiniteElementSpace &space) : pfes(space) { }

   virtual ~LOSolver() { }

   virtual void CalcLOSolution(const Vector &u, Vector &du) const = 0;
};

class Assembly;

class DiscreteUpwind : public LOSolver
{
protected:
   const SparseMatrix &K;
   mutable SparseMatrix D;
   const Array<int> &K_smap;
   const bool update_D;
   Assembly &assembly;
   const Vector &M_lumped;

   void ComputeDiscreteUpwindMatrix() const;

public:
   DiscreteUpwind(ParFiniteElementSpace &space, const SparseMatrix &adv,
                  const Array<int> &adv_smap, const Vector &Mlump,
                  Assembly &asmbly, bool updateD);

   virtual void CalcLOSolution(const Vector &u, Vector &du) const;
};

class ResidualDistribution : public LOSolver
{
protected:
   ParBilinearForm &K;
   Assembly &assembly;
   const Vector &M_lumped;
   const bool subcell_scheme;
   const bool time_dep;

public:
   ResidualDistribution(ParFiniteElementSpace &space, ParBilinearForm &Kbf,
                        Assembly &asmbly, const Vector &Mlump,
                        bool subcell, bool timedep);

   virtual void CalcLOSolution(const Vector &u, Vector &du) const;
};

} // namespace mfem

#endif // MFEM_REMHOS_LO
