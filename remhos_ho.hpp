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

#ifndef MFEM_REMHOS_HO
#define MFEM_REMHOS_HO

#include "mfem.hpp"

namespace mfem
{

// High-Order Solver.
class HOSolver
{
protected:
   ParFiniteElementSpace &pfes;

public:
   HOSolver(ParFiniteElementSpace &space) : pfes(space) {}

   // du_dt = M^{-1} (K u + b).
   virtual void CalcHOSolution(const Vector &u, Vector &du) const = 0;
};

class Assembly;

class NeumannSolver : public HOSolver
{
protected:
   const ParBilinearForm &M, &K;
   const Vector &M_lumped;
   Assembly &assembly;

public:
   NeumannSolver(ParFiniteElementSpace &space,
                 ParBilinearForm &M_, ParBilinearForm &K_, Vector &Mlump,
                 Assembly &a);

   virtual void CalcHOSolution(const Vector &u, Vector &du) const;
};

} // namespace mfem

#endif // MFEM_REMHOS_HO
