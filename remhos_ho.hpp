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

struct TimingData;

// High-Order Solver.
// Conserve mass / provide high-order convergence / may violate the bounds.
class HOSolver
{
protected:
   ParFiniteElementSpace &pfes;

public:
   HOSolver(ParFiniteElementSpace &space) : pfes(space) { }

   virtual ~HOSolver() { }

   virtual void CalcHOSolution(const Vector &u, Vector &du) const = 0;

   TimingData *timer = nullptr;
};

class CGHOSolver : public HOSolver
{
protected:
   ParBilinearForm &M, &K;

public:
   CGHOSolver(ParFiniteElementSpace &space,
              ParBilinearForm &Mbf, ParBilinearForm &Kbf);

   virtual void CalcHOSolution(const Vector &u, Vector &du) const;
};

class LocalInverseHOSolver : public HOSolver
{
protected:
   ParBilinearForm &M, &K;
   mutable DGMassInverse *M_inv = nullptr;

public:
   LocalInverseHOSolver(ParFiniteElementSpace &space,
                        ParBilinearForm &Mbf, ParBilinearForm &Kbf);

   void CalcHOSolution(const Vector &u, Vector &du) const override;
   ~LocalInverseHOSolver() { delete M_inv; }
};

class Assembly;

class NeumannHOSolver : public HOSolver
{
protected:
   const ParBilinearForm &M, &K;
   const Vector &M_lumped;
   Assembly &assembly;

public:
   NeumannHOSolver(ParFiniteElementSpace &space,
                   ParBilinearForm &Mbf, ParBilinearForm &Kbf, Vector &Mlump,
                   Assembly &a);

   virtual void CalcHOSolution(const Vector &u, Vector &du) const;
};

} // namespace mfem

#endif // MFEM_REMHOS_HO
