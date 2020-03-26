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

#ifndef MFEM_REMHOS_MONO
#define MFEM_REMHOS_MONO

#include "mfem.hpp"

namespace mfem
{

// Monolithic solvers - these solve the transport/remap problem directly,
// without splitting into HO / LO / FCT phases.
// The result should be a high-order, conservative, bound preserving solution.
class MonolithicSolver
{
protected:
   ParFiniteElementSpace &pfes;

public:
   MonolithicSolver(ParFiniteElementSpace &space) : pfes(space) { }

   virtual ~MonolithicSolver() { }

   virtual void CalcSolution(const Vector &u, Vector &du) const = 0;
};

class Assembly;
class SmoothnessIndicator;

class MonoRDSolver : public MonolithicSolver
{
protected:
   const SparseMatrix &K_mat, &M_mat;
   const Vector &M_lumped;
   Assembly &assembly;
   SmoothnessIndicator *smth_indicator;
   Vector scale;
   bool subcell_scheme;
   const bool time_dep;
   const bool mass_lim;

public:
   MonoRDSolver(ParFiniteElementSpace &space,
                const SparseMatrix &adv_mat, const SparseMatrix &mass_mat,
                const Vector &Mlump,
                Assembly &asmbly, SmoothnessIndicator *si,
                VectorFunctionCoefficient &velocity,
                bool subcell, bool timedep, bool masslim);

   void CalcSolution(const Vector &u, Vector &du) const;
};

} // namespace mfem

#endif // MFEM_REMHOS_MONO
