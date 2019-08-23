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

#ifndef MFEM_REMHOS_FCT
#define MFEM_REMHOS_FCT

#include "mfem.hpp"

namespace mfem
{

// Monotone, High-order, Conservative Solver.
class MHCSolver
{
protected:
   ParFiniteElementSpace &pfes;
   double dt;

public:
   MHCSolver(ParFiniteElementSpace &space, double dt_) : pfes(space), dt(dt_) {}

   virtual void UpdateTimeStep(double dt_) { dt = dt_; }

   // Calculate du, so that we have
   // bounds preservation: u_min_i <= u_i + dt du_i <= u_max_i,
   // conservation:        sum m_i (u_i + dt du_ho_i) = sum m_i (u_i + dt du_i).
   // Some methods utilize du_lo as a backup choice, as it satisfies the above.
   virtual void CalcMHCSolution(const Vector &u, const Vector &m,
                                const Vector &du_ho, const Vector &du_lo,
                                const Vector &u_min, const Vector &u_max,
                                Vector &du) const = 0;
};

class ClipScaleSolver : public MHCSolver
{
public:
   virtual void CalcMHCSolution(const Vector &u, const Vector &m,
                                const Vector &du_ho, const Vector &du_lo,
                                const Vector &u_min, const Vector &u_max,
                                Vector &du) const;
};

} // namespace mfem

#endif // MFEM_LAGHOS_FCT
