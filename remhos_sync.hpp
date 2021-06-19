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

#ifndef MFEM_REMHOS_SYNC
#define MFEM_REMHOS_SYNC

#define EMPTY_ZONE_TOL 1e-12

#include "mfem.hpp"

namespace mfem
{

void ComputeBoolIndicators(int NE, const Vector &u,
                           Array<bool> &ind_elem, Array<bool> &ind_dofs);

void ComputeRatio(int NE, const Vector &u_s, const Vector &u,
                  Vector &s, Array<bool> &bool_el, Array<bool> &bool_dof);

void ZeroOutEmptyDofs(const Array<bool> &ind_elem,
                      const Array<bool> &ind_dofs, Vector &u);

// Set of functions that are used for debug calls.
void ComputeMinMaxS(int NE, const Vector &us, const Vector &u,
                    double &s_min_glob, double &s_max_glob);
void ComputeMinMaxS(const Vector &s, const Array<bool> &bool_dofs, int myid);
void PrintCellValues(int cell_id, int NE, const Vector &vec, const char *msg);

// Checks if us_lo / s_lo is in the full stencil bounds.
// Full stencil here means the whole neighborhood of active dofs.
// Although it is the backbone theorem we use for the LO product solutions
// (ALE hydro paper 2018), in the case of local bounds it does not hold, because
// the LO discrete upwind procedure always uses the full stencil, without any
// notion of active dofs.
void VerifyLOProduct(int NE, const Vector &us_LO, const Vector &u_LO,
                     const Vector &s_min, const Vector &s_max,
                     const Array<bool> &active_el,
                     const Array<bool> &active_dofs);

class BoolFunctionCoefficient : public FunctionCoefficient
{
protected:
   const Array<bool> &ind;

public:
   BoolFunctionCoefficient(double (*f)(const Vector &), const Array<bool> &bi)
      : FunctionCoefficient(f), ind(bi) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

} // namespace mfem

#endif // MFEM_REMHOS_SYNC
