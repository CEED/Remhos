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

#ifndef MFEM_REMHOS_GSLIB
#define MFEM_REMHOS_GSLIB

#include "mfem.hpp"

namespace mfem
{

class InterpolationRemap
{
private:
   // For now there is only one Mesh. Its nodes are at the initial positions.
   // This class will not change the node positions of this Mesh.
   ParMesh &pmesh_init;
   // Don't touch this (direct access to the mesh positions). Used only for vis.
   Vector *x;
   // Initial mesh node positions.
   const Vector pos_init;

   // Positions of the DOFs of pfes, for the given mesh positions.
   void GetDOFPositions(const ParFiniteElementSpace &pfes,
                              const Vector &pos_mesh,
                              Vector &pos_dofs);

   // Mass of g for the given mesh positions.
   double Mass(const Vector &pos, const ParGridFunction &g);

   // Computes bounds for the DOFs of pfes_final, at the mesh positions given
   // by pos_final. The bounds are determined by the values of g_init, which
   // is defined with respect on the initial mesh.
   void CalcDOFBounds(const ParGridFunction &g_init,
                      const ParFiniteElementSpace &pfes_final,
                      const Vector &pos_final,
                      Vector &g_min, Vector &g_max);

public:
   InterpolationRemap(ParMesh &m)
       : pmesh_init(m), x(pmesh_init.GetNodes()), pos_init(*x) { }

   void Remap(const ParGridFunction &u_initial,
              const ParGridFunction &pos_final, ParGridFunction &u_final);

   bool vis_bounds = true;
};

} // namespace mfem

#endif // MFEM_REMHOS_GSLIB
