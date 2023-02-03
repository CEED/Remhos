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

#include "remhos_gslib.hpp"

using namespace std;

namespace mfem
{

void InterpolationRemap::Remap(const ParGridFunction &source,
                               const ParGridFunction &x_new,
                               ParGridFunction &interpolated)
{
   ParMesh &pmesh_src = *source.ParFESpace()->GetParMesh();
   ParFiniteElementSpace &pfes_tgt = *interpolated.ParFESpace();

   const int dim = pmesh_src.Dimension();
   MFEM_VERIFY(dim > 1, "Interpolation remap works only in 2D and 3D.");

   const int NE = pmesh_src.GetNE();
   const int nsp = interpolated.ParFESpace()->GetFE(0)->GetNodes().GetNPoints();

   // Generate list of points where the grid function will be evaluated.
   Vector vxyz;
   vxyz.SetSize(nsp * NE * dim);

   for (int e = 0; e < NE; e++)
   {
      const IntegrationRule &ir = pfes_tgt.GetFE(e)->GetNodes();

      // Transformation of the element with the new coordinates.
      IsoparametricTransformation Tr;
      pmesh_src.GetElementTransformation(e, x_new, &Tr);

      // Node positions of the interpolated f-n (new element coordinates).
      DenseMatrix pos_target_nodes;
      Tr.Transform(ir, pos_target_nodes);
      Vector rowx(vxyz.GetData() + e*nsp, nsp),
             rowy(vxyz.GetData() + e*nsp + NE*nsp, nsp), rowz;
      if (dim == 3)
      {
         rowz.SetDataAndSize(vxyz.GetData() + e*nsp + 2*NE*nsp, nsp);
      }
      pos_target_nodes.GetRow(0, rowx);
      pos_target_nodes.GetRow(1, rowy);
      if (dim == 3) { pos_target_nodes.GetRow(2, rowz); }
   }

   const int nodes_cnt = vxyz.Size() / dim;

   // Evaluate source grid function.
   Vector interp_vals(nodes_cnt);
   FindPointsGSLIB finder(pfes_tgt.GetComm());
   finder.Setup(pmesh_src);
   finder.Interpolate(vxyz, source, interp_vals);

   interpolated = interp_vals;
}

} // namespace mfem
