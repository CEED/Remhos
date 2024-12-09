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
#include "remhos_tools.hpp"

using namespace std;

namespace mfem
{

void InterpolationRemap::Remap(const ParGridFunction &u_initial,
                               const ParGridFunction &pos_final,
                               ParGridFunction &u_final)
{
   const int dim = pmesh_init.Dimension();
   MFEM_VERIFY(dim > 1, "Interpolation remap works only in 2D and 3D.");

   ParFiniteElementSpace &pfes_final = *u_final.ParFESpace();

   // Generate list of points where u_initial will be interpolated.
   Vector pos_dof_final;
   GetDOFPositions(pfes_final, pos_final, pos_dof_final);

   // Interpolate u_initial.
   const int nodes_cnt = pos_dof_final.Size() / dim;
   Vector interp_vals(nodes_cnt);
   FindPointsGSLIB finder(pmesh_init.GetComm());
   finder.Setup(pmesh_init);
   finder.Interpolate(pos_dof_final, u_initial, interp_vals);

   // This assumes L2 ordering of the DOFs (as the ordering of the quad points).
   ParGridFunction u_interpolated(&pfes_final);
   u_interpolated = interp_vals;

   // Report masses.
   const double mass_s = Mass(*pmesh_init.GetNodes(), u_initial),
                mass_t = Mass(pos_final, u_interpolated);
   if (pmesh_init.GetMyRank() == 0)
   {
      std::cout << "Mass initial: " << mass_s << std::endl
                << "Mass final  : " << mass_t << std::endl
                << "Mass diff  : " << fabs(mass_s - mass_t) << endl
                << "Mass diff %: " << fabs(mass_s - mass_t)/mass_s*100 << endl;
   }

   // Compute min / max bounds.
   Vector u_final_min, u_final_max;
   CalcDOFBounds(u_initial, pfes_final, pos_final, u_final_min, u_final_max);
   if (vis_bounds)
   {
      ParGridFunction gf_min(u_initial), gf_max(u_initial);
      gf_min = u_final_min, gf_max = u_final_max;

      socketstream vis_min, vis_max;
      char vishost[] = "localhost";
      int  visport   = 19916;
      vis_min.precision(8);
      vis_max.precision(8);

      *x = pos_final;
      VisualizeField(vis_min, vishost, visport, gf_min, "u min",
                     0, 500, 300, 300);
      VisualizeField(vis_max, vishost, visport, gf_max, "u max",
                     300, 500, 300, 300);
      *x = pos_init;
   }

   // Do some optimization here to fix the masses, using the min/max bounds,
   // staying as close as possible to u_interpolated.
   u_final = u_interpolated;
}

void InterpolationRemap::Remap(const QuadratureFunction &u_initial,
                               const ParGridFunction &pos_final,
                               QuadratureFunction &u_final)
{
   const int dim = pmesh_init.Dimension();
   MFEM_VERIFY(dim > 1, "Interpolation remap works only in 2D and 3D.");

   auto qspace_final = dynamic_cast<QuadratureSpace *>(u_final.GetSpace());
   MFEM_VERIFY(qspace_final, "Broken QuadratureSpace.");

   // Generate list of points where u_initial will be interpolated.
   Vector pos_quad_final;
   GetQuadPositions(*qspace_final, pos_final, pos_quad_final);

   // Generate a GridFunction for interpolation.
   const int order = u_initial.GetIntRule(0).GetOrder() / 2;
   const int ref_factor = order + 1;
   ParMesh pmesh_lor = ParMesh::MakeRefined(pmesh_init, ref_factor,
                                            BasisType::ClosedGL);
   L2_FECollection fec(0, dim);
   ParFiniteElementSpace fes(&pmesh_lor, &fec);
   ParGridFunction u_initial_gf(&fes);
   MFEM_VERIFY(u_initial.Size() == u_initial_gf.Size(), "Size mismatch");
   u_initial_gf = u_initial;
   {
      socketstream vis_u;
      char vishost[] = "localhost";
      int  visport   = 19916;
      vis_u.precision(8);
      VisualizeField(vis_u, vishost, visport, u_initial_gf,
                     "u as a LOR GridFunction", 800, 0, 400, 400);
   }

   // Interpolate u_initial.
   const int quads_cnt = pos_quad_final.Size() / dim;
   Vector interp_vals(quads_cnt);
   FindPointsGSLIB finder(pmesh_init.GetComm());
   finder.Setup(pmesh_lor);
   finder.Interpolate(pos_quad_final, u_initial_gf, interp_vals);

   QuadratureFunction u_interpolated(qspace_final);
   u_interpolated = interp_vals;

   // Report masses.
   const double mass_s = Mass(*pmesh_init.GetNodes(), u_initial),
                mass_t = Mass(pos_final, u_interpolated);
   if (pmesh_init.GetMyRank() == 0)
   {
      std::cout << "Mass initial: " << mass_s << std::endl
                << "Mass final  : " << mass_t << std::endl
                << "Mass diff  : " << fabs(mass_s - mass_t) << endl
                << "Mass diff %: " << fabs(mass_s - mass_t)/mass_s*100 << endl;
   }

   // Compute min / max bounds.
   Vector u_final_min, u_final_max;
   CalcQuadBounds(u_initial, pos_final, u_final_min, u_final_max);
   if (vis_bounds)
   {
      QuadratureFunction gf_min(qspace_final), gf_max(qspace_final);
      gf_min = u_final_min, gf_max = u_final_max;

      *x = pos_final;

      osockstream sol_sock_min(19916, "localhost");
      sol_sock_min << "parallel " << pmesh_init.GetNRanks() << " " << pmesh_init.GetMyRank() << "\n";
      sol_sock_min << "quadrature\n" << pmesh_init << gf_min << std::flush;
      sol_sock_min << "window_title 'Min QuadFunc'\n";
      sol_sock_min << "window_geometry 0 500 400 400\n";
      sol_sock_min << "keys rmj\n";
      sol_sock_min.send();

      osockstream sol_sock_max(19916, "localhost");
      sol_sock_max << "parallel " << pmesh_init.GetNRanks() << " " << pmesh_init.GetMyRank() << "\n";
      sol_sock_max << "quadrature\n" << pmesh_init << gf_max << std::flush;
      sol_sock_max << "window_title 'Min QuadFunc'\n";
      sol_sock_max << "window_geometry 400 500 400 400\n";
      sol_sock_max << "keys rmj\n";
      sol_sock_max.send();

      *x = pos_init;
   }

   // Do some optimization here to fix the masses, using the min/max bounds,
   // staying as close as possible to u_interpolated.
   u_final = u_interpolated;
}

void InterpolationRemap::GetDOFPositions(const ParFiniteElementSpace &pfes,
                                         const Vector &pos_mesh,
                                         Vector &pos_dofs)
{
   const int NE  = pfes.GetNE(), dim = pmesh_init.Dimension();
   const int nsp = pfes.GetFE(0)->GetNodes().GetNPoints();

   pos_dofs.SetSize(nsp * NE * dim);
   for (int e = 0; e < NE; e++)
   {
      const IntegrationRule &ir = pfes.GetFE(e)->GetNodes();

      // Transformation of the element with the pos_mesh coordinates.
      IsoparametricTransformation Tr;
      pmesh_init.GetElementTransformation(e, pos_mesh, &Tr);

      // Node positions of pfes for pos_mesh.
      DenseMatrix pos_nodes;
      Tr.Transform(ir, pos_nodes);
      Vector rowx(pos_dofs.GetData() + e*nsp, nsp),
             rowy(pos_dofs.GetData() + e*nsp + NE*nsp, nsp), rowz;
      if (dim == 3)
      {
         rowz.SetDataAndSize(pos_dofs.GetData() + e*nsp + 2*NE*nsp, nsp);
      }
      pos_nodes.GetRow(0, rowx);
      pos_nodes.GetRow(1, rowy);
      if (dim == 3) { pos_nodes.GetRow(2, rowz); }
   }
}

void InterpolationRemap::GetQuadPositions(const QuadratureSpace &qspace,
                                          const Vector &pos_mesh,
                                          Vector &pos_quads)
{
   const int NE  = qspace.GetMesh()->GetNE(), dim = pmesh_init.Dimension();
   const int nsp = qspace.GetElementIntRule(0).GetNPoints();

   pos_quads.SetSize(nsp * NE * dim);
   for (int e = 0; e < NE; e++)
   {
      const IntegrationRule &ir = qspace.GetElementIntRule(e);

      // Transformation of the element with the pos_mesh coordinates.
      IsoparametricTransformation Tr;
      pmesh_init.GetElementTransformation(e, pos_mesh, &Tr);

      // Node positions of pfes for pos_mesh.
      DenseMatrix pos_quads_e;
      Tr.Transform(ir, pos_quads_e);
      Vector rowx(pos_quads.GetData() + e*nsp, nsp),
             rowy(pos_quads.GetData() + e*nsp + NE*nsp, nsp), rowz;
      if (dim == 3)
      {
         rowz.SetDataAndSize(pos_quads.GetData() + e*nsp + 2*NE*nsp, nsp);
      }
      pos_quads_e.GetRow(0, rowx);
      pos_quads_e.GetRow(1, rowy);
      if (dim == 3) { pos_quads_e.GetRow(2, rowz); }
   }
}

double InterpolationRemap::Mass(const Vector &pos, const ParGridFunction &g)
{
   double mass = 0.0;

   const int NE = g.ParFESpace()->GetNE();
   for (int e = 0; e < NE; e++)
   {
      auto el = g.ParFESpace()->GetFE(e);
      auto ir = IntRules.Get(el->GetGeomType(), el->GetOrder() + 2);
      IsoparametricTransformation Tr;
      // Must be w.r.t. the given positions.
      g.ParFESpace()->GetParMesh()->GetElementTransformation(e, pos, &Tr);

      Vector g_vals(ir.GetNPoints());
      g.GetValues(Tr, ir, g_vals);

      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         mass += Tr.Weight() * ip.weight * g_vals(q);
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &mass, 1, MPI_DOUBLE, MPI_SUM,
                 pmesh_init.GetComm());
   return mass;
}

double InterpolationRemap::Mass(const Vector &pos, const QuadratureFunction &q)
{
   double mass = 0.0;

   auto qspace = dynamic_cast<const QuadratureSpace *>(q.GetSpace());
   const int NE = qspace->GetMesh()->GetNE();
   for (int e = 0; e < NE; e++)
   {
      const IntegrationRule &ir = qspace->GetElementIntRule(e);

      IsoparametricTransformation Tr;
      // Must be w.r.t. the given positions.
      qspace->GetMesh()->GetElementTransformation(e, pos, &Tr);

      Vector g_vals(ir.GetNPoints());
      q.GetValues(e, g_vals);

      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         mass += Tr.Weight() * ip.weight * g_vals(q);
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &mass, 1, MPI_DOUBLE, MPI_SUM,
                 pmesh_init.GetComm());
   return mass;
}

void InterpolationRemap::CalcDOFBounds(const ParGridFunction &g_init,
                                       const ParFiniteElementSpace &pfes,
                                       const Vector &pos_final,
                                       Vector &g_min, Vector &g_max)
{
   const int size_res = pfes.GetVSize();
   g_min.SetSize(size_res);
   g_max.SetSize(size_res);

   // Form the min and max functions on every MPI task.
   L2_FECollection fec_L2(0, pmesh_init.Dimension());
   ParFiniteElementSpace pfes_L2(&pmesh_init, &fec_L2);
   ParGridFunction g_el_min(&pfes_L2), g_el_max(&pfes_L2);
   for (int e = 0; e < pmesh_init.GetNE(); e++)
   {
      Vector g_vals;
      g_init.GetElementDofValues(e, g_vals);
      g_el_min(e) = g_vals.Min();
      g_el_max(e) = g_vals.Max();
   }

   Vector pos_nodes_final;
   GetDOFPositions(pfes, pos_final, pos_nodes_final);

   FindPointsGSLIB finder(pmesh_init.GetComm());
   finder.Setup(pmesh_init);
   finder.Interpolate(pos_nodes_final, g_el_min, g_min);
   finder.Interpolate(pos_nodes_final, g_el_max, g_max);
}

void InterpolationRemap::CalcQuadBounds(const QuadratureFunction &qf_init,
                                        const Vector &pos_final,
                                        Vector &g_min, Vector &g_max)
{
   const int size_res = qf_init.Size();
   g_min.SetSize(size_res);
   g_max.SetSize(size_res);

   // Form the min and max functions on every MPI task.
   L2_FECollection fec_L2(0, pmesh_init.Dimension());
   ParFiniteElementSpace pfes_L2(&pmesh_init, &fec_L2);
   ParGridFunction g_el_min(&pfes_L2), g_el_max(&pfes_L2);
   for (int e = 0; e < pmesh_init.GetNE(); e++)
   {
      Vector q_vals;
      qf_init.GetValues(e, q_vals);
      g_el_min(e) = q_vals.Min();
      g_el_max(e) = q_vals.Max();
   }

   Vector pos_quads_final;
   auto qspace = dynamic_cast<const QuadratureSpace *>(qf_init.GetSpace());
   GetQuadPositions(*qspace, pos_final, pos_quads_final);

   FindPointsGSLIB finder(pmesh_init.GetComm());
   finder.Setup(pmesh_init);
   finder.Interpolate(pos_quads_final, g_el_min, g_min);
   finder.Interpolate(pos_quads_final, g_el_max, g_max);
}

} // namespace mfem
