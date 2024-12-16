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

void InitializeQuadratureFunction(Coefficient &c,
                                  const Vector &pos_mesh,
                                  QuadratureFunction &q)
{
   auto qspace = dynamic_cast<QuadratureSpace *>(q.GetSpace());
   MFEM_VERIFY(qspace, "Broken QuadratureSpace.");

   const int NE  = qspace->GetMesh()->GetNE();
   real_t *q_data = q.GetData();
   for (int e = 0; e < NE; e++)
   {
      const IntegrationRule &ir = qspace->GetElementIntRule(e);
      const int nip = ir.GetNPoints();

      // Transformation of the element with the pos_mesh coordinates.
      IsoparametricTransformation Tr;
      qspace->GetMesh()->GetElementTransformation(e, pos_mesh, &Tr);

      for (int q = 0; q < nip; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         q_data[e*nip + q] = c.Eval(Tr, ip);
      }
   }
}

void VisQuadratureFunction(ParMesh &pmesh, QuadratureFunction &q,
                           std::string info, int x, int y)
{
   osockstream sol_sock(19916, "localhost");
   sol_sock << "parallel " << pmesh.GetNRanks() << " "
                           << pmesh.GetMyRank() << "\n";
   sol_sock << "quadrature\n" << pmesh << q << std::flush;
   sol_sock << "window_title '" << info << "'\n";
   sol_sock << "window_geometry " << x << " " << y << " 400 400\n";
   sol_sock << "keys rmj\n";
   sol_sock.send();
}

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

void InterpolationRemap::Remap(const QuadratureFunction &u_0,
                               const ParGridFunction &pos_final,
                               QuadratureFunction &u)
{
   const int dim = pmesh_init.Dimension();
   MFEM_VERIFY(dim > 1, "Interpolation remap works only in 2D and 3D.");

   auto qspace = dynamic_cast<QuadratureSpace *>(u.GetSpace());
   MFEM_VERIFY(qspace, "Broken QuadratureSpace.");

   // Generate list of points where u_initial will be interpolated.
   Vector pos_quad_final;
   GetQuadPositions(*qspace, pos_final, pos_quad_final);

   // Generate the Low-Order-Refined GridFunction for interpolation.
   const int order = u_0.GetIntRule(0).GetOrder() / 2;
   const int ref_factor = order + 1;
   ParMesh pmesh_lor = ParMesh::MakeRefined(pmesh_init, ref_factor,
                                            BasisType::ClosedGL);
   L2_FECollection fec_lor(0, dim);
   ParFiniteElementSpace pfes_lor(&pmesh_lor, &fec_lor);
   ParGridFunction u_0_lor(&pfes_lor);
   MFEM_VERIFY(u_0.Size() == u_0_lor.Size(), "Size mismatch");
   u_0_lor = u_0;

   // Visualize the initial LOR GridFunction.
   socketstream sock;
   VisualizeField(sock, "localhost", 19916, u_0_lor, "u_0 LOR", 800, 0, 400, 400);

   // Interpolate u_initial.
   const int quads_cnt = pos_quad_final.Size() / dim;
   FindPointsGSLIB finder(pmesh_init.GetComm());
   finder.Setup(pmesh_lor);
   finder.Interpolate(pos_quad_final, u_0_lor, u);

   // Report mass error.
   const double mass_0 = Integrate(*pmesh_init.GetNodes(), &u_0,
                                   nullptr, nullptr),
                mass_f = Integrate(pos_final, &u,
                                   nullptr, nullptr);
   if (pmesh_init.GetMyRank() == 0)
   {
      std::cout << "Mass initial: " << mass_0 << std::endl
                << "Mass final  : " << mass_f << std::endl
                << "Mass diff  : " << fabs(mass_0 - mass_f) << endl
                << "Mass diff %: " << fabs(mass_0 - mass_f)/mass_0*100 << endl;
   }

   // Compute min / max bounds.
   Vector u_min, u_max;
   CalcQuadBounds(u_0, pos_final, u_min, u_max);
   if (vis_bounds)
   {
      QuadratureFunction gf_min(qspace), gf_max(qspace);
      gf_min = u_min, gf_max = u_max;

      *x = pos_final;

      VisQuadratureFunction(pmesh_init, gf_min, "u_min QF", 0, 500);
      VisQuadratureFunction(pmesh_init, gf_max, "u_max QF", 400, 500);

      *x = pos_init;
   }

   // Optimize u here.
   // ...
}

void InterpolationRemap::RemapIndRhoE(const Vector ind_rho_e_0,
                                      const ParGridFunction &pos_final,
                                      Vector &ind_rho_e)
{
   const int dim = pmesh_init.Dimension();
   MFEM_VERIFY(dim > 1, "Interpolation remap works only in 2D and 3D.");
   MFEM_VERIFY(pfes_e && qspace, "Spaces are not specified.");

   // Extract initial data from the BlockVector.
   const int size_e = pfes_e->GetNDofs(), size_qf = qspace->GetSize();
   Vector *ire_ptr = const_cast<Vector *>(&ind_rho_e_0);
   QuadratureFunction ind_0(qspace, ire_ptr->GetData()),
                      rho_0(qspace, ire_ptr->GetData() + size_qf);
   ParGridFunction e_0(pfes_e, ire_ptr->GetData() + 2*size_qf);

   // Generate list of points where ire_initial will be interpolated.
   Vector pos_dof_final, pos_quad_final;
   GetDOFPositions(*pfes_e, pos_final, pos_dof_final);
   GetQuadPositions(*qspace, pos_final, pos_quad_final);

   // Generate the Low-Order-Refined GridFunctions for
   // interpolating the QuadratureFunctions.
   const int order = qspace->GetIntRule(0).GetOrder() / 2;
   const int ref_factor = order + 1;
   ParMesh pmesh_lor = ParMesh::MakeRefined(pmesh_init, ref_factor,
                                            BasisType::ClosedGL);
   L2_FECollection fec_lor(0, dim);
   ParFiniteElementSpace pfes_lor(&pmesh_lor, &fec_lor);
   ParGridFunction ind_0_lor(&pfes_lor), rho_0_lor(&pfes_lor);
   MFEM_VERIFY(ind_0.Size() == ind_0_lor.Size(), "Size mismatch ind LOR.");
   MFEM_VERIFY(rho_0.Size() == rho_0_lor.Size(), "Size mismatch rho LOR.");
   ind_0_lor = ind_0;
   rho_0_lor = rho_0;

   // Visualize the initial LOR GridFunctions.
   socketstream sock_ind, sock_rho;
   VisualizeField(sock_ind, "localhost", 19916, ind_0_lor, "ind_0 LOR", 0, 500, 400, 400);
   VisualizeField(sock_rho, "localhost", 19916, rho_0_lor, "rho_0 LOR", 400, 500, 400, 400);

   // Interpolate into ind_rho_e.
   const int quads_cnt = pos_quad_final.Size() / dim;
   const int nodes_cnt = pos_dof_final.Size() / dim;
   QuadratureFunction ind(qspace, ind_rho_e.GetData()),
                      rho(qspace, ind_rho_e.GetData() + size_qf);
   ParGridFunction e(pfes_e, ind_rho_e.GetData() + 2*size_qf);
   FindPointsGSLIB finder(pmesh_init.GetComm());
   finder.Setup(pmesh_lor);
   finder.Interpolate(pos_quad_final, ind_0_lor, ind);
   finder.Interpolate(pos_quad_final, rho_0_lor, rho);
   finder.Setup(pmesh_init);
   finder.Interpolate(pos_dof_final, e_0, e);

   // Report conservation errors of ire_final.
   const double volume_0 = Integrate(*pmesh_init.GetNodes(), &ind_0,
                                     nullptr, nullptr),
                volume_f = Integrate(pos_final, &ind,
                                     nullptr, nullptr),
                mass_0   = Integrate(*pmesh_init.GetNodes(), &ind_0, &rho_0,
                                     nullptr),
                mass_f   = Integrate(pos_final, &ind, &rho,
                                     nullptr),
                energy_0 = Integrate(*pmesh_init.GetNodes(), &ind_0, &rho_0, &e_0),
                energy_f = Integrate(pos_final, &ind, &rho, &e);
   if (pmesh_init.GetMyRank() == 0)
   {
      std::cout << "Volume initial: " << volume_0 << std::endl
                << "Volume final  : " << volume_f << std::endl
                << "Volume diff   : " << fabs(volume_0 - volume_f) << endl
                << "Volume diff % : " << fabs(volume_0 - volume_f)/volume_0*100
                << endl << "*\n"
                << "Mass initial: " << mass_0 << std::endl
                << "Mass final  : " << mass_f << std::endl
                << "Mass diff   : " << fabs(mass_0 - mass_f) << endl
                << "Mass diff % : " << fabs(mass_0 - mass_f)/mass_0*100
                << endl << "*\n"
                << "Energy initial: " << energy_0 << std::endl
                << "Energy final  : " << energy_f << std::endl
                << "Energy diff   : " << fabs(energy_0 - energy_f) << endl
                << "Energy diff % : " << fabs(energy_0 - energy_f)/energy_0*100
                << endl;
   }

   // Compute min / max bounds.
   Vector ind_min, ind_max;
   CalcQuadBounds(ind_0, pos_final, ind_min, ind_max);
   Vector rho_min, rho_max;
   CalcQuadBounds(rho_0, pos_final, rho_min, rho_max);
   Vector e_min, e_max;
   CalcDOFBounds(e_0, *pfes_e, pos_final, e_min, e_max);

   // Optimize ire_final here.
   // ...
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

double InterpolationRemap::Integrate(const Vector &pos,
                                     const QuadratureFunction *q1,
                                     const QuadratureFunction *q2,
                                     const ParGridFunction *g1)
{
   MFEM_VERIFY(q1 || q2 || g1, "At least one function must be specified.");

   const QuadratureSpace *qspace = nullptr;
   if (q1) { qspace = dynamic_cast<const QuadratureSpace *>(q1->GetSpace()); }
   if (q2) { qspace = dynamic_cast<const QuadratureSpace *>(q2->GetSpace()); }

   auto mesh = (qspace) ? qspace->GetMesh() : g1->ParFESpace()->GetMesh();
   const int NE = mesh->GetNE();
   double integral = 0.0;
   for (int e = 0; e < NE; e++)
   {
      const IntegrationRule &ir =
          (qspace) ? qspace->GetElementIntRule(e)
                   : IntRules.Get(g1->ParFESpace()->GetFE(e)->GetGeomType(), 7);
      const int nqp = ir.GetNPoints();

      // Transformation w.r.t. the given mesh positions.
      IsoparametricTransformation Tr;
      mesh->GetElementTransformation(e, pos, &Tr);

      Vector q1_vals(nqp), q2_vals(nqp), g1_vals(nqp);
      if (q1) { q1->GetValues(e, q1_vals); } else { q1_vals = 1.0; }
      if (q2) { q2->GetValues(e, q2_vals); } else { q2_vals = 1.0; }
      if (g1) { g1->GetValues(Tr, ir, g1_vals); } else { g1_vals = 1.0; }

      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         integral += Tr.Weight() * ip.weight *
                     q1_vals(q) * q2_vals(q) * g1_vals(q);
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &integral, 1, MPI_DOUBLE, MPI_SUM,
                 pmesh_init.GetComm());
   return integral;
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
