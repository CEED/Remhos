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
#include "config/config.hpp"
#include "linalg/vector.hpp"
#include "remhos_tools.hpp"
#include "remhos_HiOp.hpp"
#include "remhos_lvpp.hpp"

#include "examples/remap_opt.hpp"
#include <algorithm>

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

void InterpolationRemap::Remap(const ParGridFunction &u_init,
                               const Vector &pos_final,
                               Vector &u_final, int opt_type)
{
   pmesh_final.SetNodes(pos_final);
   ParFiniteElementSpace pfes_final(&pmesh_final,
                                    u_init.ParFESpace()->FEColl());

   const int dim = pmesh_init.Dimension(), myid = pmesh_init.GetMyRank();
   MFEM_VERIFY(dim > 1, "Interpolation remap works only in 2D and 3D.");

   {
      ParaViewDataCollection pvdc("initial_mesh", &pmesh_init);
      pvdc.SetDataFormat(VTKFormat::BINARY32);
      pvdc.SetCycle(0);
      pvdc.SetTime(1.0);

      pvdc.RegisterField("val", const_cast<ParGridFunction*>(&u_init));
      pvdc.Save();
   }

   // Generate list of points where u_initial will be interpolated.
   Vector pos_dof_final;
   GetDOFPositions(pfes_final, pos_final, pos_dof_final);

   // Interpolate u_initial.
   const int nodes_cnt = pos_dof_final.Size() / dim;
   Vector interp_vals(nodes_cnt);
   FindPointsGSLIB finder(pmesh_init.GetComm());
   finder.Setup(pmesh_init);
   finder.Interpolate(pos_dof_final, u_init, interp_vals);
   finder.FreeData();

   // This assumes L2 ordering of the DOFs (as the ordering of the quad points).
   ParGridFunction u_interpolated(&pfes_final);
   u_interpolated = interp_vals;

   // Report masses.
   double mass_0 = Mass(*pmesh_init.GetNodes(), u_init),
          mass_f = Mass(pos_final, u_interpolated);
   if (myid == 0)
   {
      std::cout << "Mass initial (old mesh):  " << mass_0 << std::endl
                << "Mass interpolated:        " << mass_f << std::endl
                << "Mass interpolated diff:   "
                << fabs(mass_0 - mass_f) << endl
                << "Mass interpolated diff %: "
                << fabs(mass_0 - mass_f) / mass_0 * 100 << endl;
   }

   // Compute min / max bounds.
   Vector u_final_min, u_final_max;
   CalcDOFBounds(u_init, pfes_final, pos_final,
                 u_final_min, u_final_max, false);

   if (visualization)
   {
      ParGridFunction gf_min(u_init), gf_max(u_init);
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

      {
         ParaViewDataCollection pvdc("bounds", &pmesh_init);
         pvdc.SetDataFormat(VTKFormat::BINARY32);
         pvdc.SetCycle(0);
         pvdc.SetTime(1.0);
         pvdc.RegisterField("field_min", &gf_min);
         pvdc.RegisterField("field_max", &gf_max);
         pvdc.RegisterField("val", &u_interpolated);
         pvdc.Save();
      }

      *x = pos_init;
   }

   //
   // Optimize to fix the masses, using the min/max bounds.
   //
   if (opt_type == 0)
   {
      u_final = u_interpolated;
   }
   else if (opt_type == 1)
   {
      OptimizationSolver* optsolver = NULL;
      {
#ifdef MFEM_USE_HIOP
         optsolver = new HiopNlpOptimizer(MPI_COMM_WORLD);
#else
         MFEM_ABORT("MFEM is not built with HiOp support!");
#endif
      }

      Vector y_out(u_interpolated.Size());
      y_out = u_interpolated;

      int NumDesVar = u_interpolated.Size();
      mfem::Array<int> optProbInd;
      mfem::Vector u_interpolated_sub;
      mfem::Vector y_out_sub;

      Vector u_final_min_copy, u_final_max_copy;
      mfem::Vector minsub;
      mfem::Vector maxsub;

      u_final_min_copy = u_final_min;
      u_final_max_copy = u_final_max;

      if (subprob)
      {
         NumDesVar = GetSizeOptimizationSubset(u_final_min_copy,u_final_max_copy);
         GetOptimizationSubsetInd(u_final_min_copy,u_final_max_copy,optProbInd);
         u_interpolated.GetSubVector(optProbInd,u_interpolated_sub);
         y_out.GetSubVector(optProbInd,y_out_sub);

         u_final_min_copy.GetSubVector(optProbInd,minsub);
         u_final_max_copy.GetSubVector(optProbInd,maxsub);

         u_final_min_copy.SetSize(NumDesVar); u_final_min_copy= minsub;
         u_final_max_copy.SetSize(NumDesVar); u_final_max_copy= maxsub;
      }

      const int numContraints = 1;
      RemhosHiOpProblem ot_prob(pfes_final,
                                u_interpolated, NumDesVar,
                                u_final_min_copy, u_final_max_copy,
                                mass_0, numContraints, h1_seminorm, optProbInd, subprob);
      optsolver->SetOptimizationProblem(ot_prob);

      optsolver->SetMaxIter(max_iter);
      optsolver->SetAbsTol(1e-7);
      optsolver->SetRelTol(1e-7);
      optsolver->SetPrintLevel(3);
      
      if (subprob)
      {
         optsolver->Mult(u_interpolated_sub, y_out_sub);
         y_out.SetSubVector(optProbInd,y_out_sub);
      }
      else { optsolver->Mult(u_interpolated, y_out); }

      // fix parallel. u_interpolated and y_out should be true vectors
      u_final = y_out;

      delete optsolver;
   }
   else if (opt_type == 2)
   {
      MDSolver md(pfes_final, mass_0, u_interpolated, u_final_min, u_final_max);
      md.Optimize(1000, 1000, max_iter);

      ParGridFunction u_final_gf(&pfes_final);
      md.SetFinal(u_final_gf);
      u_final = u_final_gf;
   }
   else if (opt_type == 3)
   {
      GridFunctionCoefficient u_interpolated_cf(&u_interpolated);
      L2Obj obj(pfes_final, u_interpolated_cf);
      BoxMirrorDescent md(obj, u_final, u_final_min, u_final_max, max_iter);
      Vector target_volume(1); target_volume[0] = mass_0;
      ScalarLatentVolumeProjector projector(target_volume, pos_final,
                                            pfes_final, u_final);
      md.AddProjector(projector);
      ParGridFunction psi(&pfes_final);
      psi = 0.0;
      md.SetVerbose(1);
      md.Optimize(psi);
      md.UpdatePrimal(psi);
   }
   else if (opt_type == 4)
   {
      Vector target_volume(1); target_volume[0] = mass_0;
      ScalarLatentVolumeProjector projector(target_volume, pos_final,
                                            pfes_final, u_final);
      ParGridFunction psi(u_interpolated);
      Vector search_l({infinity()}), search_r({-infinity()}), lambda(1);
      for (int i=0; i<u_interpolated.Size(); i++)
      {
         psi[i] = inv_sigmoid(psi[i], u_final_min[i], u_final_max[i]);
         search_l[0] = std::min(search_l[0], psi[i]);
         search_r[0] = std::max(search_r[0], psi[i]);
      }
      MPI_Allreduce(MPI_IN_PLACE, &search_l[0], 1, MFEM_MPI_REAL_T, MPI_MIN,
                    pmesh_init.GetComm());
      MPI_Allreduce(MPI_IN_PLACE, &search_r[0], 1, MFEM_MPI_REAL_T, MPI_MAX,
                    pmesh_init.GetComm());
      projector.SetVerbose(2);
      projector.Apply(psi, u_final_min, u_final_max, 1.0, search_l, search_r,
                      lambda, max_iter);
   }

   // Report masses.
   ParGridFunction u_final_gf(&pfes_final);
   u_final_gf = u_final;
   mass_f = Mass(pos_final, u_final_gf);
   if (myid == 0)
   {
      std::cout << "Mass optimized:           " << mass_f << std::endl
                << "Mass optimized diff:      "
                << fabs(mass_0 - mass_f) << endl
                << "Mass optimized diff %:    "
                << fabs(mass_0 - mass_f) / mass_0 * 100 << endl;
   }

   // Check for bounds violations.
   CheckBounds(myid, u_final, u_final_min, u_final_max);

   const double obj_L2 = Objective(u_interpolated, u_final_gf);
   if (myid == 0)
   {
      std::cout << "---\nObjective L2: " << obj_L2 << std::endl;
   }
}

void InterpolationRemap::Remap(const QuadratureFunction &u_init,
                               const Vector &pos_final,
                               Vector &u_final, int opt_type)
{
   pmesh_final.SetNodes(pos_final);
   QuadratureSpace qspace_final(pmesh_final, u_init.GetIntRule(0));

   const int dim = pmesh_init.Dimension(), myid = pmesh_init.GetMyRank();
   MFEM_VERIFY(dim > 1, "Interpolation remap works only in 2D and 3D.");

   // Generate list of points where u_initial will be interpolated.
   Vector pos_quad_final;
   GetQuadPositions(qspace_final, pos_final, pos_quad_final);

   // Generate the Low-Order-Refined GridFunction for interpolation.
   const int order = u_init.GetIntRule(0).GetOrder() / 2;
   const int ref_factor = order + 1;
   ParMesh pmesh_lor = ParMesh::MakeRefined(pmesh_init, ref_factor,
                       BasisType::ClosedGL);
   L2_FECollection fec_lor(0, dim);
   ParFiniteElementSpace pfes_lor(&pmesh_lor, &fec_lor);
   ParGridFunction u_0_lor(&pfes_lor);
   MFEM_VERIFY(u_init.Size() == u_0_lor.Size(), "Size mismatch");
   u_0_lor = u_init;

   // Visualize the initial LOR GridFunction.
   if (visualization)
   {
      socketstream sock;
      VisualizeField(sock, "localhost", 19916, u_0_lor, "u_0 LOR", 800, 0, 400, 400);
   }

   // Interpolate u_initial.
   const int quads_cnt = pos_quad_final.Size() / dim;
   Vector interp_vals(quads_cnt);
   FindPointsGSLIB finder(pmesh_init.GetComm());
   finder.Setup(pmesh_lor);
   finder.Interpolate(pos_quad_final, u_0_lor, interp_vals);
   finder.FreeData();

   QuadratureFunction u_interpolated(qspace_final);
   u_interpolated = interp_vals;

   // Report mass error.
   double mass_0 = Integrate(*pmesh_init.GetNodes(), &u_init,
                             nullptr, nullptr),
          mass_f = Integrate(pos_final, &u_interpolated, nullptr, nullptr);
   if (myid == 0)
   {
      std::cout << "Mass initial (old mesh):  " << mass_0 << std::endl
                << "Mass interpolated:        " << mass_f << std::endl
                << "Mass interpolated diff:   "
                << fabs(mass_0 - mass_f) << endl
                << "Mass interpolated diff %: "
                << fabs(mass_0 - mass_f)/mass_0*100 << endl;
   }

   // Compute min / max bounds.
   Vector u_min, u_max;
   CalcQuadBounds(u_init, u_interpolated, pos_final, u_min, u_max, ELEM_FINAL);
   if (visualization)
   {
      QuadratureFunction gf_min(qspace_final), gf_max(qspace_final);
      gf_min = u_min, gf_max = u_max;

      VisQuadratureFunction(pmesh_final, gf_min, "u_min QF", 0, 500);
      VisQuadratureFunction(pmesh_final, gf_max, "u_max QF", 400, 500);
   }

   //
   // Optimize to fix the masses, using the min/max bounds.
   //
   if (opt_type == 0)
   {
      u_final = u_interpolated;
   }
   else if (opt_type == 1)
   {
      QuadratureFunction u_desing(u_interpolated), u_initial(u_interpolated);
      OptimizationSolver* optsolver = NULL;
      {
#ifdef MFEM_USE_HIOP
         HiopNlpOptimizer *tmp_opt_ptr = new HiopNlpOptimizer(MPI_COMM_WORLD);
         optsolver = tmp_opt_ptr;
#else
         MFEM_ABORT("MFEM is not built with HiOp support!");
#endif
      }

      const double rtol = 1.e-6;
      const double atol = 1.e-6;
      Vector y_out(u_desing.Size());

      const int numContraints = 1;
      const double H1SeminormWeight = 0.0;

      RemhosQuadHiOpProblem ot_prob(*qspace, pos_final,
                                    u_initial, u_desing,
                                    u_min, u_max, mass_0,
                                    numContraints, h1_seminorm);
      optsolver->SetOptimizationProblem(ot_prob);

      optsolver->SetMaxIter(max_iter);
      optsolver->SetAbsTol(atol);
      optsolver->SetRelTol(rtol);
      optsolver->SetPrintLevel(3);
      optsolver->Mult(u_desing, y_out);

      u_final = y_out;

      delete optsolver;
   }
   else if (opt_type == 2)
   {
      QDSolver qd(qspace_final, mass_0, u_interpolated, u_min, u_max);
      qd.Optimize(1000, 1000, max_iter);

      QuadratureFunction u_final_qf(qspace_final);
      qd.SetFinal(u_final_qf);
      u_final = u_final_qf;
   }
   else if (opt_type == 3)
   {
      QuadratureFunctionCoefficient u_target_cf(u_interpolated);
      L2Obj obj(qspace_final, u_target_cf);
      BoxMirrorDescent md(obj, u_final, u_min, u_max, max_iter);
      Vector target_volume(1); target_volume[0] = mass_0;
      ScalarLatentVolumeProjector projector(target_volume, pos_final,
                                            qspace_final, u_final);
      md.AddProjector(projector);
      QuadratureFunction psi(qspace_final);
      psi = 0.0;
      md.SetVerbose(1);
      md.Optimize(psi);
   }
   else if (opt_type == 4)
   {
      Vector target_volume(1); target_volume[0] = mass_0;
      ScalarLatentVolumeProjector projector(target_volume, pos_final,
                                            qspace_final, u_final);
      QuadratureFunction psi(u_interpolated);
      Vector search_l({infinity()}), search_r({-infinity()}), lambda(1);
      for (int i=0; i<psi.Size(); i++)
      {
         psi[i] = inv_sigmoid(psi[i], u_min[i], u_max[i]);
         search_l[0] = std::min(search_l[0], psi[i]);
         search_r[0] = std::max(search_r[0], psi[i]);
      }
      MPI_Allreduce(MPI_IN_PLACE, &search_l[0], 1, MFEM_MPI_REAL_T, MPI_MIN,
                    pmesh_init.GetComm());
      MPI_Allreduce(MPI_IN_PLACE, &search_r[0], 1, MFEM_MPI_REAL_T, MPI_MAX,
                    pmesh_init.GetComm());
      projector.SetVerbose(2);
      projector.Apply(psi, u_min, u_max, 1.0, search_l, search_r,
                      lambda, max_iter);
   }

   // Report final masses.
   QuadratureFunction u_final_qf(qspace_final);
   u_final_qf = u_final;
   mass_f = Integrate(pos_final, &u_final_qf, nullptr, nullptr);
   if (myid == 0)
   {
      std::cout << "Mass optimized:           " << mass_f << std::endl
                << "Mass optimized diff:      "
                << fabs(mass_0 - mass_f) << endl
                << "Mass optimized diff %:    "
                << fabs(mass_0 - mass_f)/mass_0*100 << endl;
   }

   // Check for bounds violations.
   CheckBounds(myid, u_final, u_min, u_max);

   u_interpolated -= u_final;
   real_t obj_l2 = pow(u_interpolated.Norml2(), 2.0);
   MPI_Allreduce(MPI_IN_PLACE, &obj_l2, 1, MFEM_MPI_REAL_T, MPI_SUM,
                 pmesh_init.GetComm());
   obj_l2 = sqrt(obj_l2);
   if (myid == 0)
   {
      std::cout << "---\nObjective l2: " << obj_l2 << std::endl;
   }
}

void InterpolationRemap::Remap(std::function<real_t(const Vector &)> func,
                               double mass, const Vector &pos_final,
                               ParGridFunction &u_final, int opt_type)
{
   pmesh_final.SetNodes(pos_final);
   ParFiniteElementSpace pfes_final(&pmesh_final,
                                    u_final.ParFESpace()->FEColl());
   ParFiniteElementSpace pfes_init(&pmesh_init,
                                    u_final.ParFESpace()->FEColl());

   const int dim = pmesh_init.Dimension();
   MFEM_VERIFY(dim > 1, "Interpolation remap works only in 2D and 3D.");

   // Generate list of points where u_initial will be interpolated.
   // The interpolation is to Gauss-Legendre to keep optimal order.
   L2_FECollection fec_GL(pfes_final.FEColl()->GetOrder(),
                          dim, BasisType::GaussLegendre);
   ParFiniteElementSpace pfes_GL(&pmesh_final, &fec_GL);
   Vector pos_dof_final;
   GetDOFPositions(pfes_GL, pos_final, pos_dof_final);

   // Interpolate the function.
   const int nodes_cnt = pos_dof_final.Size() / dim;
   Vector node_pos(dim);
   ParGridFunction u_interpolated_GL(&pfes_GL);
   for (int i = 0; i < nodes_cnt; i++)
   {
      for (int d = 0; d < dim; d++)
      {
         node_pos(d) = pos_dof_final(d * nodes_cnt + i);
      }
      // This assumes L2 ordering of the DOFs
      // (as the ordering of the quad points).
      u_interpolated_GL(i) = func(node_pos);
   }

   // Go Gauss-Legendre -> Bernstein.
   ParGridFunction u_interpolated(&pfes_final);
   u_interpolated.ProjectGridFunction(u_interpolated_GL);

   // Report masses.
   double mass_f = Mass(pos_final, u_interpolated);
   if (pmesh_init.GetMyRank() == 0)
   {
      std::cout << "Mass initial (analytic):  " << mass   << std::endl
                << "Mass interpolated:        " << mass_f << std::endl
                << "Mass interpolated diff:   "
                << fabs(mass - mass_f) << endl
                << "Mass interpolated diff %: "
                << fabs(mass - mass_f) / mass * 100 << endl;
   }

   // Compute min / max bounds.
   // Projects to a GridFunction to get some reasonable min/max per element.
   // It seems better to take it on the initial mesh, I guess the uniform
   // spacing gives more uniform bounds, and things converge better.
   Vector u_final_min, u_final_max;
   ParGridFunction func_gf(&pfes_init);
   FunctionCoefficient coeff(func);
   func_gf.ProjectCoefficient(coeff);
   CalcDOFBounds(func_gf, pfes_final, pos_final,
                 u_final_min, u_final_max, true);
   if (visualization)
   {
      ParGridFunction gf_min(func_gf), gf_max(func_gf);
      gf_min = u_final_min, gf_max = u_final_max;

      socketstream vis_min, vis_max;
      char vishost[] = "localhost";
      int  visport   = 19916;
      vis_min.precision(8);
      vis_max.precision(8);

      VisualizeField(vis_min, vishost, visport, gf_min, "u min",
                     0, 500, 300, 300);
      VisualizeField(vis_max, vishost, visport, gf_max, "u max",
                     300, 500, 300, 300);
   }

   //
   // Optimize to fix the masses, using the min/max bounds.
   //
   if (opt_type == 0)
   {
      u_final = u_interpolated;
   }
   else if (opt_type == 1)
   {
      OptimizationSolver* optsolver = NULL;
      {
#ifdef MFEM_USE_HIOP
         HiopNlpOptimizer *tmp_opt_ptr = new HiopNlpOptimizer(MPI_COMM_WORLD);
         optsolver = tmp_opt_ptr;
#else
         MFEM_ABORT("MFEM is not built with HiOp support!");
#endif
      }

      Vector y_out(u_final.Size());

      mfem::Array<int> optProbInd;

      const int numContraints = 1;
      RemhosHiOpProblem ot_prob(pfes_final, u_interpolated, u_final.Size(),
                                u_final_min, u_final_max, mass,
                                numContraints, h1_seminorm, optProbInd);
      optsolver->SetOptimizationProblem(ot_prob);

      optsolver->SetMaxIter(max_iter);
      optsolver->SetAbsTol(0.0);
      optsolver->SetRelTol(1e-12);
      optsolver->SetPrintLevel(3);
      optsolver->Mult(u_interpolated, y_out);

      u_final = y_out;

      delete optsolver;
   }
   else if (opt_type == 2)
   {
      MDSolver md(pfes_final, mass, u_interpolated, u_final_min, u_final_max);
      md.Optimize(100, 1000, max_iter);

      md.SetFinal(u_final);
   }
   else if (opt_type == 3)
   {
      GridFunctionCoefficient u_interpolated_cf(&u_interpolated);
      L2Obj obj(pfes_final, u_interpolated_cf);
      BoxMirrorDescent md(obj, u_final, u_final_min, u_final_max, max_iter);
      Vector target_volume(1); target_volume[0] = mass;
      ScalarLatentVolumeProjector projector(target_volume, pos_final,
                                            pfes_final, u_final);
      md.AddProjector(projector);
      ParGridFunction psi(&pfes_final);
      psi = 0.0;
      md.SetVerbose(1);
      md.Optimize(psi);
      md.UpdatePrimal(psi);
   }
   else if (opt_type == 4)
   {
      Vector target_volume(1); target_volume[0] = mass;
      ScalarLatentVolumeProjector projector(target_volume, pos_final,
                                            *u_final.ParFESpace(), u_final);
      ParGridFunction psi(u_interpolated);
      Vector search_l({infinity()}), search_r({-infinity()}), lambda(1);
      for (int i=0; i<u_interpolated.Size(); i++)
      {
         psi[i] = inv_sigmoid(psi[i], u_final_min[i], u_final_max[i]);
         search_l[0] = std::min(search_l[0], psi[i]);
         search_r[0] = std::max(search_r[0], psi[i]);
      }
      MPI_Allreduce(MPI_IN_PLACE, &search_l[0], 1, MFEM_MPI_REAL_T, MPI_MIN,
                    pmesh_init.GetComm());
      MPI_Allreduce(MPI_IN_PLACE, &search_r[0], 1, MFEM_MPI_REAL_T, MPI_MAX,
                    pmesh_init.GetComm());
      projector.SetVerbose(2);
      projector.Apply(psi, u_final_min, u_final_max, 1.0, search_l, search_r,
                      lambda, max_iter);
   }
   else { MFEM_ABORT("Optimization type not implemented"); }

   // Report masses.
   mass_f = Mass(pos_final, u_final);
   if (pmesh_init.GetMyRank() == 0)
   {
      std::cout << "Mass optimized:           " << mass_f << std::endl
                << "Mass optimized diff:      "
                << fabs(mass - mass_f) << endl
                << "Mass optimized diff %:    "
                << fabs(mass - mass_f) / mass * 100 << endl;
   }

   // Check for bounds violations.
   if (Mpi::Root()) { std::cout << "-------\nBounds violations: \n"; }
   CheckBounds(pmesh_init.GetMyRank(), u_final, u_final_min, u_final_max);
}

void InterpolationRemap::RemapIndRhoE(const Vector &ind_rho_e_0,
                                      Array<bool> &active_el_0,
                                      const ParGridFunction &pos_final,
                                      Vector &ind_rho_e, int opt_type)
{
   const int dim = pmesh_init.Dimension();
   MFEM_VERIFY(dim > 1, "Interpolation remap works only in 2D and 3D.");
   MFEM_VERIFY(pfes_e && qspace, "Spaces are not specified.");

   // Extract initial data from the BlockVector.
   const int size_qf = qspace->GetSize();
   int size_gf  = pfes_e->GetNDofs();
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
   if (visualization)
   {
      socketstream sock_ind, sock_rho;
      VisualizeField(sock_ind, "localhost", 19916, ind_0_lor, "ind_0 LOR", 0, 500,
                     400, 400);
      VisualizeField(sock_rho, "localhost", 19916, rho_0_lor, "rho_0 LOR", 400, 500,
                     400, 400);
   }

   // Interpolate into ind_rho_e.
   QuadratureFunction ind(qspace, ind_rho_e.GetData()),
                      rho(qspace, ind_rho_e.GetData() + size_qf);
   ParGridFunction e(pfes_e, ind_rho_e.GetData() + 2*size_qf);
   FindPointsGSLIB finder(pmesh_init.GetComm());
   finder.Setup(pmesh_lor);
   finder.Interpolate(pos_quad_final, ind_0_lor, ind);
   finder.Interpolate(pos_quad_final, rho_0_lor, rho);
   finder.Setup(pmesh_init);
   finder.Interpolate(pos_dof_final, e_0, e);
   finder.FreeData();

   // Report conservation errors of ire_final.
   const double volume_0 = Integrate(*pmesh_init.GetNodes(), &ind_0,
                                     nullptr, nullptr);
   const double volume_f = Integrate(pos_final, &ind,
                                     nullptr, nullptr);
   const double mass_0   = Integrate(*pmesh_init.GetNodes(), &ind_0, &rho_0,
                                     nullptr);
   const double mass_f   = Integrate(pos_final, &ind, &rho,
                                     nullptr);
   const double energy_0 = Integrate(*pmesh_init.GetNodes(), &ind_0, &rho_0, &e_0);
   const double energy_f = Integrate(pos_final, &ind, &rho, &e);

   if (pmesh_init.GetMyRank() == 0)
   {
      std::cout << "Volume initial:             " << volume_0 << std::endl
                << "Volume interpolated:        " << volume_f << std::endl
                << "Volume interpolated diff:   "
                << fabs(volume_0 - volume_f) << endl
                << "Volume interpolated diff %: "
                << fabs(volume_0 - volume_f) / volume_0 * 100
                << endl << "*\n"
                << "Mass initial:               " << mass_0 << std::endl
                << "Mass interpolated:          " << mass_f << std::endl
                << "Mass interpolated diff:     "
                << fabs(mass_0 - mass_f) << endl
                << "Mass interpolated diff %: "
                << fabs(mass_0 - mass_f) / mass_0 * 100
                << endl << "*\n"
                << "Energy initial:             " << energy_0 << std::endl
                << "Energy interpolated:        " << energy_f << std::endl
                << "Energy interpolated diff:   "
                << fabs(energy_0 - energy_f) << endl
                << "Energy interpolated diff %: "
                << fabs(energy_0 - energy_f) / energy_0 * 100
                << endl;
   }

   // Compute min / max bounds.
   Vector ind_min, ind_max;
   CalcQuadBounds(ind_0, ind, pos_final, ind_min, ind_max, ELEM_FINAL);
   Vector rho_min, rho_max;
   CalcRhoBounds(rho, ind, ind_max, rho_min, rho_max);
   // {
   //    QuadratureFunction gf_min(qspace), gf_max(qspace);
   //    gf_min = rho_min, gf_max = rho_max;

   //    *x = pos_final;
   //    VisQuadratureFunction(pmesh_init, ind, "ind interp", 0, 500);
   //    VisQuadratureFunction(pmesh_init, rho, "rho interp", 0, 500);
   //    VisQuadratureFunction(pmesh_init, gf_min, "rho_min QF", 0, 500);
   //    VisQuadratureFunction(pmesh_init, gf_max, "rho_max QF", 400, 500);
   //    *x = pos_init;
   //    MFEM_ABORT("rho bounds");
   // }
   Vector e_min, e_max;
   CalcEBounds(e, ind_max, e_min, e_max);
   // {
   //    ParGridFunction gf_min(e), gf_max(e);
   //    gf_min = e_min, gf_max = e_max;

   //    socketstream vis_min, vis_max;
   //    char vishost[] = "localhost";
   //    int  visport   = 19916;
   //    vis_min.precision(8);
   //    vis_max.precision(8);

   //    *x = pos_final;
   //    VisualizeField(vis_min, vishost, visport, gf_min, "e min",
   //                   0, 500, 300, 300);
   //    VisualizeField(vis_max, vishost, visport, gf_max, "e max",
   //                   300, 500, 300, 300);
   //    *x = pos_init;
   //    MFEM_ABORT("e bounds");
   // }

   Array<int> offset(4);
   offset[0] = 0;
   offset[1] = offset[0] + size_qf;
   offset[2] = offset[1] + size_qf;
   offset[3] = offset[2] + size_gf;

   BlockVector initial_design(offset);
   BlockVector x_min(offset);
   BlockVector x_max(offset);

   initial_design.GetBlock(0) = ind;
   initial_design.GetBlock(1) = rho;
   initial_design.GetBlock(2) = e;

   x_min.GetBlock(0) = ind_min;
   x_min.GetBlock(1) = rho_min;
   x_min.GetBlock(2) = e_min;
   x_max.GetBlock(0) = ind_max;
   x_max.GetBlock(1) = rho_max;
   x_max.GetBlock(2) = e_max;

   if (opt_type == 0) { }
   else if (opt_type == 1)
   {
      *x = pos_final;
      OptimizationSolver* optsolver = NULL;
      {
#ifdef MFEM_USE_HIOP
         optsolver = new HiopNlpOptimizer(MPI_COMM_WORLD);
#else
         MFEM_ABORT("MFEM is not built with HiOp support!");
#endif
      }

      Vector y_out(ind_rho_e.Size());
      y_out = initial_design;
      ind_rho_e = initial_design;

      int NumDesVar = ind_rho_e.Size();
      mfem::Array<int> optProbInd;
      mfem::Vector ind_rho_e_sub;
      mfem::Vector y_out_sub;
      mfem::Vector minsub;
      mfem::Vector maxsub;

      mfem::Vector x_maxsub(NumDesVar);
      mfem::Vector x_minsub(NumDesVar); 

      if(subprob)
      {
         NumDesVar = GetSizeOptimizationSubset(x_min,x_max);
         GetOptimizationSubsetInd(x_min,x_max,optProbInd);
         ind_rho_e.GetSubVector(optProbInd,ind_rho_e_sub);
         y_out.GetSubVector(optProbInd,y_out_sub);

         x_min.GetSubVector(optProbInd,minsub);
         x_max.GetSubVector(optProbInd,maxsub);

         x_maxsub.SetSize(NumDesVar);
         x_minsub.SetSize(NumDesVar);
         x_maxsub= maxsub;
         x_minsub= minsub;

      }   
      else{
         x_maxsub= x_max;
         x_minsub= x_min;
      }


      RemhosIndRhoEHiOpProblem ot_prob(*qspace, *pfes_e,
                                       pos_final,
                                       initial_design,
                                       ind_rho_e,
                                       NumDesVar,
                                       x_minsub, x_maxsub,
                                       volume_0, mass_0, energy_0,
                                       3, false, optProbInd, true, subprob);

      optsolver->SetOptimizationProblem(ot_prob);

      optsolver->SetMaxIter(max_iter);
      optsolver->SetAbsTol(1e-7);
      optsolver->SetRelTol(1e-7);
      optsolver->SetPrintLevel(3);

      if(subprob)
      {
         optsolver->Mult(ind_rho_e_sub, y_out_sub);
         y_out.SetSubVector(optProbInd,y_out_sub);
      }
      else{
         optsolver->Mult(ind_rho_e, y_out);
      }

      ind_rho_e = y_out;

      QuadratureFunction ind_temp(qspace, ind_rho_e.GetData());
      QuadratureFunction rho_temp(qspace, ind_rho_e.GetData() + size_qf);
      ParGridFunction    energy_temp(pfes_e, ind_rho_e.GetData() + 2*size_qf);

      ind = ind_temp;
      rho = rho_temp;
      e   = energy_temp;

      delete optsolver;
   }
   else if (opt_type == 4)
   {
      Vector target_volume(3);
      target_volume[0] = volume_0;
      target_volume[1] = mass_0;
      target_volume[2] = energy_0;
      IndRhoEVolumeProjectorCorrect projector(target_volume, pos_final,
                                              *qspace, *pfes_e, ind_rho_e);
      Vector psi(ind_rho_e);
      int offset = 0;
      for (int i=0; i<ind.Size(); i++)
      {
         psi[i] = inv_sigmoid(psi[i], ind_min[i], ind_max[i]);
      }
      offset += ind.Size();
      for (int i=0; i<rho.Size(); i++)
      {
         psi[offset + i] = inv_sigmoid(psi[offset + i], rho_min[i], rho_max[i]);
      }
      offset += rho.Size();
      L2_FECollection nodal_fec(pfes_e->GetOrder(0), pfes_e->GetParMesh()->Dimension());
      ParFiniteElementSpace pfes_nodal(pfes_e->GetParMesh(), &nodal_fec);
      ParGridFunction E_gf(pfes_e, ind_rho_e.GetData() + offset);
      ParGridFunction lower_gf(&pfes_nodal, e_min);
      ParGridFunction upper_gf(&pfes_nodal, e_max);
      LogitCoefficient logit_coeff(E_gf, lower_gf, upper_gf);
      ParGridFunction psi_gf(&pfes_nodal, psi.GetData() + offset);
      psi_gf.ProjectCoefficient(logit_coeff);
      projector.SetVerbose(1);
      Vector search_l, search_r, lambda; // not used anymore..
      projector.Apply(psi, x_min, x_max, 1e-01,
                      search_l, search_r, lambda, 1e03);
   }
   else { MFEM_ABORT("not implemented!"); }

   const double volume_f_opt = Integrate(pos_final, &ind, nullptr, nullptr);
   const double mass_f_opt   = Integrate(pos_final, &ind, &rho,    nullptr);
   const double energy_f_opt = Integrate(pos_final, &ind, &rho,    &e);
   if (Mpi::Root())
   {
      std::cout << "-------\n"
                << "Volume initial:          " << volume_0 << std::endl
                << "Volume optimized:        " << volume_f_opt << std::endl
                << "Volume optimized diff:   "
                << (volume_f_opt - volume_0) << endl
                << "Volume optimized diff %: "
                << (volume_f_opt - volume_0) / volume_0 * 100
                << endl << "*\n"
                << "Mass initial:            " << mass_0 << std::endl
                << "Mass optimized:          " << mass_f_opt << std::endl
                << "Mass optimized diff:     "
                << (mass_f_opt - mass_0) << endl
                << "Mass optimized diff %:   "
                << (mass_f_opt - mass_0) / mass_0 * 100
                << endl << "*\n"
                << "Energy initial:          " << energy_0 << std::endl
                << "Energy optimized:        " << energy_f_opt << std::endl
                << "Energy optimized diff:   "
                << (energy_f_opt- energy_0) << endl
                << "Energy optimized diff %: "
                << (energy_f_opt- energy_0) / energy_0 * 100
                << endl;
   }

   // Check for bounds violations.
   if (Mpi::Root()) { std::cout << "-------\nIndicator violations: \n"; }
   CheckBounds(pmesh_init.GetMyRank(), ind, ind_min, ind_max);
   if (Mpi::Root()) { std::cout << "*\nDensity violations: \n"; }
   CheckBounds(pmesh_init.GetMyRank(), rho, rho_min, rho_max);
   if (Mpi::Root()) { std::cout << "*\nInternal Energy violations: \n"; }
   CheckBounds(pmesh_init.GetMyRank(), e, e_min, e_max);
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
      Vector rowx(pos_dofs.GetData() + e*nsp, nsp),
             rowy(pos_dofs.GetData() + e*nsp + NE*nsp, nsp), rowz;
      if (dim == 3)
      {
         rowz.SetDataAndSize(pos_dofs.GetData() + e*nsp + 2*NE*nsp, nsp);
      }

      DenseMatrix pos_nodes;
      Tr.Transform(ir, pos_nodes);
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

double InterpolationRemap::Objective(const ParGridFunction &g_interp,
                                     const ParGridFunction &g)
{
   GridFunctionCoefficient ci(&g_interp);
   return g.ComputeL2Error(ci);
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
      if (q1) { q1->GetValues(e, q1_vals); }
      else { q1_vals = 1.0; }
      if (q2) { q2->GetValues(e, q2_vals); }
      else { q2_vals = 1.0; }
      if (g1) { g1->GetValues(Tr, ir, g1_vals); }
      else { g1_vals = 1.0; }

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

#define EMPTY_VALUE -1.0

void InterpolationRemap::CalcDOFBounds(const ParGridFunction &g_init,
                                       const ParFiniteElementSpace &pfes,
                                       const Vector &pos_final,
                                       Vector &g_min, Vector &g_max,
                                       bool use_el_nbr, Array<bool> *active_el)
{
   if (active_el)
   {
      MFEM_VERIFY(use_el_nbr == true,
                  "Bounds around inactive elements require use_el_nbr = true.");
   }

   const int size_res = pfes.GetVSize(), NE = pmesh_init.GetNE();
   g_min.SetSize(size_res);
   g_max.SetSize(size_res);

   // Form the min and max functions on every MPI task.
   // All on the initial mesh.
   L2_FECollection fec_L2(0, pmesh_init.Dimension());
   ParFiniteElementSpace pfes_L2(&pmesh_init, &fec_L2);
   ParGridFunction g_el_min(&pfes_L2), g_el_max(&pfes_L2);
   for (int e = 0; e < NE; e++)
   {
      if (active_el && (*active_el)[e] == false)
      {
         g_el_min(e) = EMPTY_VALUE;
         g_el_max(e) = EMPTY_VALUE;
         continue;
      }

      Vector g_init_vals;
      g_init.GetElementDofValues(e, g_init_vals);
      g_el_min(e) = g_init_vals.Min();
      g_el_max(e) = g_init_vals.Max();
   }

   Vector pos_nodes_final;
   GetDOFPositions(pfes, pos_final, pos_nodes_final);

   FindPointsGSLIB finder(pmesh_init.GetComm());
   finder.Setup(pmesh_init);
   finder.Interpolate(pos_nodes_final, g_el_min, g_min);
   finder.Interpolate(pos_nodes_final, g_el_max, g_max);
   finder.FreeData();

   // On the new mesh, take min/max over DOFs in the same element.
   if (use_el_nbr)
   {
      for (int e = 0; e < NE; e++)
      {
         Array<int> dofs;
         pfes.GetElementDofs(e, dofs);
         const int s = dofs.Size();

         Vector g_min_el, g_max_el;
         g_min.GetSubVector(dofs, g_min_el);
         g_max.GetSubVector(dofs, g_max_el);
         double min_el =   std::numeric_limits<double>::infinity(),
                max_el = - std::numeric_limits<double>::infinity();
         bool has_value = false;
         for (int i = 0; i < s; i++)
         {
            if (g_min_el(i) != EMPTY_VALUE)
            {
               has_value = true;
               min_el = std::min(min_el, g_min_el(i));
               max_el = std::max(max_el, g_max_el(i));
            }
         }

         // The element is completely empty -> we want to get zeros in it.
         if (has_value == false) { min_el = max_el = 0.0; }

         for (int i = 0; i < s; i++)
         {
            g_min(s * e + i) = min_el;
            g_max(s * e + i) = max_el;
         }
      }
   }
}

void InterpolationRemap::CalcQuadBounds(const QuadratureFunction &qf_init,
                                        const QuadratureFunction &qf_interp,
                                        const Vector &pos_final,
                                        Vector &g_min, Vector &g_max,
                                        BoundsType bounds_type)
{
   const int size_res = qf_init.Size(), NE = pmesh_init.GetNE();
   g_min.SetSize(size_res);
   g_max.SetSize(size_res);
   g_min = qf_interp;
   g_max = qf_interp;

   if (bounds_type == ELEM_INIT || bounds_type == ELEM_BOTH)
   {
      // Form the min and max functions on every MPI task.
      L2_FECollection fec_L2(0, pmesh_init.Dimension());
      ParFiniteElementSpace pfes_L2(&pmesh_init, &fec_L2);
      ParGridFunction g_el_min(&pfes_L2), g_el_max(&pfes_L2);
      for (int e = 0; e < NE; e++)
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
      finder.FreeData();
   }

   // On the new mesh, take min/max over quads in the same element.
   if (bounds_type == ELEM_FINAL || bounds_type == ELEM_BOTH)
   {
      int el_e_idx = 0;
      for (int e = 0; e < NE; e++)
      {
         const IntegrationRule &ir = qspace->GetElementIntRule(e);
         const int nqp = ir.GetNPoints();

         double min_el =   std::numeric_limits<double>::infinity(),
                max_el = - std::numeric_limits<double>::infinity();
         for (int q = 0; q < nqp; q++)
         {
            min_el = std::min(min_el, g_min(el_e_idx + q));
            max_el = std::max(max_el, g_max(el_e_idx + q));
         }

         for (int q = 0; q < nqp; q++)
         {
            g_min(el_e_idx + q) = min_el;
            g_max(el_e_idx + q) = max_el;
         }

         el_e_idx += nqp;
      }
   }
}

void InterpolationRemap::CalcRhoBounds(const QuadratureFunction &rho_interp,
                                       const QuadratureFunction &ind_interp,
                                       const Vector &ind_max,
                                       Vector &rho_min, Vector &rho_max)
{
   const int size_rho = rho_interp.Size(), NE = pmesh_init.GetNE();
   rho_min.SetSize(size_rho); rho_max.SetSize(size_rho);

   int el_e_idx = 0;
   for (int e = 0; e < NE; e++)
   {
      const IntegrationRule &ir = qspace->GetElementIntRule(e);
      const int nqp = ir.GetNPoints();

      // Min and max density in the new mesh element.
      double el_min =   std::numeric_limits<double>::infinity(),
             el_max = - std::numeric_limits<double>::infinity();
      bool el_has_ind_value = false, el_has_ind_max = false;
      for (int q = 0; q < nqp; q++)
      {
         // Bounds are taken only from points where material is present.
         if (ind_interp(el_e_idx + q) > 1e-12)
         {
            el_has_ind_value = true;
            el_min = std::min(el_min, rho_interp(el_e_idx + q));
            el_max = std::max(el_max, rho_interp(el_e_idx + q));
         }
         if (ind_max(el_e_idx + q) > 1e-12) { el_has_ind_max = true; }
      }

      // The new mesh element is completely empty -> we want zeros in it.
      if (el_has_ind_value == false)
      {
         MFEM_VERIFY(el_has_ind_max == false, "Indicator values / bound mess!");
         el_min = el_max = 0.0;
      }

      for (int q = 0; q < nqp; q++)
      {
         // Bounds are set only where the material will be present. This is a
         // special case for QuadratureFunctions to get sub-element behavior.
         if (ind_max(el_e_idx + q) > 1e-12)
         {
            rho_min(el_e_idx + q) = el_min;
            rho_max(el_e_idx + q) = el_max;
         }
         else
         {
            // No material at the DOF.
            rho_min(el_e_idx + q) = 0.0;
            rho_max(el_e_idx + q) = 0.0;
         }
      }

      el_e_idx += nqp;
   }
}

void InterpolationRemap::CalcEBounds(const ParGridFunction &e_interp,
                                     const Vector &ind_max,
                                     Vector &e_min, Vector &e_max)
{
   const int size_e = e_interp.Size(), NE = pmesh_init.GetNE();
   const int s = size_e / NE;
   e_min.SetSize(size_e); e_max.SetSize(size_e);

   for (int e = 0; e < NE; e++)
   {
      const IntegrationRule &ir = qspace->GetElementIntRule(e);
      const int nqp = ir.GetNPoints();

      // Check if the new mesh element has material.
      bool el_has_ind = false;
      for (int q = 0; q < nqp; q++)
      {
         if (ind_max(e*nqp + q) > 1e-12) { el_has_ind = true; break; }
      }

      // Min and max energy in the new mesh element.
      double el_min =   std::numeric_limits<double>::infinity(),
             el_max = - std::numeric_limits<double>::infinity();
      // The new mesh element is completely empty -> we want zeros in it.
      if (el_has_ind == false) { el_min = el_max = 0.0; }
      else
      {
         Vector e_interp_el;
         e_interp.GetElementDofValues(e, e_interp_el);
         bool el_has_e_value = true;
         for (int i = 0; i < s; i++)
         {
            // Don't consider zeros for the bounds. The assumption is that those
            // DOFs fall in initial mesh elements with no material.
            if (fabs(e_interp_el(i)) > 1e-14)
            {
               el_has_e_value = true;
               el_min = std::min(el_min, e_interp_el(i));
               el_max = std::max(el_max, e_interp_el(i));
            }
         }
         MFEM_VERIFY(el_has_e_value == true, "No e values in the new element!");
      }

      for (int i = 0; i < s; i++)
      {
         e_min(s * e + i) = el_min;
         e_max(s * e + i) = el_max;
      }
   }
}

void InterpolationRemap::CheckBounds(int myid, const Vector &v,
                                     const Vector &v_min, const Vector &v_max)
{
   int s = v.Size();
   int err_cnt = 0;
   double err_max = 0.0;
   for (int i = 0; i < s; i++)
   {
      if (v(i) < v_min(i) - 1e-12)
      {
         err_cnt++;
         err_max = std::max(err_max, v_min(i) - v(i));
      }
      if (v(i) > v_max(i) + 1e-12)
      {
         err_cnt++;
         err_max = std::max(err_max, v(i) - v_max(i));
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &s, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &err_cnt, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &err_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   double m = v.Max();
   MPI_Allreduce(MPI_IN_PLACE, &m, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

   if (myid == 0)
   {
      std::cout << "Bound errors: " << err_cnt
                << " (out of " << s << " values )\n"
                << "Max error:    " << err_max
                << " (max function value is " << m << ")\n";
   }
}

} // namespace mfem
