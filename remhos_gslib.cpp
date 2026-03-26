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
#include "remhos_HiOp.hpp"
#include "remhos_lvpp.hpp"

#include <algorithm>

using namespace std;

namespace mfem
{

void InitializeQuadratureFunction(int ind_id, Coefficient &c,
                                  const Vector &pos_mesh,
                                  QuadratureFunction &qf,
                                  const Array<bool> *active_quads)
{
   auto qspace = dynamic_cast<QuadratureSpace *>(qf.GetSpace());
   MFEM_VERIFY(qspace, "Broken QuadratureSpace.");

   const int NE  = qspace->GetMesh()->GetNE();
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
         Vector coord(Tr.GetSpaceDim());
         Tr.Transform(ip, coord);
         if (active_quads && (*active_quads)[e * nip + q] == false)
         {
            qf(e*nip + q) = 0.0;
         }
         else
         {
            if (ind_id == 0)
            {
               qf(e*nip + q) = c.Eval(Tr, ip);
            }
            else if (ind_id == 1 && coord(0) < 0.5)
            {
               qf(e*nip + q) = 1.0 - c.Eval(Tr, ip);
            }
            else if (ind_id == 2 && coord(0) > 0.5)
            {
               qf(e*nip + q) = 1.0 - c.Eval(Tr, ip);
            }
            else
            {
               qf(e*nip + q) = 0.0;
            }
         }
      }
   }
}

void InitializeRho(Coefficient &rho_coeff, const Vector &pos_mesh,
                   QuadratureFunction &rho_qf, const Array<bool> &active_quads)
{
   auto qspace = dynamic_cast<QuadratureSpace *>(rho_qf.GetSpace());
   MFEM_VERIFY(qspace, "Broken QuadratureSpace.");

   const int NE  = qspace->GetMesh()->GetNE();
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
         if (active_quads[e * nip + q] == false)
         {
            rho_qf(e*nip + q) = 0.0;
         }
         else
         {
            rho_qf(e*nip + q) = rho_coeff.Eval(Tr, ip);
         }
      }
   }
}

void ComputePressureQF(const QuadratureFunction &rho,
                       const ParGridFunction &energy,
                       QuadratureFunction &p)
{
   auto qspace = dynamic_cast<QuadratureSpace *>(p.GetSpace());
   MFEM_VERIFY(qspace, "Broken QuadratureSpace.");

   const int NE  = qspace->GetMesh()->GetNE();
   for (int e = 0; e < NE; e++)
   {
      const IntegrationRule &ir = qspace->GetElementIntRule(e);
      const int nip = ir.GetNPoints();
      IsoparametricTransformation Tr;
      energy.ParFESpace()->GetElementTransformation(e, &Tr);
      for (int q = 0; q < nip; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         p(e*nip + q) = rho(e*nip + q) * energy.GetValue(Tr, ip);
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
   sol_sock.close();
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
   double mass_0 = Mass(pos_init,  u_init),
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
   CalcDOFBounds(u_init, u_interpolated, pfes_final, pos_final,
                 u_final_min, u_final_max, ELEM_INIT);

   if (visualization)
   {
      ParGridFunction gf_min(&pfes_final), gf_max(&pfes_final);
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

      ot_prob.setWeightedSpaceType(weightedSpace);

      optsolver->SetOptimizationProblem(ot_prob);
      optsolver->SetMaxIter(1e06);
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
      QuadratureSpace qspace_final(&pmesh_final, pfes_final.GetMaxElementOrder()+2);
      std::vector<ParFiniteElementSpace*> fes({&pfes_final});

      Array<int> space_idx({0});
      Array<int> offsets({0, pfes_final.GetTrueVSize()});

      BlockVector x_initial(u_interpolated.GetTrueVector(), offsets);

      ComposedFunctional vol_func(remap::volume_f, remap::volume_df, qspace_final,
                                  fes, space_idx);
      vol_func.SetComm(pmesh_final.GetComm());
      vol_func.SetTarget(mass_0);

      StackedSharedFunctional C(offsets.Last());
      C.AddFunctional(vol_func);

      PointwiseFermiDirac sigmoid(u_final_min, u_final_max);
      Array<LegendreFunction*> legendre_funcs({&sigmoid});

      MassOperator mass(pfes_final);
      Dykstra projector(u_interpolated.ParFESpace()->GetComm(), C, mass,
                        legendre_funcs, offsets,
                        u_final_min, u_final_max, atol, max_iter);
      u_final = u_interpolated;
      projector.Project(u_final);
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

   // Print final objective value.
   const real_t obj_L2 = ObjectiveGF(u_interpolated, u_final_gf);
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
   double mass_0 = Integrate(pos_init,  &u_init,
                             nullptr, nullptr, nullptr),
                   mass_f = Integrate(pos_final, &u_interpolated,
                                      nullptr, nullptr, nullptr);
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
      std::vector<ParFiniteElementSpace*> fes({});

      Array<int> space_idx({-1});
      Array<int> offsets({0, qspace_final.GetSize()});

      BlockVector x_initial(u_interpolated, offsets);

      ComposedFunctional vol_func(remap::volume_f, remap::volume_df, qspace_final,
                                  fes, space_idx);
      vol_func.SetComm(pmesh_final.GetComm());
      vol_func.SetTarget(mass_0);

      StackedSharedFunctional C(offsets.Last());
      C.AddFunctional(vol_func);
      MassOperator mass(qspace_final);
      PointwiseFermiDirac sigmoid(u_min, u_max);
      Array<LegendreFunction*> legendre_funcs({&sigmoid});
      Dykstra projector(pmesh_final.GetComm(), C, mass,
                        legendre_funcs, offsets,
                        u_min, u_max, atol, max_iter);
      u_final = u_interpolated;
      projector.Project(u_final);
   }

   // Report final masses.
   QuadratureFunction u_final_qf(qspace_final);
   u_final_qf = u_final;
   mass_f = Integrate(pos_final, &u_final_qf, nullptr, nullptr, nullptr);
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

   // Print final objective value.
   const real_t obj_l2 = ObjectiveQF(u_interpolated, u_final);
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
   FunctionCoefficient coeff(func);
   ParFiniteElementSpace pfes_init_GL(&pmesh_init, &fec_GL);
   ParGridFunction func_GL(&pfes_init_GL); func_GL.ProjectCoefficient(coeff);
   ParGridFunction func_gf(&pfes_init);    func_gf.ProjectGridFunction(func_GL);
   CalcDOFBounds(func_gf, u_interpolated, pfes_final, pos_final,
                 u_final_min, u_final_max, ELEM_FINAL);
   if (visualization)
   {
      ParGridFunction gf_min(u_interpolated), gf_max(u_interpolated);
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

      ot_prob.setWeightedSpaceType(weightedSpace);
      optsolver->SetOptimizationProblem(ot_prob);

      optsolver->SetMaxIter(max_iter);
      optsolver->SetAbsTol(1e-12);
      optsolver->SetRelTol(1e-14);
      optsolver->SetPrintLevel(3);
      optsolver->Mult(u_interpolated, y_out);

      u_final = y_out;

      delete optsolver;
   }
   else if (opt_type == 2)
   {
      QuadratureSpace qspace_final(&pmesh_final, pfes_final.GetMaxElementOrder()+2);
      std::vector<ParFiniteElementSpace*> fes({&pfes_final});

      Array<int> space_idx({0});
      Array<int> offsets({0, pfes_final.GetTrueVSize()});

      BlockVector x_initial(u_interpolated.GetTrueVector(), offsets);

      ComposedFunctional vol_func(remap::volume_f, remap::volume_df, qspace_final,
                                  fes, space_idx);
      vol_func.SetComm(pmesh_final.GetComm());
      vol_func.SetTarget(mass);

      StackedSharedFunctional C(offsets.Last());
      C.AddFunctional(vol_func);

      MassOperator mass(pfes_final);
      PointwiseFermiDirac sigmoid(u_final_min, u_final_max);
      Array<LegendreFunction*> legendre_funcs({&sigmoid});
      Dykstra projector(u_interpolated.ParFESpace()->GetComm(), C, mass,
                        legendre_funcs, offsets,
                        u_final_min, u_final_max, atol, max_iter);
      u_final = u_interpolated;
      projector.Project(u_final);
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

   // Print final objective value.
   const double obj_L2 = ObjectiveGF(u_interpolated, u_final);
   if (myid == 0)
   {
      std::cout << "---\nObjective L2: " << obj_L2 << std::endl;
   }
}

void InterpolationRemap::RemapHydro(const std::vector<BlockVector>
                                    &ind_rho_e_v_0,
                                    bool remap_v, bool p_control,
                                    const QuadratureFunction &p_0,
                                    std::vector<Array<bool>> &active_el_0,
                                    const Vector &pos_final,
                                    std::vector<BlockVector> &ind_rho_e_v, int opt_type)
{
   const int dim = pmesh_init.Dimension();
   MFEM_VERIFY(dim > 1, "Interpolation remap works only in 2D and 3D.");
   MFEM_VERIFY(qspace && pfes_e && pfes_v, "Spaces are not specified.");

   pmesh_final.SetNodes(pos_final);
   QuadratureSpace qspace_final(pmesh_final, qspace->GetIntRule(0));
   ParFiniteElementSpace pfes_e_final(&pmesh_final, pfes_e->FEColl());
   ParFiniteElementSpace pfes_v_final(&pmesh_final, pfes_v->FEColl(), dim);
   ParFiniteElementSpace pfes_v_scalar_final(&pmesh_final, pfes_v->FEColl());

   // Extract initial data from the BlockVector.
   const int size_qf   = qspace->GetSize(),
             size_gf_e = pfes_e->GetVSize(),
             size_gf_v = pfes_v->GetVSize(),
             size_gf_v_true = pfes_v->GetTrueVSize();

   // Generate list of points where ire_initial will be interpolated.
   Vector pos_quad_final, pos_dof_e_final, pos_dof_v_final;
   GetQuadPositions(qspace_final, pos_final, pos_quad_final);
   GetDOFPositions(pfes_e_final, pos_final, pos_dof_e_final);
   GetDOFPositions(pfes_v_final, pos_final, pos_dof_v_final);

   // Generate the Low-Order-Refined GridFunctions for
   // interpolating the QuadratureFunctions.
   const int order = qspace->GetIntRule(0).GetOrder() / 2;
   const int ref_factor = order + 1;
   ParMesh pmesh_lor = ParMesh::MakeRefined(pmesh_init, ref_factor,
                       BasisType::ClosedGL);
   L2_FECollection fec_lor(0, dim);
   ParFiniteElementSpace pfes_lor(&pmesh_lor, &fec_lor);

   int numBlocks = 4;
   if (remap_v) { numBlocks = 5; }

   Array<int> offset(numBlocks);
   offset[0] = 0;
   offset[1] = offset[0] + size_qf;
   offset[2] = offset[1] + size_qf;
   offset[3] = offset[2] + size_gf_e;
   if (remap_v) { offset[4] = offset[3] + size_gf_v; }

   int ind_cnt = ind_rho_e_v_0.size();

   // Preprocessing: merge all per-material (ind_k, rho_k, e_k[, v_k]) block vectors
   // into a single global block vector for joint optimization.
   // Global block layout: [ind_0, rho_0, e_0, (v_0), ind_1, rho_1, e_1, (v_1), ...]
   const int blocks_per_mat = numBlocks - 1;   // actual blocks per material
   const int mat_size       = offset[numBlocks - 1]; // data size per material

   Array<int> global_offset(ind_cnt * blocks_per_mat + 1);
   global_offset[0] = 0;
   for (int k = 0; k < ind_cnt; k++)
   {
      const int base = k * blocks_per_mat;
      for (int b = 0; b < blocks_per_mat; b++)
      {
         global_offset[base + b + 1] = global_offset[base + b]
                                       + (offset[b + 1] - offset[b]);
      }
   }

   // global_x0: merged initial state (read-only source for interpolation / optimization)
   BlockVector global_x0(global_offset);
   for (int k = 0; k < ind_cnt; k++)
   {
      std::copy(ind_rho_e_v_0[k].GetData(),
                ind_rho_e_v_0[k].GetData() + mat_size,
                global_x0.GetData() + k * mat_size);
   }

   // global_x: merged output state (initialized to global_x0, will hold optimized result)
   BlockVector global_x(global_offset);
   global_x = global_x0;

   // global_x_interp: merged interpolated state; k-th slice is filled inside the loop
   BlockVector global_x_interp(global_offset);

   // global_x_min / global_x_max: merged bounds; k-th slice is filled inside the loop
   BlockVector global_x_min(global_offset);
   BlockVector global_x_max(global_offset);

   Vector volume_0_all(ind_cnt), mass_0_all(ind_cnt), energy_0_all(ind_cnt),
          moment_0_all(ind_cnt*dim), tot_en_0_all(ind_cnt);

   for (int k = 0; k < ind_cnt; k++)
   {

      BlockVector *irev_ptr = const_cast<BlockVector *>(&ind_rho_e_v_0[k]);
      QuadratureFunction ind_0(qspace, irev_ptr->GetData()),
                         rho_0(qspace, irev_ptr->GetData() + size_qf);
      ParGridFunction e_0(pfes_e, irev_ptr->GetData() + 2*size_qf),
                      v_0(pfes_v, irev_ptr->GetData() + 2*size_qf + size_gf_e);

      ParGridFunction ind_0_lor(&pfes_lor), rho_0_lor(&pfes_lor);
      MFEM_VERIFY(ind_0.Size() == ind_0_lor.Size(), "Size mismatch ind LOR.");
      MFEM_VERIFY(rho_0.Size() == rho_0_lor.Size(), "Size mismatch rho LOR.");
      ind_0_lor = ind_0;
      rho_0_lor = rho_0;
      // Pressure function (not part of the solution state).
      ParGridFunction p_0_lor;
      if (p_control)
      {
         p_0_lor.SetSpace(&pfes_lor);
         MFEM_VERIFY(p_0.Size() == p_0_lor.Size(), "Size mismatch p LOR.");
         p_0_lor = p_0;
      }

      // Visualize the initial LOR GridFunctions.
      if (visualization)
      {
         socketstream sock_ind, sock_rho;
         VisualizeField(sock_ind, "localhost", 19916, ind_0_lor, "ind_0 LOR",
                        0, 500, 400, 400);
         VisualizeField(sock_rho, "localhost", 19916, rho_0_lor, "rho_0 LOR",
                        400, 500, 400, 400);
         if (p_control)
         {
            socketstream sock_p;
            VisualizeField(sock_p, "localhost", 19916, p_0_lor, "p_0 LOR",
                           800, 500, 400, 400);
         }
      }

      // Interpolate into the k-th slice of global_x_interp.
      BlockVector ind_rho_e_v_interp(global_x_interp.GetData() + k * mat_size,
                                     offset);
      real_t *irev_data = ind_rho_e_v_interp.GetData();
      QuadratureFunction ind_interp(&qspace_final, irev_data),
                         rho_interp(&qspace_final, irev_data + size_qf);
      QuadratureFunction p_interp(&qspace_final);
      ParGridFunction e_interp(&pfes_e_final, irev_data + 2*size_qf),
                      v_interp(&pfes_v_final, irev_data + 2*size_qf + size_gf_e);
      FindPointsGSLIB finder(pmesh_init.GetComm());
      finder.SetL2AvgType(FindPointsGSLIB::NONE);
      finder.Setup(pmesh_lor);
      finder.Interpolate(pos_quad_final, ind_0_lor, ind_interp);
      finder.Interpolate(pos_quad_final, rho_0_lor, rho_interp);
      if (p_control) { finder.Interpolate(pos_quad_final, p_0_lor, p_interp); }
      finder.Setup(pmesh_init);
      finder.SetL2AvgType(FindPointsGSLIB::NONE);
      finder.Interpolate(pos_dof_e_final, e_0, e_interp);
      Vector v_interp_vals(pos_dof_v_final.Size());
      finder.Interpolate(pos_dof_v_final, v_0, v_interp_vals);
      {
         Array<int> vdofs;
         const int nsp = pfes_v_final.GetFE(0)->GetNodes().GetNPoints();
         const int NE  = pfes_v_final.GetNE();
         Vector elem_dof_vals(nsp*dim);

         for (int e = 0; e < NE; e++)
         {
            for (int j = 0; j < nsp; j++)
            {
               for (int d = 0; d < dim; d++)
               {
                  int idx = d*nsp*NE + e*nsp + j;
                  elem_dof_vals(d*nsp + j) = v_interp_vals(idx);
               }
            }
            pfes_v_final.GetElementVDofs(e, vdofs);
            v_interp.SetSubVector(vdofs, elem_dof_vals);
         }
      }
      finder.FreeData();

      if (p_control)
      {
         VisQuadratureFunction(pmesh_final, p_interp, "p QF interpolated", 0, 0);
      }

      // Report conservation errors of ire_final.
      const double volume_0 = Integrate(pos_init,
                                        &ind_0, nullptr, nullptr, nullptr);
      const double volume_f = Integrate(pos_final,
                                        &ind_interp, nullptr, nullptr, nullptr);
      const double mass_0   = Integrate(pos_init,
                                        &ind_0, &rho_0, nullptr, nullptr);
      const double mass_f   = Integrate(pos_final,
                                        &ind_interp, &rho_interp, nullptr, nullptr);
      const double energy_0 = Integrate(pos_init,
                                        &ind_0, &rho_0, &e_0, nullptr);
      const double energy_f = Integrate(pos_final,
                                        &ind_interp, &rho_interp, &e_interp, nullptr);
      volume_0_all(k) = volume_0;
      mass_0_all(k) = mass_0;
      energy_0_all(k) = energy_0;
      Vector moment_0(dim), moment_f(dim);
      for (int d = 0; d < dim; d++)
      {
         moment_0(d) = Integrate(pos_init,
                                 &ind_0, &rho_0, nullptr, &v_0, d);
         moment_f(d) = Integrate(pos_final,
                                 &ind_interp, &rho_interp, nullptr, &v_interp, d);
         moment_0_all(k*dim + d) = moment_0(d);
      }
      const double tot_en_0 = Integrate(pos_init,
                                        &ind_0, &rho_0, &e_0, &v_0);
      const double tot_en_f = Integrate(pos_final,
                                        &ind_interp, &rho_interp, &e_interp, &v_interp);
      tot_en_0_all(k) = tot_en_0;

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
                   << "Mass interpolated diff %:   "
                   << fabs(mass_0 - mass_f) / mass_0 * 100
                   << endl << "*\n"
                   << "Intern energy initial:      " << energy_0 << std::endl
                   << "Intern energy interp:       " << energy_f << std::endl
                   << "Intern energy interp diff:  "
                   << fabs(energy_0 - energy_f) << endl
                   << "Intern energy interp diff %:"
                   << fabs(energy_0 - energy_f) / energy_0 * 100
                   << endl;
         if (remap_v)
         {
            for (int d = 0; d < dim; d++)
            {
               cout << "*\n"
                    << "Momentum " << d << " initial:       " << moment_0(d) << std::endl
                    << "Momentum " << d << " interp:        " << moment_f(d) << std::endl
                    << "Momentum " << d << " interp diff:   "
                    << fabs(moment_0(d) - moment_f(d)) << endl
                    << "Momentum " << d << " interp diff %: "
                    << fabs(moment_0(d) - moment_f(d)) / moment_0(d) * 100
                    << endl;
            }
            cout <<   "*\n"
                 << "Total energy initial:             " << tot_en_0 << std::endl
                 << "Total energy interpolated:        " << tot_en_f << std::endl
                 << "Total energy interpolated diff:   "
                 << fabs(tot_en_0 - tot_en_f) << endl
                 << "Total energy interpolated diff %: "
                 << fabs(tot_en_0 - tot_en_f) / tot_en_0 * 100
                 << endl;
         }
      }

      // Compute min / max bounds.
      // Also adjust interpolated values in some special cases.
      Vector ind_min, ind_max;
      CalcQuadBounds(ind_0, ind_interp, pos_final, ind_min, ind_max, ELEM_FINAL);
      CleanEmptyZones(ind_interp, ind_min, ind_max);
      Vector rho_min, rho_max;
      CalcRhoBounds(rho_interp, ind_interp, ind_max, rho_min, rho_max);
      UpdateRhoInterp(rho_interp, rho_min, rho_max);

      Vector e_min, e_max;
      if (p_control)
      {
         CalcEBounds(e_0, active_el_0[k], e_interp, pos_final, ind_max,
                     e_min, e_max, ELEM_INIT);
      }
      else
      {
         CalcEBounds(e_0, active_el_0[k], e_interp, pos_final, ind_max,
                     e_min, e_max, ELEM_FINAL);
      }
      UpdateEInterp(e_interp, e_min, e_max);

      if (p_control)
      {
         rho_min -= 1e-2;
         rho_max += 1e-2;
      }

      Vector v_min, v_max;
      if (remap_v)
      {
         CalcVBounds(v_interp, v_min, v_max);
      }

      BlockVector x_min(global_x_min.GetData() + k * mat_size, offset);
      BlockVector x_max(global_x_max.GetData() + k * mat_size, offset);

      x_min.GetBlock(0) = ind_min;
      x_min.GetBlock(1) = rho_min;
      x_min.GetBlock(2) = e_min;
      x_max.GetBlock(0) = ind_max;
      x_max.GetBlock(1) = rho_max;
      x_max.GetBlock(2) = e_max;

      if (remap_v)
      {
         x_min.GetBlock(3) = v_min;
         x_max.GetBlock(3) = v_max;
      }

      // Optimize.
      if (opt_type == 0)
      {
         ind_rho_e_v[k] = ind_rho_e_v_interp;
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
         Array<int> offset_true(numBlocks);
         offset_true[0] = 0;
         offset_true[1] = offset_true[0] + size_qf;
         offset_true[2] = offset_true[1] + size_qf;
         offset_true[3] = offset_true[2] + size_gf_e;
         if (remap_v)
         {
            offset_true[4] = offset_true[3] + size_gf_v_true;
         }

         Vector rho_target, e_target, v_target;
         GetTargetValues( rho_interp, rho_min, rho_max, rho_target );
         GetTargetValues( e_interp, e_min, e_max, e_target );

         BlockVector initial_design(offset_true);
         BlockVector design_min    (offset_true);
         BlockVector design_max    (offset_true);
         initial_design.GetBlock(0) = ind_interp;
         initial_design.GetBlock(1) = rho_target;
         initial_design.GetBlock(2) = e_target;
         design_min.GetBlock(0) = ind_min;
         design_min.GetBlock(1) = rho_min;
         design_min.GetBlock(2) = e_min;
         design_max.GetBlock(0) = ind_max;
         design_max.GetBlock(1) = rho_max;
         design_max.GetBlock(2) = e_max;
         if (remap_v)
         {
            ParGridFunction vtmp_min(&pfes_v_final, v_min);
            ParGridFunction vtmp_max(&pfes_v_final, v_max);

            v_interp.SetTrueVector();
            vtmp_min.SetTrueVector();
            vtmp_max.SetTrueVector();

            mfem::Vector & true_v_interp = v_interp.GetTrueVector();
            mfem::Vector & true_v_min    = vtmp_min.GetTrueVector();
            mfem::Vector & true_v_max    = vtmp_max.GetTrueVector();

            initial_design.GetBlock(3) = true_v_interp;
            design_min    .GetBlock(3) = true_v_min;
            design_max    .GetBlock(3) = true_v_max;
         }

         int NumDesVar = initial_design.Size();
         BlockVector y_out(offset_true);

         y_out = initial_design;

         mfem::Array<int> optProbInd;
         mfem::Vector ind_rho_e_sub;
         mfem::Vector y_out_sub;
         mfem::Vector minsub;
         mfem::Vector maxsub;

         Vector x_maxsub(NumDesVar), x_minsub(NumDesVar);

         if (subprob)
         {
            NumDesVar = GetSizeOptimizationSubset(design_min,design_max);
            GetOptimizationSubsetInd(design_min,design_max,optProbInd);

            initial_design.GetSubVector(optProbInd,ind_rho_e_sub);
            y_out.GetSubVector(optProbInd,y_out_sub);

            design_min.GetSubVector(optProbInd,minsub);
            design_max.GetSubVector(optProbInd,maxsub);

            x_maxsub.SetSize(NumDesVar);
            x_minsub.SetSize(NumDesVar);
            x_maxsub = maxsub;
            x_minsub = minsub;
         }
         else
         {
            x_maxsub = design_max;
            x_minsub = design_min;
         }

         OptimizationProblem *ot_prob = nullptr;

         if (remap_v)
         {
            MFEM_VERIFY(p_control == false, "Remap v + p not implemented.");
            ot_prob = new RemhosHydroHiOpProblem(qspace_final,
                                                 pfes_e_final,
                                                 pfes_v_final,
                                                 pos_final,
                                                 initial_design,
                                                 NumDesVar,
                                                 x_minsub, x_maxsub,
                                                 volume_0, mass_0, moment_0, tot_en_0,
                                                 5, false, optProbInd, true, subprob);

            dynamic_cast<RemhosHydroHiOpProblem*>(ot_prob)->setWeightedSpaceType(
               weightedSpace);
         }
         else
         {
            ot_prob = new RemhosIndRhoEHiOpProblem(qspace_final,
                                                   pfes_e_final,
                                                   pos_final,
                                                   initial_design,
                                                   p_interp,
                                                   NumDesVar,
                                                   x_minsub, x_maxsub,
                                                   volume_0, mass_0, energy_0,
                                                   3, false, optProbInd, true,
                                                   subprob, p_control);

            dynamic_cast<RemhosIndRhoEHiOpProblem*>(ot_prob)->setWeightedSpaceType(
               weightedSpace);
         }

         optsolver->SetOptimizationProblem(*ot_prob);
         optsolver->SetMaxIter(max_iter);
         optsolver->SetAbsTol(1e-7);
         optsolver->SetRelTol(1e-7);
         optsolver->SetPrintLevel(3);

         if (subprob)
         {
            optsolver->Mult(ind_rho_e_sub, y_out_sub);
            y_out.SetSubVector(optProbInd, y_out_sub);
         }
         else { optsolver->Mult(initial_design, y_out); }

         BlockVector T_vector_design(offset_true);
         BlockVector L_vector_design(offset);

         T_vector_design = y_out;

         {
            QuadratureFunction rho_opt(&qspace_final,
                                       T_vector_design.GetBlock(1).GetData());
            QuadratureFunction pressure_opt(&qspace_final); pressure_opt = 0.0;
            ParGridFunction    e_opt  (&pfes_e_final,
                                       T_vector_design.GetBlock(2).GetData());
            ComputePressure( pos_final, rho_opt, e_opt, pressure_opt);

            ParaViewDataCollection pvdc("IndRhoE_pressure_opt", &pmesh_final);
            pvdc.SetDataFormat(VTKFormat::BINARY32);
            pvdc.SetCycle(0);
            pvdc.SetTime(1.0);
            pvdc.RegisterQField("rho", &rho_opt);
            pvdc.RegisterQField("pressure", &pressure_opt);
            pvdc.RegisterQField("pressure_interp", &p_interp);
            pvdc.Save();

            ParaViewDataCollection pvdc1("IndRhoE_pressure_opt1", &pmesh_final);
            pvdc1.SetDataFormat(VTKFormat::BINARY32);
            pvdc1.SetCycle(0);
            pvdc1.SetTime(1.0);

            pvdc1.RegisterField("e", &e_opt);
            pvdc1.Save();
         }


         L_vector_design.GetBlock(0) = T_vector_design.GetBlock(0);
         L_vector_design.GetBlock(1) = T_vector_design.GetBlock(1);
         L_vector_design.GetBlock(2) = T_vector_design.GetBlock(2);

         if (remap_v)
         {
            ParGridFunction vel_final(&pfes_v_final);
            Vector vel_true(T_vector_design.GetData() + 2*size_qf + size_gf_e,
                            size_gf_v_true);
            vel_final.SetFromTrueDofs(vel_true);
            L_vector_design.GetBlock(3) = vel_final;
         }

         ind_rho_e_v[k] = L_vector_design;

         delete optsolver;
         delete ot_prob;
      }
      else if (opt_type == 2)
      {
         // it will be optimized outside the loop after setting up the full problem with all materials
      }
      else { MFEM_ABORT("not implemented!"); }
   }
   if (opt_type == 2)
   {
      std::vector<ParFiniteElementSpace*> fes({&pfes_e_final, &pfes_v_scalar_final});

      // Blocks per material: [ind, rho, e, v_0, ..., v_{dim-1}].
      // Velocity is per-material so that momentum_f/energy_f can use shift_f.
      const int num_vars = 3 + dim * (int)remap_v;
      Array<int> space_idx(num_vars * ind_cnt);
      for (int k = 0; k < ind_cnt; k++)
      {
         const int base = k * num_vars;
         space_idx[base + 0] = -1; // ind - qf
         space_idx[base + 1] = -1; // rho - qf
         space_idx[base + 2] = 0;  // e - l2
         for (int d = 0; d < dim * (int)remap_v; d++)
         {
            space_idx[base + 3 + d] = 1; // v_d - h1
         }
      }

      MFEM_VERIFY(dynamic_cast<const L2_FECollection*>(pfes_e_final.FEColl()) !=
                  nullptr,
                  "Expecting L2_FECollection for pfes_e_final.");

      // Per-material T-vector offsets (sizes first, then PartialSum).
      // For QF and L2, TrueVSize == VSize.
      Array<int> space_size({size_qf, size_qf, size_gf_e});
      for (int d = 0; d < dim * (int)remap_v; d++)
      {
         space_size.Append(size_gf_v_true/dim);
      }

      // Global T-vector offsets: tile the per-material offsets ind_cnt times.
      Array<int> global_t_offsets(ind_cnt * num_vars + 1);
      global_t_offsets[0] = 0;
      for (int k = 0; k < ind_cnt; k++)
      {
         global_t_offsets.Append(space_size);
      }
      global_t_offsets.PartialSum();
      Array<int> offsets({0});
      offsets.Append(space_size);
      offsets.PartialSum();
      const int per_mat_size = offsets.Last();

      BlockVector x_initial(global_t_offsets);
      BlockVector x_min_final(global_t_offsets);
      BlockVector x_max_final(global_t_offsets);

      // Fill x_initial, x_min_final, x_max_final from per-material L-vectors.
      // For QF and L2 (ind, rho, e): L-vector == T-vector, direct copy.
      // For H1 (velocity): GetTrueDofs for L->T conversion.
      for (int k = 0; k < ind_cnt; k++)
      {
         BlockVector x_init_k(global_x_interp.GetData() + k * mat_size, offset);
         BlockVector x_min_k(global_x_min.GetData() + k * mat_size, offset);
         BlockVector x_max_k(global_x_max.GetData() + k * mat_size, offset);

         for (int i = 0; i < 3; i++)
         {
            x_initial.GetBlock(k * num_vars + i)   = x_init_k.GetBlock(i);
            x_min_final.GetBlock(k * num_vars + i) = x_min_k.GetBlock(i);
            x_max_final.GetBlock(k * num_vars + i) = x_max_k.GetBlock(i);
         }

         if (remap_v)
         {
            ParGridFunction vtmp(&pfes_v_scalar_final, (real_t*)nullptr);
            const int n = pfes_v_scalar_final.GetVSize();
            MFEM_VERIFY(n * dim == pfes_v_final.GetVSize(),
                        "Expecting dim*n dofs for pfes_v_scalar_final.");
            for (int d = 0; d < dim; d++)
            {
               vtmp.MakeRef(&pfes_v_scalar_final, x_init_k.GetBlock(3), d * n);
               vtmp.GetTrueDofs(x_initial.GetBlock(k * num_vars + 3 + d));

               vtmp.MakeRef(&pfes_v_scalar_final, x_min_k.GetBlock(3), d * n);
               vtmp.GetTrueDofs(x_min_final.GetBlock(k * num_vars + 3 + d));

               vtmp.MakeRef(&pfes_v_scalar_final, x_max_k.GetBlock(3), d * n);
               vtmp.GetTrueDofs(x_max_final.GetBlock(k * num_vars + 3 + d));
            }
         }
      }

      // Objective function: 0.5 * || u - u_initial ||^2
      remap::RemapObjectiveFunctional remap_obj(qspace_final, fes, x_initial,
            space_idx);

      // Constraint functionals: one set per material.
      // funcs_per_mat = volume + mass + energy/potential + dim*momentum (if remap_v)
      const int funcs_per_mat = 3 + dim * (int)remap_v;
      std::vector<std::unique_ptr<ComposedFunctional>> funcs(funcs_per_mat * ind_cnt);

      // shift_f / shift_df: wrap a single-material function so that it
      // extracts the k-th material's slice from the global vector before
      // forwarding to the original function.
      auto shift_f = [num_vars](
                        std::function<real_t(const Vector &)> f, int material_idx)
      {
         return [f, material_idx, num_vars](const Vector &x) -> real_t
         {
            const Vector x_k(x.GetData() + material_idx * num_vars, num_vars);
            return f(x_k);
         };
      };
      auto shift_df = [num_vars](
                         std::function<void(const Vector &, Vector &)> df, int material_idx)
      {
         return [df, material_idx, num_vars](const Vector &x, Vector &y) -> void
         {
            y = 0.0;
            const Vector x_k(x.GetData() + material_idx * num_vars, num_vars);
            Vector y_k(y.GetData() + material_idx * num_vars, num_vars);
            df(x_k, y_k);
         };
      };
      // if (Mpi::Root())
      // {
      //    volume_0_all.Print();
      //    mass_0_all.Print();
      //    energy_0_all.Print();
      //    moment_0_all.Print();
      //    tot_en_0_all.Print();
      // }

      for (int k = 0; k < ind_cnt; k++)
      {
         funcs[funcs_per_mat * k + 0] = std::make_unique<ComposedFunctional>(
                                           shift_f(remap::volume_f, k),
                                           shift_df(remap::volume_df, k),
                                           qspace_final, fes, space_idx);
         funcs[funcs_per_mat * k + 0]->SetTarget(volume_0_all[k]);
         funcs[funcs_per_mat * k + 1] = std::make_unique<ComposedFunctional>(
                                           shift_f(remap::mass_f, k),
                                           shift_df(remap::mass_df, k),
                                           qspace_final, fes, space_idx);
         funcs[funcs_per_mat * k + 1]->SetTarget(mass_0_all[k]);
         if (!remap_v) // use potential
         {
            funcs[funcs_per_mat * k + 2] = std::make_unique<ComposedFunctional>(
                                              shift_f(remap::potential_f, k),
                                              shift_df(remap::potential_df, k),
                                              qspace_final, fes, space_idx);
            funcs[funcs_per_mat * k + 2]->SetTarget(energy_0_all[k]);
         }
         else
         {
            for (int i = 0; i < dim; i++)
            {
               funcs[funcs_per_mat * k + 3 + i] = std::make_unique<ComposedFunctional>(
               shift_f([i](const Vector &x) { return remap::momentum_f(x, i); }, k),
               shift_df([i](const Vector &x, Vector &g) { remap::momentum_df(x, g, i); }, k),
               qspace_final, fes, space_idx);
               funcs[funcs_per_mat * k + 3 + i]->SetTarget(moment_0_all[dim * k + i]);
            }
            funcs[funcs_per_mat * k + 2] = std::make_unique<ComposedFunctional>(
                                              shift_f(remap::energy_f, k),
                                              shift_df(remap::energy_df, k),
                                              qspace_final, fes, space_idx);
            funcs[funcs_per_mat * k + 2]->SetTarget(tot_en_0_all[k]);
         }
      }

      StackedSharedFunctional C(ind_cnt * per_mat_size);
      for (auto &f : funcs)
      {
         f->SetComm(pmesh_final.GetComm());
         C.AddFunctional(*f);
      }

      MassOperator mass_q(qspace_final), mass_l2(pfes_e_final),
                   mass_h1(pfes_v_scalar_final);
      MultiMassOperator mass;
      for (int k = 0; k < ind_cnt; k++)
      {
         mass.Append(mass_q);
         mass.Append(mass_q);
         mass.Append(mass_l2);
         if (remap_v) { for (int d = 0; d < dim; d++) { mass.Append(mass_h1); } }
      }

      PointwiseFermiDirac sigmoid(x_min_final, x_max_final);
      Array<LegendreFunction*> legendre_funcs({&sigmoid});
      Array<int> dummy_offset({0, x_min_final.Size()});
      Dykstra projector(pmesh_final.GetComm(), C, mass,
                        legendre_funcs, dummy_offset,
                        x_min_final, x_max_final, atol, max_iter);
      projector.Project(x_initial);

      // Write back optimized values to ind_rho_e_v for each material.
      // For QF and L2 (ind, rho, e): T-vector == L-vector, direct copy.
      // For H1 (velocity): SetFromTrueDofs for T->L conversion.
      for (int k = 0; k < ind_cnt; k++)
      {
         BlockVector result_L(ind_rho_e_v[k].GetData(), offset);
         for (int i = 0; i < 3; i++)
         {
            result_L.GetBlock(i) = x_initial.GetBlock(k * num_vars + i);
         }
         if (remap_v)
         {
            ParGridFunction vtmp(&pfes_v_final, result_L.GetBlock(3));
            const int v_true_size = per_mat_size - offsets[3];
            Vector v_true_k(x_initial.GetData() + k * per_mat_size + offsets[3],
                            v_true_size);
            vtmp.SetFromTrueDofs(v_true_k);
         }
      }
   }


   for (int k = 0; k < ind_cnt; k++)
   {
      BlockVector ind_rho_e_v_interp(global_x_interp.GetData() + k * mat_size,
                                     offset);
      real_t *irev_data = ind_rho_e_v_interp.GetData();
      QuadratureFunction ind_interp(&qspace_final, irev_data),
                         rho_interp(&qspace_final, irev_data + size_qf);
      QuadratureFunction p_interp(&qspace_final);
      ParGridFunction e_interp(&pfes_e_final, irev_data + 2*size_qf),
                      v_interp(&pfes_v_final, irev_data + 2*size_qf + size_gf_e);

      QuadratureFunction ind(&qspace_final, ind_rho_e_v[k].GetData()),
                         rho(&qspace_final, ind_rho_e_v[k].GetData() + size_qf);
      ParGridFunction e(&pfes_e_final, ind_rho_e_v[k].GetData() + 2*size_qf);
      ParGridFunction v(&pfes_v_final,
                        ind_rho_e_v[k].GetData() + 2*size_qf + size_gf_e);

      // Print conservation errors.
      const double volume_f_opt = Integrate(pos_final, &ind, nullptr, nullptr,
                                            nullptr);
      const double mass_f_opt   = Integrate(pos_final, &ind, &rho,    nullptr,
                                            nullptr);
      const double energy_f_opt = Integrate(pos_final, &ind, &rho,    &e, nullptr);
      Vector moment_f_opt(dim);
      for (int d = 0; d < dim; d++)
      {
         moment_f_opt(d) = Integrate(pos_final, &ind, &rho, nullptr, &v, d);
      }
      const double tot_energy_f_opt = Integrate(pos_final, &ind, &rho, &e, &v);
      if (Mpi::Root())
      {
         std::cout << "-------\n"
                   << "Volume initial:          " << volume_0_all[k] << std::endl
                   << "Volume optimized:        " << volume_f_opt << std::endl
                   << "Volume optimized diff:   "
                   << (volume_f_opt - volume_0_all[k]) << endl
                   << "Volume optimized diff %: "
                   << (volume_f_opt - volume_0_all[k]) / volume_0_all[k] * 100
                   << endl << "*\n"
                   << "Mass initial:            " << mass_0_all[k] << std::endl
                   << "Mass optimized:          " << mass_f_opt << std::endl
                   << "Mass optimized diff:     "
                   << (mass_f_opt - mass_0_all[k]) << endl
                   << "Mass optimized diff %:   "
                   << (mass_f_opt - mass_0_all[k]) / mass_0_all[k] * 100
                   << endl << "*\n";

         if (remap_v)
         {
            for (int d = 0; d < dim; d++)
            {
               std::cout << "Moment in dim "<<d+1 <<" initial:            " <<
                         moment_0_all[k*dim + d] <<
                         std::endl
                         << "Moment in dim "<<d+1 <<" optimized:          " << moment_f_opt(
                            d) << std::endl
                         << "Moment in dim "<<d+1 <<" optimized diff:     "
                         << (moment_f_opt(d) - moment_0_all[k*dim + d]) << endl
                         << "Moment in dim "<<d+1 <<" optimized diff %:   "
                         << (moment_f_opt(d) - moment_0_all[k*dim + d]) / moment_0_all[k*dim + d] * 100
                         << endl << "*\n";
            }

            std::cout<< "Total energy initial:          " << tot_en_0_all[k] << std::endl
                     << "Total energy optimized:        " << tot_energy_f_opt << std::endl
                     << "Total energy optimized diff:   "
                     << (tot_energy_f_opt- tot_en_0_all[k]) << endl
                     << "Total energy optimized diff %: "
                     << (tot_energy_f_opt- tot_en_0_all[k]) / tot_en_0_all[k] * 100
                     << endl;
         }
         else
         {
            std::cout << "Energy initial:          " << energy_0_all[k] << std::endl
                      << "Energy optimized:        " << energy_f_opt << std::endl
                      << "Energy optimized diff:   "
                      << (energy_f_opt- energy_0_all[k]) << endl
                      << "Energy optimized diff %: "
                      << (energy_f_opt- energy_0_all[k]) / energy_0_all[k] * 100
                      << endl;
         }
      }

      // Check for bounds violations.
      if (Mpi::Root()) { std::cout << "-------\nIndicator violations: \n"; }
      CheckBounds(pmesh_init.GetMyRank(), ind, global_x_min.GetBlock(k*ind_cnt),
                  global_x_max.GetBlock(k*ind_cnt));
      if (Mpi::Root()) { std::cout << "*\nDensity violations: \n"; }
      CheckBounds(pmesh_init.GetMyRank(), rho, global_x_min.GetBlock(k*ind_cnt + 1),
                  global_x_min.GetBlock(k*ind_cnt + 1));
      if (Mpi::Root()) { std::cout << "*\nInternal Energy violations: \n"; }
      CheckBounds(pmesh_init.GetMyRank(), e, global_x_min.GetBlock(k*ind_cnt + 2),
                  global_x_min.GetBlock(k*ind_cnt + 2));
      if (remap_v)
      {
         if (Mpi::Root()) { std::cout << "*\nVelocity violations: \n"; }
         CheckBounds(pmesh_init.GetMyRank(), v, global_x_min.GetBlock(k*ind_cnt + 3),
                     global_x_min.GetBlock(k*ind_cnt + 3));
      }

      // Print final objective values.
      double ind_obj_l2 = ObjectiveQF(ind_interp, ind),
             rho_obj_l2 = ObjectiveQF(rho_interp, rho),
             e_obj_L2 = ObjectiveGF(e_interp, e),
             v_obj_L2;
      if (remap_v)
      {
         v_obj_L2 = ObjectiveVecGF(v_interp, v);
      }

      if (myid == 0)
      {
         std::cout << "---\nObjective ind l2: " << ind_obj_l2 << std::endl;
         std::cout <<      "Objective rho l2: " << rho_obj_l2 << std::endl;
         std::cout <<      "Objective e   L2: " << e_obj_L2 << std::endl;
         if (remap_v)
         {
            std::cout <<      "Objective v   L2: " << v_obj_L2 << std::endl;
         }
      }
   }
}

void InterpolationRemap::GetDOFPositions(const ParFiniteElementSpace &pfes,
      const Vector &pos_mesh, Vector &pos_dofs)
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

real_t InterpolationRemap::ObjectiveGF(const ParGridFunction &g_interp,
                                       const ParGridFunction &g)
{
   GridFunctionCoefficient ci(&g_interp);
   return g.ComputeL2Error(ci);
}

real_t InterpolationRemap::ObjectiveVecGF(const ParGridFunction &g_interp,
      const ParGridFunction &g)
{
   VectorGridFunctionCoefficient ci(&g_interp);
   return g.ComputeL2Error(ci);
}

real_t InterpolationRemap::ObjectiveQF(const Vector &g_interp,
                                       const Vector &g)
{
   Vector tmp(g_interp);
   tmp -= g;
   real_t obj_l2 = pow(tmp.Norml2(), 2.0);
   MPI_Allreduce(MPI_IN_PLACE, &obj_l2, 1, MFEM_MPI_REAL_T, MPI_SUM,
                 MPI_COMM_WORLD);
   return sqrt(obj_l2);
}

double InterpolationRemap::Integrate(const Vector &pos,
                                     const QuadratureFunction *ind,
                                     const QuadratureFunction *rho,
                                     const ParGridFunction *e,
                                     const ParGridFunction *v, int comp)
{
   MFEM_VERIFY(ind || rho || e, "At least one function must be specified.");

   const QuadratureSpace *qspace = nullptr;
   if (ind) { qspace = dynamic_cast<const QuadratureSpace *>(ind->GetSpace()); }
   if (rho) { qspace = dynamic_cast<const QuadratureSpace *>(rho->GetSpace()); }

   auto mesh = (qspace) ? qspace->GetMesh() : e->ParFESpace()->GetMesh();
   const int NE = mesh->GetNE(), dim = mesh->Dimension();
   double integral = 0.0;
   for (int j = 0; j < NE; j++)
   {
      const IntegrationRule &ir =
         (qspace) ? qspace->GetElementIntRule(j)
         : IntRules.Get(e->ParFESpace()->GetFE(j)->GetGeomType(), 7);
      const int nqp = ir.GetNPoints();

      // Transformation w.r.t. the given mesh positions.
      IsoparametricTransformation Tr;
      mesh->GetElementTransformation(j, pos, &Tr);

      Vector ind_vals(nqp), rho_vals(nqp), e_vals(nqp);
      DenseMatrix v_vals(dim, nqp);
      if (ind) { ind->GetValues(j, ind_vals); }
      else { ind_vals = 1.0; }
      if (rho) { rho->GetValues(j, rho_vals); }
      else { rho_vals = 1.0; }
      if (e) { e->GetValues(Tr, ir, e_vals); }
      else { e_vals = 1.0; }
      if (v) { v->GetVectorValues(Tr, ir, v_vals); }
      else { v_vals = 0.0; }

      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         real_t vv = 0.0;
         for (int d = 0; d < dim; d++) { vv += v_vals(d, q) * v_vals(d, q); }
         if (v != nullptr && e == nullptr)
         {
            // Momentum case.
            integral += Tr.Weight() * ip.weight *
                        ind_vals(q) * rho_vals(q) * v_vals(comp, q);
         }
         else
         {
            // Volume / mass / internal energy / total energy cases.
            integral += Tr.Weight() * ip.weight *
                        (ind_vals(q) * rho_vals(q) * e_vals(q) +
                         0.5 * ind_vals(q) * rho_vals(q) * vv);
         }
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &integral, 1, MPI_DOUBLE, MPI_SUM,
                 pmesh_init.GetComm());
   return integral;
}

#define EMPTY_VALUE -1.0

void InterpolationRemap::CalcDOFBounds(const ParGridFunction &g_init,
                                       const ParGridFunction &gf_interp,
                                       const ParFiniteElementSpace &pfes,
                                       const Vector &pos_final,
                                       Vector &g_min, Vector &g_max,
                                       BoundsType bounds_type)
{
   const int size_res = pfes.GetVSize(), NE = pmesh_init.GetNE();
   g_min.SetSize(size_res);
   g_max.SetSize(size_res);
   g_min = gf_interp;
   g_max = gf_interp;

   if (bounds_type == ELEM_INIT || bounds_type == ELEM_BOTH)
   {
      // Form the min and max functions on every MPI task.
      // All on the initial mesh.
      L2_FECollection fec_L2(0, pmesh_init.Dimension());
      ParFiniteElementSpace pfes_L2(&pmesh_init, &fec_L2);
      ParGridFunction g_el_min(&pfes_L2), g_el_max(&pfes_L2);
      for (int e = 0; e < NE; e++)
      {
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
   }

   // On the new mesh, take min/max over DOFs in the same element.
   if (bounds_type == ELEM_FINAL || bounds_type == ELEM_BOTH)
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

void InterpolationRemap::CleanEmptyZones(QuadratureFunction &ind_interp,
      Vector &ind_min, Vector &ind_max)
{
   const double eps = 1e-12;
   const int s = ind_interp.Size();
   for (int q = 0; q < s; q++)
   {
      if (ind_max(q) < eps)
      {
         ind_interp(q) = 0.0;
         ind_min(q) = 0.0;
         ind_max(q) = 0.0;
      }
   }
}

void InterpolationRemap::CalcRhoBounds(const QuadratureFunction &rho_interp,
                                       const QuadratureFunction &ind_interp,
                                       const Vector &ind_max,
                                       Vector &rho_min, Vector &rho_max)
{
   const double eps = 1e-12;

   const int size_rho = rho_interp.Size(), NE = pmesh_init.GetNE();
   rho_min.SetSize(size_rho); rho_max.SetSize(size_rho);

   int el_e_idx = 0;
   for (int e = 0; e < NE; e++)
   {
      const IntegrationRule &ir = qspace->GetElementIntRule(e);
      const int nqp = ir.GetNPoints();

      // Compute min and max density in the new mesh element.
      double el_min =   std::numeric_limits<double>::infinity(),
             el_max = - std::numeric_limits<double>::infinity();
      bool el_has_ind_value = false, el_has_ind_max = false;
      for (int q = 0; q < nqp; q++)
      {
         // Bounds are taken only from points where the material was present
         // initially. This is achieved by checking the interpolated values,
         // which come from the values on the initial mesh.
         if (ind_interp(el_e_idx + q) > eps)
         {
            el_has_ind_value = true;
            el_min = std::min(el_min, rho_interp(el_e_idx + q));
            el_max = std::max(el_max, rho_interp(el_e_idx + q));
         }
         if (ind_max(el_e_idx + q) > eps) { el_has_ind_max = true; }
      }

      // The new mesh element is completely empty -> we want zeros in it.
      if (el_has_ind_value == false)
      {
         MFEM_VERIFY(el_has_ind_max == false, "Indicator values / bound mess!");
         el_min = el_max = 0.0;
      }

      // Set the bounds.
      for (int q = 0; q < nqp; q++)
      {
         // Bounds are set only where the material will be present. This is
         // achieved by checking the ind_max values. This enables the
         // propagation of density to newly active points in the new mesh.
         // We can get subzonal behavior for the density when there is
         // local variation in ind_max.
         if (ind_max(el_e_idx + q) > 1e-12)
         {
            rho_min(el_e_idx + q) = el_min;
            rho_max(el_e_idx + q) = el_max;

            // Note that it's fine to be lower than the mininum, i.e., in
            // elements where the ind function propages due to small diffusion.
            MFEM_VERIFY(rho_interp(el_e_idx + q) < rho_max(el_e_idx + q) + eps,
                        "Error: interpolated density is above upper bound: "
                        << rho_interp(el_e_idx + q) << " "
                        << rho_max(el_e_idx + q));
         }
         else
         {
            // No material at the DOF.
            rho_min(el_e_idx + q) = 0.0;
            rho_max(el_e_idx + q) = 0.0;

            // No material, but has density - must be checked.
            // In this case the interpolation will be out of bounds.
            MFEM_VERIFY(fabs(rho_interp(el_e_idx + q)) < eps,
                        "Nonzero density at an empty position: "
                        << rho_interp(el_e_idx + q));
         }
      }

      el_e_idx += nqp;
   }
}

void InterpolationRemap::UpdateRhoInterp(QuadratureFunction &rho_interp,
      Vector &rho_min, Vector &rho_max)
{
   const int s = rho_interp.Size();
   for (int i = 0; i < s; i++)
   {
      if (rho_interp(i) + 1e-12 < rho_min(i))
      {
         rho_interp(i) = 0.5 * (rho_min(i) + rho_max(i));
      }
   }
}

void InterpolationRemap::CalcEBounds(const ParGridFunction &e_init,
                                     Array<bool> &active_el_0,
                                     const ParGridFunction &e_interp,
                                     const Vector &pos_final,
                                     const Vector &ind_max,
                                     Vector &e_min, Vector &e_max,
                                     BoundsType bounds_type)
{
   const double eps = 1e-12;

   const int size_e = e_interp.Size(), NE = pmesh_init.GetNE();
   const int s = size_e / NE;
   e_min.SetSize(size_e); e_max.SetSize(size_e);
   e_min = 0.0; e_max = 0.0;

   if (bounds_type == ELEM_INIT)
   {
      // Form the min and max functions on every MPI task.
      // All on the initial mesh.
      L2_FECollection fec_L2(0, pmesh_init.Dimension());
      ParFiniteElementSpace pfes_L2(&pmesh_init, &fec_L2);
      ParGridFunction e_el_min(&pfes_L2), e_el_max(&pfes_L2);
      for (int e = 0; e < NE; e++)
      {
         if (active_el_0[e] == false) { continue; }
         Vector e_init_vals;
         e_init.GetElementDofValues(e, e_init_vals);
         e_el_min(e) = e_init_vals.Min();
         e_el_max(e) = e_init_vals.Max();
      }

      Vector pos_nodes_final;
      GetDOFPositions(*e_interp.ParFESpace(), pos_final, pos_nodes_final);

      FindPointsGSLIB finder(pmesh_init.GetComm());
      finder.Setup(pmesh_init);
      finder.SetL2AvgType(FindPointsGSLIB::NONE);
      finder.Interpolate(pos_nodes_final, e_el_min, e_min);
      finder.Interpolate(pos_nodes_final, e_el_max, e_max);
      finder.FreeData();

      return;
   }

   if (bounds_type == ELEM_BOTH) { MFEM_ABORT("not implemented."); }

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

      // Compute min and max density in the new mesh element.
      double el_min =   std::numeric_limits<double>::infinity(),
             el_max = - std::numeric_limits<double>::infinity();
      // The new mesh element is completely empty -> we want zeros in it.
      if (el_has_ind == false) { el_min = el_max = 0.0; }
      else
      {
         Vector e_interp_el;
         e_interp.GetElementDofValues(e, e_interp_el);
         bool el_has_e_value = false;
         for (int i = 0; i < s; i++)
         {
            // Bounds are taken only from points where the material was present
            // initially. This is achieved by checking the interpolated values,
            // which come from the values on the initial mesh. We assume that
            // zero corresponds to no material. We can't check in indicator
            // values, as those are not available at the energy DOF locations.
            if (fabs(e_interp_el(i)) > eps)
            {
               el_has_e_value = true;
               el_min = std::min(el_min, e_interp_el(i));
               el_max = std::max(el_max, e_interp_el(i));
            }
         }
         MFEM_VERIFY(el_has_e_value == true, "No e values in the new element!");
      }

      // Set the bounds.
      // When material is present, we set bounds at all energy DOFs.
      for (int i = 0; i < s; i++)
      {
         e_min(s * e + i) = el_min;
         e_max(s * e + i) = el_max;

         // Note that it's fine to be lower than the mininum, i.e., in
         // elements where the ind function propages due to small diffusion.
         if (e_interp(s * e + i) > e_max(s * e + i) + eps && el_has_ind == true)
         {
            MFEM_ABORT("Error: interpolated energy is above upper bound: "
                       << e_interp(s * e + i) << " " << e_max(s * e + i));

            // Note that el_has_ind == true is needed, because we can get an
            // element with no material, but with nonzero interpolated energy.
            // This is due to the misfit between the locations of the inicator
            // quad points and the energy DOFs.
         }
      }
   }
}

void InterpolationRemap::UpdateEInterp(ParGridFunction &e_interp,
                                       Vector &e_min, Vector &e_max)
{
   const int s = e_interp.Size();
   for (int i = 0; i < s; i++)
   {
      // This is used to get the "extensions", as the interpolated values
      // are zero at those parts of a new element where there was no material.
      if (e_interp(i) < e_min(i) || e_interp(i) > e_max(i))
      {
         e_interp(i) = 0.5 * (e_min(i) + e_max(i));
      }
   }
}

void InterpolationRemap::CalcVBounds(const ParGridFunction &v_interp,
                                     Vector &v_min, Vector &v_max)
{
   real_t max = v_interp.Max(), min = v_interp.Min();
   MPI_Allreduce(MPI_IN_PLACE, &max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   v_min.SetSize(v_interp.Size());
   v_max.SetSize(v_interp.Size());

   // Make it more strict, per component, if this looks bad.
   v_min = min;
   v_max = max;
}

void InterpolationRemap::GetTargetValues(const Vector &interp,
      const Vector &min, const Vector &max, Vector &target)
{
   int size = interp.Size();

   target.SetSize(size);

   for (int ik = 0; ik< size; ik++)
   {
      if ( std::abs(min[ik] - max[ik]) < 1e-12)
      {
         target[ik] = min[ik];
      }
      else { target[ik] = interp[ik]; }
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

void InterpolationRemap::ComputePressure(const Vector &pos,
      const QuadratureFunction &rho_,
      const ParGridFunction &e_,
      QuadratureFunction &pressure)
{
   const QuadratureSpace *qspace = dynamic_cast<const QuadratureSpace *>
                                   (rho_.GetSpace());

   auto mesh = qspace->GetMesh();
   const int NE = mesh->GetNE();

   int counter = 0;

   for (int e = 0; e < NE; e++)
   {
      const IntegrationRule &ir = qspace->GetElementIntRule(e);
      const int nqp = ir.GetNPoints();

      // Transformation w.r.t. the given mesh positions.
      IsoparametricTransformation Tr;
      mesh->GetElementTransformation(e, pos, &Tr);

      Vector rho_vals(nqp), e_vals(nqp);
      rho_.GetValues(e, rho_vals);
      e_.GetValues(Tr, ir, e_vals);

      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         pressure[counter] = 0.4* rho_vals(q) * e_vals(q);
         counter++;
      }
   }


}

} // namespace mfem
