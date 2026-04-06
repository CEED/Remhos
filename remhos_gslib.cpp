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

void InitializeQuadratureFunction(Coefficient &c,
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
         if (active_quads && (*active_quads)[e * nip + q] == false)
         {
            qf(e*nip + q) = 0.0;
         }
         else
         {
            qf(e*nip + q) = c.Eval(Tr, ip);
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
   QuadratureFunction u_interpolated(qspace_final);
   Vector interp_vals(quads_cnt);
   FindPointsGSLIB finder(pmesh_init.GetComm());
   finder.Setup(pmesh_lor);
   finder.Interpolate(pos_quad_final, u_0_lor, u_interpolated);
   finder.FreeData();


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

void InterpolationRemap::RemapHydro(const Vector &ind_rho_e_v_0,
                                    bool remap_v, bool p_control,
                                    const QuadratureFunction &p_0,
                                    Array<bool> &active_el_0,
                                    const Vector &pos_final,
                                    Vector &ind_rho_e_v, int opt_type,
                                    bool interpolate_e_HO,
                                    bool adjust_diffusion)
{
   const int dim = pmesh_init.Dimension();
   MFEM_VERIFY(dim > 1, "Interpolation remap works only in 2D and 3D.");
   MFEM_VERIFY(qspace && pfes_e && pfes_v, "Spaces are not specified.");

   pmesh_final.SetNodes(pos_final);
   QuadratureSpace qspace_final(pmesh_final, qspace->GetIntRule(0));
   ParFiniteElementSpace pfes_e_final(&pmesh_final, pfes_e->FEColl());
   ParFiniteElementSpace pfes_v_final(&pmesh_final, pfes_v->FEColl(), dim);
   ParFiniteElementSpace pfes_v_scalar_final(&pmesh_final, pfes_v->FEColl());

   // Generate list of points where e will be interpolated.
   // The interpolation is to Gauss-Legendre to keep optimal order.
   L2_FECollection fec_GL(pfes_e_final.FEColl()->GetOrder(),
                          dim, BasisType::GaussLegendre);
   ParFiniteElementSpace pfes_GL(&pmesh_final, &fec_GL);
   Vector pos_dof_GL_final;
   if (interpolate_e_HO)
   {
      GetDOFPositions(pfes_GL, pos_final, pos_dof_GL_final);
   }

   // Extract initial data from the BlockVector.
   const int size_qf   = qspace->GetSize(),
             size_gf_e = pfes_e->GetVSize(),
             size_gf_v = pfes_v->GetVSize(),
             size_gf_v_true = pfes_v->GetTrueVSize();
   Vector *irev_ptr = const_cast<Vector *>(&ind_rho_e_v_0);
   QuadratureFunction ind_0(qspace, irev_ptr->GetData()),
                      rho_0(qspace, irev_ptr->GetData() + size_qf);
   ParGridFunction e_0(pfes_e, irev_ptr->GetData() + 2*size_qf),
                   v_0(pfes_v, irev_ptr->GetData() + 2*size_qf + size_gf_e);

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

   //
   // Interpolate into ind_rho_e_v_interp.
   //
   Vector ind_rho_e_v_interp(ind_rho_e_v.Size());
   real_t *irev_data = ind_rho_e_v_interp.GetData();
   QuadratureFunction ind_interp(&qspace_final, irev_data),
                      rho_interp(&qspace_final, irev_data + size_qf);
   QuadratureFunction p_interp(&qspace_final);
   QuadratureFunction e_interp_qf(&qspace_final);
   ParGridFunction e_interp(&pfes_e_final, irev_data + 2*size_qf),
                   e_interp_GL(&pfes_GL),
                   v_interp(&pfes_v_final, irev_data + 2*size_qf + size_gf_e);
   FindPointsGSLIB finder(pmesh_init.GetComm());
   finder.SetL2AvgType(FindPointsGSLIB::NONE);
   finder.Setup(pmesh_lor);
   // Interpolate ind at the quadrature positions.
   finder.Interpolate(pos_quad_final, ind_0_lor, ind_interp);
   // Interpolate rho at the quadrature positions.
   finder.Interpolate(pos_quad_final, rho_0_lor, rho_interp);
   // For control of p, interpolate p at the quadrature positions.
   if (p_control)
   {
      finder.Interpolate(pos_quad_final, p_0_lor, p_interp);
      VisQuadratureFunction(pmesh_final, p_interp, "p QF interpolated", 0, 0);
   }
   finder.Setup(pmesh_init);
   finder.SetL2AvgType(FindPointsGSLIB::NONE);

   if (interpolate_e_HO)
   {
      // Interpolate e as a Gauss-Legendre GF to preserve the order.
      // Then project to the Bernstein one.
      // This oscillates badly when e has a jump. No bounds preservation.
      finder.Interpolate(pos_dof_GL_final, e_0, e_interp_GL);
      e_interp.ProjectGridFunction(e_interp_GL);
   }
   else
   {
      // Standard interpolation at the DOFs. Preserves bounds, not the order.
      finder.Interpolate(pos_dof_e_final, e_0, e_interp);
   }
   // Energy is additionally interpolated at the quad positions,
   // to have spatial correspondence to the volume fractions.
   finder.Interpolate(pos_quad_final, e_0, e_interp_qf);
   // Interpolate and fix the H1 vector ordering.
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

   // Clean material from diffused elements.
   if (adjust_diffusion)
   {
      Array<bool> active_el;
      AdjustDiffusion(ind_interp, rho_interp, e_interp, active_el);
   }

   //
   // Compute min / max bounds.
   //
   // Also adjust interpolated values in some special cases.
   Vector ind_min, ind_max;
   CalcQuadBounds(ind_0, ind_interp, pos_final, ind_min, ind_max, ELEM_FINAL);
   CleanEmptyZones(ind_interp, ind_min, ind_max);
   Vector rho_min, rho_max;
   CalcRhoBounds(rho_interp, ind_interp, ind_max, rho_min, rho_max);
   UpdateRhoInterp(rho_interp, rho_min, rho_max);
   if (p_control)
   {
      rho_min -= 0.1;
      rho_max += 0.1;
   }
   // {
   //    QuadratureFunction gf_min(qspace), gf_max(qspace);
   //    gf_min = rho_min, gf_max = rho_max;

   //    VisQuadratureFunction(pmesh_final, ind_interp, "ind interp", 0, 500);
   //    VisQuadratureFunction(pmesh_final, rho_interp, "rho interp", 0, 500);
   //    VisQuadratureFunction(pmesh_final, gf_min, "rho_min QF", 0, 500);
   //    VisQuadratureFunction(pmesh_final, gf_max, "rho_max QF", 400, 500);
   //    MFEM_ABORT("rho bounds");
   // }
   Vector e_min, e_max;
   CalcEBounds(e_0, active_el_0, e_interp, e_interp_qf, pos_final, ind_max,
               e_min, e_max, ELEM_FINAL);
   UpdateEInterp(e_interp, e_min, e_max);
   // {
   //    ParGridFunction gf_min(e_interp), gf_max(e_interp);
   //    gf_min = e_min, gf_max = e_max;

   //    socketstream vis_min, vis_max;
   //    char vishost[] = "localhost";
   //    int  visport   = 19916;
   //    vis_min.precision(8);
   //    vis_max.precision(8);

   //    VisualizeField(vis_min, vishost, visport, gf_min, "e min",
   //                   0, 500, 300, 300);
   //    VisualizeField(vis_max, vishost, visport, gf_max, "e max",
   //                   300, 500, 300, 300);
   //    //MFEM_ABORT("e bounds");
   // }
   Vector v_min, v_max;
   if (remap_v) { CalcVBounds(v_interp, v_min, v_max); }

   //
   // Report conservation errors of ire_interp.
   //
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
   Vector moment_0(dim), moment_f(dim);
   for (int d = 0; d < dim; d++)
   {
      moment_0(d) = Integrate(pos_init,
                              &ind_0, &rho_0, nullptr, &v_0, d);
      moment_f(d) = Integrate(pos_final,
                              &ind_interp, &rho_interp, nullptr, &v_interp, d);
   }
   const double tot_en_0 = Integrate(pos_init,
                                     &ind_0, &rho_0, &e_0, &v_0);
   const double tot_en_f = Integrate(pos_final,
                                     &ind_interp, &rho_interp, &e_interp, &v_interp);

   if (pmesh_init.GetMyRank() == 0)
   {
      auto old_flags = std::cout.flags();
      auto old_prec  = std::cout.precision();
      cout << std::scientific << std::showpos << setprecision(3);
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
                << fabs(mass_0 - mass_f) / mass_0 * 100 << endl;
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
                 << fabs(moment_0(d) - moment_f(d)) / moment_0(d) * 100 << endl;
         }
         cout << "*\n"
              << "Total energy initial:       " << tot_en_0 << std::endl
              << "Total energy interp:        " << tot_en_f << std::endl
              << "Total energy interp diff:   "
              << fabs(tot_en_0 - tot_en_f) << endl
              << "Total energy interp diff %: "
              << fabs(tot_en_0 - tot_en_f) / tot_en_0 * 100 << endl;
      }
      else
      {
         std::cout << "*\n"
                   << "Intern energy initial:      " << energy_0 << std::endl
                   << "Intern energy interp:       " << energy_f << std::endl
                   << "Intern energy interp diff:  "
                   << fabs(energy_0 - energy_f) << endl
                   << "Intern energy interp diff %:"
                   << fabs(energy_0 - energy_f) / energy_0 * 100 << endl;
      }
      std::cout.flags(old_flags);
      std::cout.precision(old_prec);
   }

   // Construct BlockVectors for the min/max values of all fields.
   const int numBlocks = (remap_v) ? 5 : 4;
   Array<int> offset(numBlocks);
   offset[0] = 0;
   offset[1] = offset[0] + size_qf;
   offset[2] = offset[1] + size_qf;
   offset[3] = offset[2] + size_gf_e;
   if (remap_v)
   {
      offset[4] = offset[3] + size_gf_v;
   }
   BlockVector x_min(offset);
   BlockVector x_max(offset);
   x_min.GetBlock(0) = ind_min;
   x_min.GetBlock(1) = rho_min;
   x_min.GetBlock(2) = e_min;
   if (remap_v) { x_min.GetBlock(3) = v_min; }
   x_max.GetBlock(0) = ind_max;
   x_max.GetBlock(1) = rho_max;
   x_max.GetBlock(2) = e_max;
   if (remap_v) { x_max.GetBlock(3) = v_max; }

   //
   // Optimize.
   //
   if (opt_type == 0)
   {
      ind_rho_e_v = ind_rho_e_v_interp;
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
         //ind_rho_e_v.GetSubVector(optProbInd,ind_rho_e_sub);
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
         auto hiop = new RemhosHydroHiOpProblem(qspace_final,
                                                pfes_e_final, pfes_v_final,
                                                pos_final,
                                                initial_design, p_interp,
                                                NumDesVar, x_minsub, x_maxsub,
                                                volume_0, mass_0, moment_0, tot_en_0,
                                                5, false, optProbInd, true,
                                                subprob, p_control);
         hiop->setWeightedSpaceType(weightedSpace);

         if (p_control)
         {
            hiop->w_1 = 1.0;
            hiop->w_2 = 1.0;
            hiop->w_3 = 1.0;
            hiop->w_4 = 0.0;
            hiop->w_p = 1e3;
         }

         ot_prob = hiop;
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

      ind_rho_e_v = L_vector_design;

      delete optsolver;
      delete ot_prob;
   }
   else if (opt_type == 2)
   {
      std::vector<ParFiniteElementSpace*> fes({&pfes_e_final, &pfes_v_scalar_final});

      Array<int> space_idx({-1, -1, 0});
      Array<int> offsets({0,
                          qspace_final.GetSize(),
                          qspace_final.GetSize(),
                          pfes_e_final.GetTrueVSize()});
      Array<int> b_offsets(offsets);
      Array<bool> has_bounds({true, true, true});
      if (remap_v)
      {
         for (int i=0; i<dim; i++)
         {
            space_idx.Append(1);
            offsets.Append(pfes_v_scalar_final.GetTrueVSize());
            // // Bound constraint for v
            b_offsets.Append(pfes_v_scalar_final.GetTrueVSize());
            has_bounds.Append(true);
            // No bound constraint for v
            // b_offsets.Append(0);
            // has_bounds.Append(false);
         }
      }
      offsets.PartialSum();
      b_offsets.PartialSum();
      MFEM_VERIFY(dynamic_cast<const L2_FECollection*>(pfes_e_final.FEColl()) !=
                  nullptr,
                  "Expecting L2_FECollection for pfes_e_final.");

      BlockVector x_initial(offsets);
      BlockVector x_initial_LVec(ind_rho_e_v_interp, offset);
      // Since all functions other than v satisfy L-vector == T-vector,
      // we can use the L-vector for bounds.
      BlockVector x_min_final_LVec(x_min.GetData(), offset);
      BlockVector x_max_final_LVec(x_max.GetData(), offset);
      BlockVector x_min_final(b_offsets);
      BlockVector x_max_final(b_offsets);
      for (int i=0; i<3; i++)
      {
         x_initial.GetBlock(i) = x_initial_LVec.GetBlock(i);
         x_min_final.GetBlock(i) = x_min_final_LVec.GetBlock(i);
         x_max_final.GetBlock(i) = x_max_final_LVec.GetBlock(i);
      }
      if (remap_v)
      {
         ParGridFunction vtmp(&pfes_v_scalar_final, (real_t*)nullptr);
         const int n = pfes_v_scalar_final.GetVSize();
         MFEM_VERIFY(n*dim == pfes_v_final.GetVSize(),
                     "Expecting 3*n dofs for pfes_v_scalar_final.");
         for (int i=0; i<dim; i++)
         {
            vtmp.MakeRef(&pfes_v_scalar_final, x_initial_LVec.GetBlock(3), i*n);
            vtmp.GetTrueDofs(x_initial.GetBlock(3+i));

            if (!has_bounds[3 + i]) { continue; }
            vtmp.MakeRef(&pfes_v_scalar_final, x_min_final_LVec.GetBlock(3), i*n);
            vtmp.GetTrueDofs(x_min_final.GetBlock(3+i));

            vtmp.MakeRef(&pfes_v_scalar_final, x_max_final_LVec.GetBlock(3), i*n);
            vtmp.GetTrueDofs(x_max_final.GetBlock(3+i));
         }
      }

      // Objective function: 0.5 * || u - u_initial ||^2
      remap::RemapObjectiveFunctional remap_obj(qspace_final, fes, x_initial,
            space_idx);
      // Constraint
      std::vector<std::unique_ptr<ComposedFunctional>> funcs(3 + remap_v*dim);

      funcs[0] = std::make_unique<ComposedFunctional>(
                    remap::volume_f, remap::volume_df, qspace_final, fes, space_idx);
      funcs[0]->SetTarget(volume_0);
      funcs[1] = std::make_unique<ComposedFunctional>(
                    remap::mass_f, remap::mass_df, qspace_final, fes, space_idx);
      funcs[1]->SetTarget(mass_0);
      if (!remap_v) // use potential
      {
         funcs[2] = std::make_unique<ComposedFunctional>(
                       remap::potential_f, remap::potential_df, qspace_final, fes, space_idx);
         funcs[2]->SetTarget(energy_0);
      }
      else
      {
         for (int i=0; i<dim; i++)
         {
            funcs[3+i] = std::make_unique<ComposedFunctional>(
            [i](const Vector &x) { return remap::momentum_f(x, i); },
            [i](const Vector &x, Vector &g) { remap::momentum_df(x, g, i); },
            qspace_final, fes, space_idx);
            funcs[3+i]->SetTarget(moment_0[i]);
         }
         funcs[2] = std::make_unique<ComposedFunctional>(
                       remap::energy_f, remap::energy_df, qspace_final, fes, space_idx);
         funcs[2]->SetTarget(tot_en_0);
      }

      StackedSharedFunctional C(offsets.Last());
      for (auto &f : funcs)
      {
         f->SetComm(pmesh_final.GetComm());
         C.AddFunctional(*f);
      }
      MassOperator mass_q(qspace_final), mass_l2(pfes_e_final),
                   mass_h1(pfes_v_scalar_final);
      MultiMassOperator mass;
      mass.Append(mass_q);
      mass.Append(mass_q);
      mass.Append(mass_l2);
      if (remap_v) { for (int i=0; i<dim; i++) { mass.Append(mass_h1); } }
      PointwiseFermiDirac sigmoid(x_min_final, x_max_final);
      Array<LegendreFunction*> legendre_funcs({&sigmoid});
      Array<int> dummy_offset({0, x_min_final.Size()});
      Dykstra projector(pmesh_final.GetComm(), C, mass,
                        legendre_funcs, dummy_offset,
                        x_min_final, x_max_final, atol, max_iter);
      projector.Project(x_initial);
      BlockVector x_final_LVector(ind_rho_e_v, offset);
      for (int i=0; i<3; i++)
      {
         x_final_LVector.GetBlock(i) = x_initial.GetBlock(i);
      }
      if (remap_v)
      {
         ParGridFunction vtmp(&pfes_v_final, x_final_LVector.GetBlock(3));
         Vector v_final_TVector(x_initial.GetBlock(3).GetData(),
                                pfes_v_final.GetTrueVSize());
         vtmp.SetFromTrueDofs(v_final_TVector);
      }
   }
   else { MFEM_ABORT("not implemented!"); }

   QuadratureFunction ind(&qspace_final, ind_rho_e_v.GetData()),
                      rho(&qspace_final, ind_rho_e_v.GetData() + size_qf);
   ParGridFunction e(&pfes_e_final, ind_rho_e_v.GetData() + 2*size_qf);
   ParGridFunction v(&pfes_v_final, ind_rho_e_v.GetData() + 2*size_qf + size_gf_e);

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
      auto old_flags = std::cout.flags();
      auto old_prec  = std::cout.precision();
      cout << std::scientific << std::showpos << setprecision(3);
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
                << endl << "*\n";

      if (remap_v)
      {
         for (int d = 0; d < dim; d++)
         {
            std::cout << "Moment in dim "<<d+1 <<" initial:          " << moment_0[d] <<
                      std::endl
                      << "Moment in dim "<<d+1 <<" optimized:        " << moment_f_opt(
                         d) << std::endl
                      << "Moment in dim "<<d+1 <<" optimized diff:   "
                      << (moment_f_opt(d) - moment_0[d]) << endl
                      << "Moment in dim "<<d+1 <<" optimized diff %: "
                      << (moment_f_opt(d) - moment_0[d]) / moment_0[d] * 100
                      << endl << "*\n";
         }

         std::cout<< "Total energy initial:          " << tot_en_0 << std::endl
                  << "Total energy optimized:        " << tot_energy_f_opt << std::endl
                  << "Total energy optimized diff:   "
                  << (tot_energy_f_opt- tot_en_0) << endl
                  << "Total energy optimized diff %: "
                  << (tot_energy_f_opt- tot_en_0) / tot_en_0 * 100
                  << endl;
      }
      else
      {
         std::cout << "Energy initial:          " << energy_0 << std::endl
                   << "Energy optimized:        " << energy_f_opt << std::endl
                   << "Energy optimized diff:   "
                   << (energy_f_opt- energy_0) << endl
                   << "Energy optimized diff %: "
                   << (energy_f_opt- energy_0) / energy_0 * 100
                   << endl;
      }
      cout.flags(old_flags);
      cout.precision(old_prec);
   }

   // Check for bounds violations.
   if (Mpi::Root()) { std::cout << "-------\nIndicator violations: \n"; }
   CheckBounds(pmesh_init.GetMyRank(), ind, ind_min, ind_max);
   if (Mpi::Root()) { std::cout << "*\nDensity violations: \n"; }
   CheckBounds(pmesh_init.GetMyRank(), rho, rho_min, rho_max);
   if (Mpi::Root()) { std::cout << "*\nInternal Energy violations: \n"; }
   CheckBounds(pmesh_init.GetMyRank(), e, e_min, e_max);
   if (remap_v)
   {
      if (Mpi::Root()) { std::cout << "*\nVelocity violations: \n"; }
      CheckBounds(pmesh_init.GetMyRank(), v, v_min, v_max);
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

void InterpolationRemap::AdjustDiffusion(QuadratureFunction &ind_interp,
                                         QuadratureFunction &rho_interp,
                                         ParGridFunction &e_interp,
                                         Array<bool> &active_el)
{
   // Idea: if an element doesn't have ind > cutoff at least at one point,
   // then the values in this element come from diffusion. This is cleaned.
   const double ind_cutoff = 0.9;
   const IntegrationRule &ir = qspace->GetIntRule(0);
   const int NE = pmesh_final.GetNE();
   const int ndof = e_interp.Size() / NE,
             nqp  = ir.GetNPoints();

   active_el.SetSize(NE);
   for (int e = 0; e < NE; e++)
   {
      active_el[e] = false;
      for (int q = 0; q < nqp; q++)
      {
         if (ind_interp(e*nqp + q) > ind_cutoff) { active_el[e] = true; break; }
      }

      if (active_el[e]) { continue; }

      for (int q = 0; q < nqp; q++)
      {
         ind_interp(e*nqp + q) = 0.0;
         rho_interp(e*nqp + q) = 0.0;
      }
      for (int i = 0; i < ndof; i++)
      {
         e_interp(e*ndof + i) = 0.0;
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
            MFEM_VERIFY(rho_interp(el_e_idx + q) < el_max + eps,
                        "Error: interpolated density is above upper bound: "
                        << rho_interp(el_e_idx + q) << " "
                        << rho_max(el_e_idx + q) );
         }
         else
         {
            // No material at the DOF.
            rho_min(el_e_idx + q) = 0.0;
            rho_max(el_e_idx + q) = 0.0;

            // No material, but has density - must be checked.
            // In this case the interpolation will be out of bounds.
            MFEM_VERIFY(fabs(rho_interp(el_e_idx + q)) < eps,
                        "Error: nonzero density at an empty position: "
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
      // Happens when a new-mesh-element overlaps only part of the old indicator
      // support. The material must diffuse in the whole new-mesh-element.
      if (rho_interp(i) + 1e-12 < rho_min(i))
      {
         rho_interp(i) = 0.5 * (rho_min(i) + rho_max(i));
      }
   }
}

void InterpolationRemap::CalcEBounds(const ParGridFunction &e_init,
                                     Array<bool> &active_el_0,
                                     const ParGridFunction &e_interp,
                                     const QuadratureFunction &e_interp_qf,
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
      bool el_has_e_value = false;
      for (int q = 0; q < nqp; q++)
      {
         if (ind_max(e*nqp + q) > 1e-12)           { el_has_ind = true; }
         if (fabs(e_interp_qf(e*nqp + q)) > 1e-12) { el_has_e_value = true; }
      }

      // Compute min and max energy in the new mesh element.
      double el_min =   std::numeric_limits<double>::infinity(),
             el_max = - std::numeric_limits<double>::infinity();
      // The new mesh element is completely empty -> we want zeros in it.
      if (el_has_ind == false) { el_min = el_max = 0.0; }
      else
      {
         MFEM_VERIFY(el_has_e_value == true, "No e values in the new element!");

         for (int q = 0; q < nqp; q++)
         {
            // Bounds are taken only from points where the material was present
            // initially. This is achieved by checking the interpolated values,
            // which come from the values on the initial mesh. We assume that
            // zero corresponds to no material.
            if (fabs(e_interp_qf(e*nqp + q)) > 1e-12)
            {
               el_min = std::min(el_min, e_interp_qf(e*nqp + q));
               el_max = std::max(el_max, e_interp_qf(e*nqp + q));
            }
         }

         // Taking bounds only at quad points can lead to clipping and reducing
         // the interpolation order.
         for (int i = 0; i < s; i++)
         {
            if (e_interp(s * e + i) > eps)
            {
               el_min = std::min(el_min, e_interp(s * e + i));
               el_max = std::max(el_max, e_interp(s * e + i));
            }
         }
      }

      // Set the bounds at the DOFs.
      // When material is present, we set bounds at all energy DOFs.
      for (int i = 0; i < s; i++)
      {
         e_min(s * e + i) = el_min;
         e_max(s * e + i) = el_max;
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
   v_min.SetSize(v_interp.Size());
   v_max.SetSize(v_interp.Size());
   v_min = v_interp;
   v_max = v_interp;

   // real_t max = v_interp.Max(), min = v_interp.Min();
   // MPI_Allreduce(MPI_IN_PLACE, &max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   // MPI_Allreduce(MPI_IN_PLACE, &min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);


   // // Make it more strict, per component, if this looks bad.
   // v_min = min;
   // v_max = max;
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

void InterpolationRemap::DiffuseIndicator(int diffused_ind_order,
      QuadratureFunction &ind)
{
   if (diffused_ind_order <= 0) { return; }

   const int NE = pmesh_final.GetNE(), dim = pmesh_final.Dimension();
   // Temporary L2 Bernstein function, for duffusion of ind.
   DG_FECollection fec(diffused_ind_order, dim, BasisType::Positive);
   ParFiniteElementSpace pfes_ind_final(&pmesh_final, &fec);
   ParGridFunction tmp(&pfes_ind_final);

   const IntegrationRule &ir = qspace->GetIntRule(0);
   MassIntegrator mass_integ(&ir);
   const int ndof = pfes_ind_final.GetFE(0)->GetDof(),
             nqp  = ir.GetNPoints();
   DenseMatrix M(ndof);
   Vector rhs(ndof), shape(ndof), ML(ndof),
          u(ndof), u_ho(ndof), u_lo(ndof), beta(ndof), z(ndof);
   for (int e = 0; e < NE; e++)
   {
      // Max and min of the FE solution u.
      double u_max = ind(e*nqp), u_min = ind(e*nqp);
      for (int q = 1; q < nqp; q++)
      {
         u_max = fmax(u_max, ind(e*nqp + q));
         u_min = fmin(u_min, ind(e*nqp + q));
      }

      if (fabs(u_min - u_max) < 1e-12) { continue; }

      // Mass matrix and diagonal.
      ElementTransformation *T = pmesh_final.GetElementTransformation(e);
      const FiniteElement *el = pfes_ind_final.GetFE(e);
      mass_integ.AssembleElementMatrix(*el, *T, M);
      M.GetRowSums(ML);

      // RHS.
      rhs = 0.0;
      double rhs_max = 0.0, rhs_min = 0.0;
      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         el->CalcShape(ip, shape);
         T->SetIntPoint(&ip);
         for (int i = 0; i < ndof; i++)
         {
            rhs(i) += ip.weight * T->Weight() * ind(e*nqp + q) * shape(i);
         }
         rhs_max += ip.weight * T->Weight() * u_max;
         rhs_min += ip.weight * T->Weight() * u_min;
      }

      // HO solution.
      DenseMatrixInverse M_inv(&M);
      M_inv.Factor();
      M_inv.Mult(rhs, u_ho);

      // LO solution.
      u_lo = rhs.Sum() / ML.Sum();

      //
      // FCT-project tricks.
      //
      for (int i = 0; i < ndof; i++)
      {
         beta(i) = ML(i);
         z(i) = rhs(i) - ML(i) * u_lo(i);
      }
      beta /= beta.Sum();

      DenseMatrix F(ndof);
      for (int i = 1; i < ndof; i++)
      {
         for (int j = 0; j < i; j++)
         {
            F(i, j) = M(i, j) * (u_ho(i) - u_ho(j)) +
                      (beta(j) * z(i) - beta(i) * z(j));
         }
      }

      Vector gp(ndof), gm(ndof);
      gp = 0.0;
      gm = 0.0;
      for (int i = 1; i < ndof; i++)
      {
         for (int j = 0; j < i; j++)
         {
            double fij = F(i, j);
            if (fij >= 0.0)
            {
               gp(i) += fij;
               gm(j) -= fij;
            }
            else
            {
               gm(i) += fij;
               gp(j) -= fij;
            }
         }
      }

      u = u_lo;

      for (int i = 0; i < ndof; i++)
      {
         double rp = std::max(ML(i) * (u_max - u(i)), 0.0);
         double rm = std::min(ML(i) * (u_min - u(i)), 0.0);
         double sp = gp(i), sm = gm(i);

         gp(i) = (rp < sp) ? rp / sp : 1.0;
         gm(i) = (rm > sm) ? rm / sm : 1.0;
      }

      for (int i = 1; i < ndof; i++)
      {
         for (int j = 0; j < i; j++)
         {
            double fij = F(i, j), aij;

            if (fij >= 0.0) { aij = std::min(gp(i), gm(j)); }
            else            { aij = std::min(gm(i), gp(j)); }

            fij *= aij;
            u(i) += fij / ML(i);
            u(j) -= fij / ML(j);
         }
      }

      // Interpolate back into ind at quad points.
      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         el->CalcShape(ip, shape);
         ind(e*nqp + q) = 0.0;
         for (int i = 0; i < ndof; i++)
         {
            ind(e*nqp + q) += u(i) * shape(i);
         }
      }
   }
}

} // namespace mfem
