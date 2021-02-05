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
//
//                    ____                 __
//                   / __ \___  ____ ___  / /_  ____  _____
//                  / /_/ / _ \/ __ `__ \/ __ \/ __ \/ ___/
//                 / _, _/  __/ / / / / / / / / /_/ (__  )
//                /_/ |_|\___/_/ /_/ /_/_/ /_/\____/____/
//
//                       High-order Remap Miniapp
//
// Remhos (REMap High-Order Solver) is a miniapp that solves the pure advection
// equations that are used to perform discontinuous field interpolation (remap)
// as part of the Eulerian phase in Arbitrary-Lagrangian Eulerian (ALE)
// simulations.
//
// Sample runs: see README.md, section 'Verification of Results'.

#define MFEM_DEBUG_COLOR 154
#include "debug.hpp"

#include "remhos.hpp"

using namespace std;
using namespace mfem;

const int Wx = 800, Wy = 300; // window position
const int Ww = 640, Wh = 640; // window size

// Choice for the problem setup. The fluid velocity, initial condition and
// inflow boundary condition are chosen based on this parameter.
int problem_num = 0;

// 0 is standard transport.
// 1 is standard remap (mesh moves, solution is fixed).
int exec_mode;

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v);

// Initial condition
double u0_function(const Vector &x);
double s0_function(const Vector &x);

// Inflow boundary condition
double inflow_function(const Vector &x);

// Mesh bounding box
Vector bb_min, bb_max;

// Boundary Conditions
void ZeroItOutOnBoundaries(const ParMesh *subcell_mesh,
                           const ParGridFunction *xsub,
                           ParGridFunction &v_sub_gf,
                           VectorGridFunctionCoefficient &v_sub_coef)
{
   Array<int> ess_bdr, ess_vdofs;
   if (subcell_mesh->bdr_attributes.Size() > 0)
   {
      ess_bdr.SetSize(subcell_mesh->bdr_attributes.Max());
   }
   ess_bdr = 1;
   xsub->ParFESpace()->GetEssentialVDofs(ess_bdr, ess_vdofs);
   for (int i = 0; i < ess_vdofs.Size(); i++)
   {
      if (ess_vdofs[i] == -1) { v_sub_gf(i) = 0.0; }
   }
   v_sub_coef.SetGridFunction(&v_sub_gf);
}

/// Main ///
int main(int argc, char *argv[])
{
   // Initialize MPI.
   MPI_Session mpi(argc, argv);
   const int myid = mpi.WorldRank();

   const char *mesh_file = "data/periodic-square.mesh";
   int rs_levels = 2;
   int rp_levels = 0;
   int order = 3;
   int mesh_order = 2;
   int ode_solver_type = 3;
   HOSolverType ho_type           = HOSolverType::LocalInverse;
   LOSolverType lo_type           = LOSolverType::None;
   FCTSolverType fct_type         = FCTSolverType::None;
   MonolithicSolverType mono_type = MonolithicSolverType::None;
   bool pa = false;
   int smth_ind_type = 0;
   double t_final = 4.0;
   double dt = 0.005;
   int max_tsteps = -1;
   bool visualization = true;
   bool visit = false;
   bool verify_bounds = false;
   bool product_sync = false;
   int vis_steps = 100;
   const char *device_config = "cpu";
   bool amr = false;
   int amr_estimator = amr::Estimator::L2ZZ;
   double amr_ref_threshold = 1e-3;
   double amr_deref_threshold = 1e-5;
   int amr_max_level = 1;
   int amr_nc_limit = 0;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem_num, "-p", "--problem",
                  "Problem setup to use. See options in velocity_function().");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite element solution.");
   args.AddOption(&mesh_order, "-mo", "--mesh-order",
                  "Order (degree) of the mesh.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6.");
   args.AddOption((int*)(&ho_type), "-ho", "--ho-type",
                  "High-Order Solver: 0 - No HO solver,\n\t"
                  "                   1 - Neumann iteration,\n\t"
                  "                   2 - CG solver,\n\t"
                  "                   3 - Local inverse.");
   args.AddOption((int*)(&lo_type), "-lo", "--lo-type",
                  "Low-Order Solver: 0 - No LO solver,\n\t"
                  "                  1 - Discrete Upwind,\n\t"
                  "                  2 - Preconditioned Discrete Upwind,\n\t"
                  "                  3 - Residual Distribution,\n\t"
                  "                  4 - Subcell Residual Distribution.");
   args.AddOption((int*)(&fct_type), "-fct", "--fct-type",
                  "Correction type: 0 - No nonlinear correction,\n\t"
                  "                 1 - Flux-based FCT,\n\t"
                  "                 2 - Local clip + scale,\n\t"
                  "                 3 - Local clip + nonlinear penalization.");
   args.AddOption((int*)(&mono_type), "-mono", "--mono-type",
                  "Monolithic solver: 0 - No monolithic solver,\n\t"
                  "                   1 - Residual Distribution,\n\t"
                  "                   2 - Subcell Residual Distribution.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly",
                  "Enable or disable partial assembly for the HO solution.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&smth_ind_type, "-si", "--smth_ind",
                  "Smoothness indicator: 0 - no smoothness indicator,\n\t"
                  "                      1 - approx_quadratic,\n\t"
                  "                      2 - exact_quadratic.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&max_tsteps, "-ms", "--max-steps",
                  "Maximum number of steps (negative means no restriction).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&verify_bounds, "-vb", "--verify-bounds", "-no-vb",
                  "--no-verify-bounds",
                  "Verify solution bounds after each time step.");
   args.AddOption(&product_sync, "-ps", "--product-sync", "-no-ps",
                  "--no-product-sync",
                  "Enable remap of synchronized product fields.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&amr, "-amr", "--enable-amr", "-no-amr", "--disable-amr",
                  "Enable adaptive mesh refinement.");
   args.AddOption(&amr_estimator, "-ae", "--amr-estimator",
                  "AMR estimator: 0:ZZ, 1:L2ZZ, 2:JJt, 3:Custom");
   args.AddOption(&amr_ref_threshold, "-ar", "--amr-ref-threshold",
                  "AMR refinement threshold.");
   args.AddOption(&amr_deref_threshold, "-ad", "--amr-deref-threshold",
                  "AMR refinement threshold.");
   args.AddOption(&amr_max_level, "-am", "--amr-max-level",
                  "AMR max refined level "
                  "(after the initial serial and parallel refinements)");
   args.AddOption(&amr_nc_limit, "-al", "--amr-nc-limit",
                  "AMR maximum level of hanging nodes, (0 = unlimited)");

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   // Enable hardware devices such as GPUs, and programming models such as
   // CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // When not using lua, exec mode is derived from problem number convention
   if (problem_num < 10)      { exec_mode = 0; }
   else if (problem_num < 20) { exec_mode = 1; }
   else { MFEM_ABORT("Unspecified execution mode."); }

   // AMR sanity checks
   // Re-evaluate amr_max_level after arguments have been parsed
   if (amr)
   {
      MFEM_VERIFY(exec_mode == 0, "Only standard transport is supported.")
   }

   // Read the serial mesh from the given mesh file on all processors.
   // Refine the mesh in serial to increase the resolution.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   const int dim = mesh->Dimension();
   if (amr)
   {
      mesh->EnsureNCMesh();
      amr_max_level += rs_levels + rp_levels;
   }

   const amr::Options amr_options = { amr_estimator,
                                      order, mesh_order,
                                      amr_max_level, amr_nc_limit,
                                      amr_ref_threshold, amr_deref_threshold
                                    };

   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   mesh->GetBoundingBox(bb_min, bb_max, max(order, 1));
   //mesh->Finalize(true);

   // Parallel partitioning of the mesh.
   // Refine the mesh further in parallel to increase the resolution.
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < rp_levels; lev++) { pmesh.UniformRefinement(); }


   // Define the ODE solver used for time integration. Several explicit
   // Runge-Kutta methods are available.
   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK2Solver(1.0); break;
      case 3: ode_solver = new RK3SSPSolver; break;
      case 4:
         if (myid == 0) { MFEM_WARNING("RK4 may violate the bounds."); }
         ode_solver = new RK4Solver; break;
      case 6:
         if (myid == 0) { MFEM_WARNING("RK6 may violate the bounds."); }
         ode_solver = new RK6Solver; break;
      default:
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         return 3;
   }

   // Check if the input mesh is periodic.
   const bool periodic = pmesh.GetNodes() != NULL &&
                         dynamic_cast<const L2_FECollection *>
                         (pmesh.GetNodes()->FESpace()->FEColl()) != NULL;
   pmesh.SetCurvature(mesh_order, periodic);

   if (amr) { pmesh.EnsureNCMesh(); }

   FiniteElementCollection *mesh_fec;
   if (periodic)
   {
      mesh_fec = new L2_FECollection(mesh_order, dim, BasisType::GaussLobatto);
   }
   else
   {
      assert(false);
      //mesh_fec = new H1_FECollection(mesh_order, dim, BasisType::GaussLobatto);
   }
   // Current mesh positions.
   ParFiniteElementSpace mesh_pfes(&pmesh, mesh_fec, dim);
   ParGridFunction x(&mesh_pfes);
   pmesh.SetNodalGridFunction(&x);

   // Store initial mesh positions.
   Vector x0(x.Size());
   x0 = x;

   // Velocity for the problem. Depending on the execution mode, this is the
   // advective velocity (transport) or mesh velocity (remap).
   VectorFunctionCoefficient velocity(dim, velocity_function);

   // Mesh velocity.
   GridFunction v_gf(x.FESpace());
   VectorGridFunctionCoefficient v_coef(&v_gf);

   // If remap is on, obtain the mesh velocity by moving the mesh to the final
   // mesh positions, and taking the displacement vector.
   // The mesh motion resembles a time-dependent deformation, e.g., similar to
   // a deformation that is obtained by a Lagrangian simulation.
   if (exec_mode == 1)
   {
      assert(false);/*
      ParGridFunction v(&mesh_pfes);
      VectorFunctionCoefficient vcoeff(dim, velocity_function);
      v.ProjectCoefficient(vcoeff);

      double t = 0.0;
      while (t < t_final)
      {
         t += dt;
         // Move the mesh nodes.
         x.Add(std::min(dt, t_final-t), v);
         // Update the node velocities.
         v.ProjectCoefficient(vcoeff);
      }

      // Pseudotime velocity.
      add(x, -1.0, x0, v_gf);

      // Return the mesh to the initial configuration.
      x = x0;*/
   }

   // Define the discontinuous DG finite element space of the given
   // polynomial order on the refined mesh.
   const int btype = BasisType::Positive;
   DG_FECollection fec(order, dim, btype);
   ParFiniteElementSpace pfes(&pmesh, &fec);

   // Check for meaningful combinations of parameters.
   const bool forced_bounds = lo_type   != LOSolverType::None ||
                              mono_type != MonolithicSolverType::None;
   if (forced_bounds)
   {
      assert(false);/*
      MFEM_VERIFY(btype == 2,
                  "Monotonicity treatment requires Bernstein basis.");

      if (order == 0)
      {
         // Disable monotonicity treatment for piecewise constants.
         if (myid == 0)
         { mfem_warning("For -o 0, monotonicity treatment is disabled."); }
         lo_type = LOSolverType::None;
         fct_type = FCTSolverType::None;
         mono_type = MonolithicSolverType::None;
      }*/
   }

   const bool use_subcell_RD =
      ( lo_type   == LOSolverType::ResDistSubcell ||
        mono_type == MonolithicSolverType::ResDistMonoSubcell );

   if (use_subcell_RD && order==1)
   { MFEM_ABORT("Subcell schemes are not applicable to linear FE."); }

   const int prob_size = pfes.GlobalTrueVSize();
   if (myid == 0) { cout << "Number of unknowns: " << prob_size << endl; }

   // Fields related to inflow BC.
   FunctionCoefficient inflow(inflow_function);
   ParGridFunction inflow_gf(&pfes);
   if (problem_num == 7) // Convergence test: use high order projection.
   {
      assert(false);/*
      L2_FECollection l2_fec(order, dim);
      ParFiniteElementSpace l2_fes(&pmesh, &l2_fec);
      ParGridFunction l2_inflow(&l2_fes);
      l2_inflow.ProjectCoefficient(inflow);
      inflow_gf.ProjectGridFunction(l2_inflow);*/
   }
   else { inflow_gf.ProjectCoefficient(inflow); }

   // Set up the bilinear and linear forms corresponding to the DG
   // discretization.
   ParBilinearForm Mbf(&pfes);
   Mbf.AddDomainIntegrator(new MassIntegrator);

   ParBilinearForm M_HO(&pfes);
   M_HO.AddDomainIntegrator(new MassIntegrator);

   ParBilinearForm Kbf(&pfes);
   ParBilinearForm K_HO(&pfes);
   if (exec_mode == 0)
   {
      Kbf.AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
      K_HO.AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
   }
   else if (exec_mode == 1)
   {
      assert(false);
      //Kbf.AddDomainIntegrator(new ConvectionIntegrator(v_coef));
      //K_HO.AddDomainIntegrator(new ConvectionIntegrator(v_coef));
   }

   if (ho_type == HOSolverType::CG ||
       ho_type == HOSolverType::LocalInverse ||
       fct_type == FCTSolverType::FluxBased)
   {
      if (exec_mode == 0)
      {
         DGTraceIntegrator *dgt_i = new DGTraceIntegrator(velocity, 1.0, -0.5);
         DGTraceIntegrator *dgt_b = new DGTraceIntegrator(velocity, 1.0, -0.5);
         K_HO.AddInteriorFaceIntegrator(new TransposeIntegrator(dgt_i));
         K_HO.AddBdrFaceIntegrator(new TransposeIntegrator(dgt_b));
      }
      else if (exec_mode == 1)
      {
         assert(false);/*
         DGTraceIntegrator *dgt_i = new DGTraceIntegrator(v_coef, -1.0, -0.5);
         DGTraceIntegrator *dgt_b = new DGTraceIntegrator(v_coef, -1.0, -0.5);
         K_HO.AddInteriorFaceIntegrator(new TransposeIntegrator(dgt_i));
         K_HO.AddBdrFaceIntegrator(new TransposeIntegrator(dgt_b));*/
      }

      K_HO.KeepNbrBlock(true);
   }

   if (pa)
   {
      assert(false);/*
      M_HO.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      K_HO.SetAssemblyLevel(AssemblyLevel::PARTIAL);*/
   }

   M_HO.Assemble();
   K_HO.Assemble(0);

   if (pa == false)
   {
      M_HO.Finalize();
      K_HO.Finalize(0);
   }

   // Compute the lumped mass matrix.
   Vector lumpedM;
   ParBilinearForm ml(&pfes);
   ml.AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator));
   ml.Assemble();
   ml.Finalize();
   ml.SpMat().GetDiag(lumpedM);

   Mbf.Assemble();
   Mbf.Finalize();
   int skip_zeros = 0;
   Kbf.Assemble(skip_zeros);
   Kbf.Finalize(skip_zeros);

   // Store topological dof data.
   DofInfo dofs(pfes);

   // Precompute data required for high and low order schemes. This could be put
   // into a separate routine. I am using a struct now because the various
   // schemes require quite different information.
   LowOrderMethod lom;
   lom.subcell_scheme = use_subcell_RD;

   lom.pk = nullptr;
   if (lo_type == LOSolverType::DiscrUpwind)
   {
      assert(false);
      lom.smap = SparseMatrix_Build_smap(Kbf.SpMat());
      lom.D = Kbf.SpMat();

      if (exec_mode == 0)
      {
         ComputeDiscreteUpwindingMatrix(Kbf.SpMat(), lom.smap, lom.D);
      }
   }
   else if (lo_type == LOSolverType::DiscrUpwindPrec)
   {
      assert(false);
      lom.pk = new ParBilinearForm(&pfes);
      if (exec_mode == 0)
      {
         lom.pk->AddDomainIntegrator(
            new PrecondConvectionIntegrator(velocity, -1.0) );
      }
      else if (exec_mode == 1)
      {
         lom.pk->AddDomainIntegrator(
            new PrecondConvectionIntegrator(v_coef) );
      }
      lom.pk->Assemble(skip_zeros);
      lom.pk->Finalize(skip_zeros);

      lom.smap = SparseMatrix_Build_smap(lom.pk->SpMat());
      lom.D = lom.pk->SpMat();

      if (exec_mode == 0)
      {
         ComputeDiscreteUpwindingMatrix(lom.pk->SpMat(), lom.smap, lom.D);
      }
   }
   if (exec_mode == 1)
   {
      assert(false);
      lom.coef = &v_coef;
   }
   else
   {
      lom.coef = &velocity;
   }

   // Face integration rule.
   const FaceElementTransformations *ft =
      pmesh.GetFaceElementTransformations(0);
   const int el_order = pfes.GetFE(0)->GetOrder();
   int ft_order = ft->Elem1->OrderW() + 2 * el_order;
   if (pfes.GetFE(0)->Space() == FunctionSpace::Pk) { ft_order++; }
   lom.irF = &IntRules.Get(ft->FaceGeom, ft_order);

   DG_FECollection fec0(0, dim, btype);
   DG_FECollection fec1(1, dim, btype);

   ParMesh *subcell_mesh = nullptr;
   lom.SubFes0 = nullptr;
   lom.SubFes1 = nullptr;
   FiniteElementCollection *fec_sub = nullptr;
   ParFiniteElementSpace *pfes_sub = nullptr;;
   ParGridFunction *xsub = nullptr;
   ParGridFunction v_sub_gf;
   VectorGridFunctionCoefficient v_sub_coef;
   Vector x0_sub;

   if (order > 1)
   {
      // The mesh corresponding to Bezier subcells of order p is constructed.
      // NOTE: The mesh is assumed to consist of quads or hexes.
      MFEM_VERIFY(order > 1, "This code should not be entered for order = 1.");

      // Get a uniformly refined mesh.
      subcell_mesh = new ParMesh(&pmesh, order, BasisType::ClosedUniform);

      // Check if the mesh is periodic.
      const L2_FECollection *L2_coll = dynamic_cast<const L2_FECollection *>
                                       (pmesh.GetNodes()->FESpace()->FEColl());
      if (L2_coll == NULL)
      {
         // Standard non-periodic mesh.
         // Note that the fine mesh is always linear.
         fec_sub = new H1_FECollection(1, dim, BasisType::ClosedUniform);
         pfes_sub = new ParFiniteElementSpace(subcell_mesh, fec_sub, dim);
         xsub = new ParGridFunction(pfes_sub);
         subcell_mesh->SetCurvature(1);
         subcell_mesh->SetNodalGridFunction(xsub);
      }
      else
      {
         // Periodic mesh - the node positions must be corrected after the call
         // to the above Mesh constructor. Note that the fine mesh is always
         // linear.
         const bool disc_nodes = true;
         subcell_mesh->SetCurvature(1, disc_nodes);

         fec_sub = new L2_FECollection(1, dim, BasisType::ClosedUniform);
         pfes_sub = new ParFiniteElementSpace(subcell_mesh, fec_sub, dim);
         xsub = new ParGridFunction(pfes_sub);
         subcell_mesh->SetNodalGridFunction(xsub);

         GridFunction *coarse = pmesh.GetNodes();
         InterpolationGridTransfer transf(*coarse->FESpace(), *pfes_sub);
         transf.ForwardOperator().Mult(*coarse, *xsub);
      }

      lom.SubFes0 = new FiniteElementSpace(subcell_mesh, &fec0);
      lom.SubFes1 = new FiniteElementSpace(subcell_mesh, &fec1);

      // Submesh velocity.
      v_sub_gf.SetSpace(pfes_sub);
      v_sub_gf.ProjectCoefficient(velocity);

      // Zero it out on boundaries (not moving boundaries).
      ZeroItOutOnBoundaries(subcell_mesh, xsub, v_sub_gf, v_sub_coef);

      // Store initial submesh positions.
      x0_sub = *xsub;

      // Integrator on the submesh.
      if (exec_mode == 0)
      {
         lom.VolumeTerms = new MixedConvectionIntegrator(velocity, -1.0);
      }
      else if (exec_mode == 1)
      {
         lom.VolumeTerms = new MixedConvectionIntegrator(v_sub_coef);
      }
   }
   else { subcell_mesh = &pmesh; }

   Assembly asmbl(dofs, lom, inflow_gf, pfes, subcell_mesh, exec_mode);

   LOSolver *lo_solver = nullptr;
   //Array<int> lo_smap;
   //const bool time_dep = (exec_mode == 0) ? false : true;
   if (lo_type == LOSolverType::DiscrUpwind)
   {
      assert(false);
      /*lo_smap = SparseMatrix_Build_smap(Kbf.SpMat());
      lo_solver = new DiscreteUpwind(pfes, Kbf.SpMat(), lo_smap,
                                     lumpedM, asmbl, time_dep);*/
   }
   else if (lo_type == LOSolverType::DiscrUpwindPrec)
   {
      assert(false);
      /*lo_smap = SparseMatrix_Build_smap(lom.pk->SpMat());
      lo_solver = new DiscreteUpwind(pfes, lom.pk->SpMat(), lo_smap,
                                     lumpedM, asmbl, time_dep);*/
   }
   else if (lo_type == LOSolverType::ResDist)
   {
      assert(false);
      /*const bool subcell_scheme = false;
      lo_solver = new ResidualDistribution(pfes, Kbf, asmbl, lumpedM,
                                           subcell_scheme, time_dep);*/
   }
   else if (lo_type == LOSolverType::ResDistSubcell)
   {
      assert(false);
      /*const bool subcell_scheme = true;
      lo_solver = new ResidualDistribution(pfes, Kbf, asmbl, lumpedM,
                                           subcell_scheme, time_dep);*/
   }

   // Setup the initial conditions.
   const int vsize = pfes.GetVSize();
   Array<int> offset((product_sync) ? 3 : 2);
   for (int i = 0; i < offset.Size(); i++) { offset[i] = i*vsize; }
   BlockVector S(offset, Device::GetMemoryType());

   // Primary scalar field is u.
   ParGridFunction u(&pfes);
   u.MakeRef(&pfes, S, offset[0]);
   FunctionCoefficient u0(u0_function);
   u.ProjectCoefficient(u0);
   u.SyncAliasMemory(S);
   dbg("pmesh.GetNE: %d, u.Size: %d", pmesh.GetNE(), u.Size());

   // For the case of product remap, we also solve for s and u_s.
   ParGridFunction s, us;
   Array<bool> u_bool_el, u_bool_dofs;
   if (product_sync)
   {
      assert(false);/*
      s.SetSpace(&pfes);
      ComputeBoolIndicators(pmesh.GetNE(), u, u_bool_el, u_bool_dofs);
      BoolFunctionCoefficient sc(s0_function, u_bool_el);
      s.ProjectCoefficient(sc);

      us.MakeRef(&pfes, S, offset[1]);
      us.HostWrite();
      u.HostRead();
      s.HostRead();
      // Simple - we don't target conservation at initialization.
      for (int i = 0; i < s.Size(); i++) { us(i) = u(i) * s(i); }
      us.SyncAliasMemory(S);*/
   }

   // Smoothness indicator.
   SmoothnessIndicator *smth_indicator = nullptr;
   if (smth_ind_type)
   {
      assert(false);/*
      smth_indicator = new SmoothnessIndicator(smth_ind_type, *subcell_mesh,
                                               pfes, u, dofs);*/
   }

   // Setup of the high-order solver (if any).
   HOSolver *ho_solver = nullptr;
   if (ho_type == HOSolverType::Neumann)
   {
      assert(false);
      //ho_solver = new NeumannHOSolver(pfes, Mbf, Kbf, lumpedM, asmbl);
   }
   else if (ho_type == HOSolverType::CG)
   {
      assert(false);
      //ho_solver = new CGHOSolver(pfes, M_HO, K_HO);
   }
   else if (ho_type == HOSolverType::LocalInverse)
   {
      ho_solver = new LocalInverseHOSolver(pfes, M_HO, K_HO);
   }

   // Setup of the monolithic solver (if any).
   MonolithicSolver *mono_solver = nullptr;
   //bool mass_lim = (problem_num != 6 && problem_num != 7) ? true : false;
   if (mono_type == MonolithicSolverType::ResDistMono)
   {
      assert(false);/*
      const bool subcell_scheme = false;
      mono_solver = new MonoRDSolver(pfes, Kbf.SpMat(), Mbf.SpMat(), lumpedM,
                                     asmbl, smth_indicator, velocity,
                                     subcell_scheme, time_dep, mass_lim);*/
   }
   else if (mono_type == MonolithicSolverType::ResDistMonoSubcell)
   {
      assert(false);/*
      const bool subcell_scheme = true;
      mono_solver = new MonoRDSolver(pfes, Kbf.SpMat(), Mbf.SpMat(), lumpedM,
                                     asmbl, smth_indicator, velocity,
                                     subcell_scheme, time_dep, mass_lim);*/
   }

   // Print the starting meshes and initial condition.
   /*ofstream meshHO("meshHO_init.mesh");
   meshHO.precision(precision);
   pmesh.PrintAsOne(meshHO);
   if (subcell_mesh)
   {
      ofstream meshLO("meshLO_init.mesh");
      meshLO.precision(precision);
      subcell_mesh->PrintAsOne(meshLO);
   }
   ofstream sltn("sltn_init.gf");
   sltn.precision(precision);
   u.SaveAsOne(sltn);*/

   // Create data collection for solution output: either VisItDataCollection for
   // ASCII data files, or SidreDataCollection for binary data files.
   /*DataCollection *dc = NULL;
   if (visit)
   {
      dc = new VisItDataCollection("Remhos", &pmesh);
      dc->SetPrecision(precision);
      dc->RegisterField("solution", &u);
      dc->SetCycle(0);
      dc->SetTime(0.0);
      dc->Save();
   }*/

   socketstream sout, vis_s, vis_us;
   char vishost[] = "localhost";
   int  visport   = 19916;
   if (visualization)
   {
      // Make sure all MPI ranks have sent their 'v' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh.GetComm());

      sout.precision(8);
      vis_s.precision(8);
      vis_us.precision(8);

      u.HostRead();
      s.HostRead();
      VisualizeField(sout, vishost, visport,
                     u, "Solution u",
                     Wx, Wy, Ww, Wh, "gAmRj");
      if (product_sync)
      {
         VisualizeField(vis_s, vishost, visport, s, "Solution s",
                        Wx + Ww, Wy, Ww, Wh);
         VisualizeField(vis_us, vishost, visport, us, "Solution u_s",
                        Wx + 2*Ww, Wy, Ww, Wh);
      }
   }

   // Record the initial mass.
   MPI_Comm comm = pmesh.GetComm();
   Vector masses(lumpedM);
   const double mass0_u_loc = lumpedM * u;
   double mass0_u;//, mass0_us;
   MPI_Allreduce(&mass0_u_loc, &mass0_u, 1, MPI_DOUBLE, MPI_SUM, comm);
   if (product_sync)
   {
      assert(false);
      /*const double mass0_us_loc = lumpedM * us;
      MPI_Allreduce(&mass0_us_loc, &mass0_us, 1, MPI_DOUBLE, MPI_SUM, comm);*/
   }

   {
      double mass_u, mass_u_loc = masses * u;
      MPI_Allreduce(&mass_u_loc, &mass_u, 1, MPI_DOUBLE, MPI_SUM, comm);
      if (myid == 0)
      {
         std::cout << setprecision(10)
                   << "Initial mass u: " << mass_u << std::endl
                   << "   Mass loss u: " << abs(mass0_u - mass_u) << std::endl;
      }
   }

   // Setup of the FCT solver (if any).
   Array<int> K_HO_smap;
   FCTSolver *fct_solver = nullptr;
   if (fct_type == FCTSolverType::FluxBased)
   {
      assert(false);/*
      MFEM_VERIFY(pa == false, "Flux-based FCT and PA are incompatible.");

      K_HO_smap = SparseMatrix_Build_smap(K_HO.SpMat());
      const int fct_iterations = 1;
      fct_solver = new FluxBasedFCT(pfes, smth_indicator, dt, K_HO.SpMat(),
                                    K_HO_smap, M_HO.SpMat(), fct_iterations);*/
   }
   else if (fct_type == FCTSolverType::ClipScale)
   {
      assert(false);/*
      fct_solver = new ClipScaleSolver(pfes, smth_indicator, dt);*/
   }
   else if (fct_type == FCTSolverType::NonlinearPenalty)
   {
      assert(false);/*
      fct_solver = new NonlinearPenaltySolver(pfes, smth_indicator, dt);*/
   }

   AdvectionOperator adv(S.Size(), Mbf, ml, lumpedM, Kbf, M_HO, K_HO,
                         x, xsub, v_gf, v_sub_gf, asmbl, lom, dofs,
                         ho_solver, lo_solver, fct_solver, mono_solver);

   double t = 0.0;
   adv.SetTime(t);
   ode_solver->Init(adv);

   double umin, umax;
   GetMinMax(u, umin, umax);

   if (exec_mode == 1)
   {
      assert(false);/*
      adv.SetRemapStartPos(x0, x0_sub);

      // For remap, the pseudo-time always evolves from 0 to 1.
      t_final = 1.0;*/
   }

   ParGridFunction res = u;
   double residual = 0.0;

   // AMR operator
   amr::Operator *AMR = nullptr;
   if (amr) { AMR = new amr::Operator(pfes, pmesh, u, amr_options); }

   // Time-integration (loop over the time iterations, ti, with a time-step dt).
   bool done = false;
   int depth = amr ? GetMeshDepth(pmesh) : 0;
   for (int ti = 0; !done;)
   {
      dbg("\033[31m###########################");
      dbg("\033[31m######## TIME LOOP ########");
      dbg("\033[31m######## dt:%f ########\033[m",dt);
      double dt_real = min(dt, t_final - t);

      adv.SetDt(dt_real);

      // 13. The inner refinement loop. At the end we want to have the current
      //     time step resolved to the prescribed tolerance in each element.
      if (amr)
      {
         AMR->Reset();
         dbg("\t\033[33m##########################");
         dbg("\t\033[33m######## AMR LOOP ########");
         for (int ref_it = 1; ; ref_it++)
         {
            const int new_depth = mfem::GetMeshDepth(pmesh);
            // problem 2 needs at least this factor
            const double factor = 2.0;
            if (new_depth > depth) { dt /= factor; }
            if (new_depth < depth) { dt *= factor; }
            if (new_depth != depth)
            {
               double dt_real = min(dt, t_final - t);
               adv.SetDt(dt_real);
               depth = new_depth;
               if (myid == 0)
               {
                  std::cout << "time step: " << ti
                            << ", time: " << t
                            << ", dt: " << dt;
                  std::cout << endl;
               }
            }
            //pmesh.pncmesh->PrintStats(std::cout);

            dbg("AMR->Apply");
            AMR->Apply();

            if (!AMR->Refined()) { dbg("Refines STOP!");  break; }

            dbg("AMR->Update");
            AMR->Update(adv, ode_solver, S, offset,
                        lom, subcell_mesh, pfes_sub, xsub, v_sub_gf,
                        lumpedM, mass0_u, inflow_gf, inflow);
            /*if (visualization)
            {
               MPI_Barrier(pmesh.GetComm());
               u.HostRead();
               VisualizeField(sout, vishost, visport, u, "Solution",
                              Wx, Wy, Ww, Wh);
            }*/
         }
         /*if (myid == 0)
         {
            ml.SpMat().GetDiag(lumpedM);
            masses = lumpedM;
            double mass_u, mass_u_loc = masses * u;
            MPI_Allreduce(&mass_u_loc, &mass_u, 1, MPI_DOUBLE, MPI_SUM, comm);
            std::cout << setprecision(10)
                      << "Before DEREFINMENT, u size:" << u.Size() << std::endl
                      << "Current mass u: " << mass_u << std::endl
                      << "   Mass loss u: " << abs(mass0_u - mass_u) << std::endl;
         }*/
         assert (!AMR->Refined());
         if (AMR->DeRefined())
         {
            dbg("DEREFINE, AMR->Update");
            AMR->Update(adv, ode_solver, S, offset,
                        lom, subcell_mesh, pfes_sub, xsub, v_sub_gf,
                        lumpedM, mass0_u, inflow_gf, inflow);
            /*if (visualization)
            {
               MPI_Barrier(pmesh.GetComm());
               u.HostRead();
               VisualizeField(sout, vishost, visport, u, "Solution",
                              Wx, Wy, Ww, Wh);
            }*/
            //assert(false);
         }
      }

      ode_solver->Step(S, t, dt_real);
      ti++;

      //S has been modified, update the alias
      u.SyncMemory(S);
      if (product_sync) { us.SyncMemory(S); }

      // Monotonicity check for debug purposes mainly.
      if (verify_bounds && forced_bounds && smth_indicator == NULL)
      {
         double umin_new, umax_new;
         GetMinMax(u, umin_new, umax_new);
         if (problem_num % 10 != 6 && problem_num % 10 != 7)
         {
            if (myid == 0)
            {
               MFEM_VERIFY(umin_new > umin - 1e-12,
                           "Undershoot of " << umin - umin_new);
               MFEM_VERIFY(umax_new < umax + 1e-12,
                           "Overshoot of " << umax_new - umax);
            }
            umin = umin_new;
            umax = umax_new;
         }
         else
         {
            if (myid == 0)
            {
               MFEM_VERIFY(umin_new > 0.0 - 1e-12,
                           "Undershoot of " << 0.0 - umin_new);
               MFEM_VERIFY(umax_new < 1.0 + 1e-12,
                           "Overshoot of " << umax_new - 1.0);
            }
         }
      }

      if (exec_mode == 1)
      {
         assert(false);/*
         add(x0, t, v_gf, x);
         add(x0_sub, t, v_sub_gf, *xsub);*/
      }

      const bool steady_state_problem =
         problem_num == 6 || problem_num == 7 || problem_num == 8;

      if (!steady_state_problem)
      {
         done = (t >= t_final - 1.e-8*dt);
         dbg("done: %s", done?"yes":"no");
      }
      else
      {
         assert(false);/*
         dbg("Steady state problems");
         // Steady state problems - stop at convergence.
         double res_loc = 0.;
         lumpedM.HostReadWrite(); u.HostReadWrite(); res.HostReadWrite();
         for (int i = 0; i < res.Size(); i++)
         {
            res_loc += pow( (lumpedM(i) * u(i) / dt) - (lumpedM(i) * res(i) / dt), 2. );
         }
         MPI_Allreduce(&res_loc, &residual, 1, MPI_DOUBLE, MPI_SUM, comm);

         residual = sqrt(residual);
         if (residual < 1.e-12 && t >= 1.) { done = true; u = res; }
         else { res = u; }*/
      }

      if (ti == max_tsteps) { dbg("max_tsteps reached!"); done = true; }

      if (done || ti % vis_steps == 0)
      {
         if (myid == 0)
         {
            cout << "time step: " << ti << ", time: " << t << ", dt: " << dt;
            if (steady_state_problem) { cout << ", residual: " << residual ; }
            cout << endl;
         }

         if (visualization)
         {
            MPI_Barrier(pmesh.GetComm());
            u.HostRead();

            VisualizeField(sout, vishost, visport, u, "Solution",
                           Wx, Wy, Ww, Wh);
            if (product_sync)
            {
               // Recompute s = u_s / u.
               ComputeRatio(pmesh.GetNE(), us, u, s, u_bool_el, u_bool_dofs);
               VisualizeField(vis_s, vishost, visport, s, "Solution s",
                              Wx + Ww, Wy, Ww, Wh);
               VisualizeField(vis_us, vishost, visport, us, "Solution u_s",
                              Wx + 2*Ww, Wy, Ww, Wh);
            }
         }

         /*if (visit)
         {
            dc->SetCycle(ti);
            dc->SetTime(t);
            dc->Save();
         }*/
      }
   }

   // Print the final meshes and solution.
   {
      ofstream meshHO("meshHO_final.mesh");
      meshHO.precision(precision);
      pmesh.PrintAsOne(meshHO);
      if (subcell_mesh)
      {
         ofstream meshLO("meshLO_final.mesh");
         meshLO.precision(precision);
         subcell_mesh->PrintAsOne(meshLO);
      }
      ofstream sltn("sltn_final.gf");
      sltn.precision(precision);
      u.SaveAsOne(sltn);
   }

   // Check for mass conservation.
   double mass_u_loc = 0.0, mass_us_loc = 0.0;
   if (exec_mode == 1)
   {
      ml.BilinearForm::operator=(0.0);
      ml.Assemble();
      lumpedM.HostRead();
      ml.SpMat().GetDiag(lumpedM);
      mass_u_loc = lumpedM * u;
      if (product_sync) { mass_us_loc = lumpedM * us; }
   }
   else
   {
      ml.SpMat().GetDiag(lumpedM);
      masses = lumpedM;
      mass_u_loc = masses * u;
      if (product_sync) { mass_us_loc = masses * us; }
   }
   double mass_u, mass_us, s_max;
   MPI_Allreduce(&mass_u_loc, &mass_u, 1, MPI_DOUBLE, MPI_SUM, comm);
   const double umax_loc = u.Max();
   MPI_Allreduce(&umax_loc, &umax, 1, MPI_DOUBLE, MPI_MAX, comm);
   if (product_sync)
   {
      ComputeRatio(pmesh.GetNE(), us, u, s, u_bool_el, u_bool_dofs);
      const double s_max_loc = s.Max();
      MPI_Allreduce(&mass_us_loc, &mass_us, 1, MPI_DOUBLE, MPI_SUM, comm);
      MPI_Allreduce(&s_max_loc, &s_max, 1, MPI_DOUBLE, MPI_MAX, comm);
   }
   if (myid == 0)
   {
      cout << setprecision(10)
           << "Final mass u:  " << mass_u << endl
           << "Max value u:   " << umax << endl << setprecision(6)
           << "Mass loss u:   " << abs(mass0_u - mass_u) << endl;
      if (product_sync)
      {
         /*
           cout << setprecision(10)
                << "Final mass us: " << mass_us << endl
                << "Max value s:   " << s_max << endl << setprecision(6)
                << "Mass loss us:  " << abs(mass0_us - mass_us) << endl;*/
      }
   }

   // Compute errors, if the initial condition is equal to the final solution
   if (problem_num == 4) // solid body rotation
   {
      double err = u.ComputeLpError(1., u0);
      if (myid == 0) { cout << "L1-error: " << err << "." << endl; }
   }
   else if (problem_num == 7)
   {
      FunctionCoefficient u_ex(inflow_function);
      double e1 = u.ComputeLpError(1., u_ex);
      double e2 = u.ComputeLpError(2., u_ex);
      double eInf = u.ComputeLpError(numeric_limits<double>::infinity(), u_ex);
      if (myid == 0)
      {
         cout << "L1-error: " << e1 << "." << endl;

         // write output
         ofstream file("errors.txt", ios_base::app);

         if (!file)
         {
            MFEM_ABORT("Error opening file.");
         }
         else
         {
            ostringstream strs;
            strs << e1 << " " << e2 << " " << eInf << "\n";
            string str = strs.str();
            file << str;
            file.close();
         }
      }
   }

   if (smth_indicator)
   {
      // Print the values of the smoothness indicator.
      ParGridFunction si_val;
      smth_indicator->ComputeSmoothnessIndicator(u, si_val);
      {
         ofstream smth("si_final.gf");
         smth.precision(precision);
         si_val.SaveAsOne(smth);
      }
   }

   // Free the used memory.
   delete mono_solver;
   delete fct_solver;
   delete smth_indicator;
   delete ho_solver;

   delete ode_solver;
   delete mesh_fec;
   delete lom.pk;
   //delete dc;

   if (order > 1)
   {
      delete subcell_mesh;
      delete fec_sub;
      delete pfes_sub;
      delete xsub;
      delete lom.SubFes0;
      delete lom.SubFes1;
      delete lom.VolumeTerms;
   }

   return 0;
}

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   int ProbExec = problem_num % 20;

   switch (ProbExec)
   {
      case 0:
      {
         // Translations in 1D, 2D, and 3D
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = sqrt(2./3.); v(1) = sqrt(1./3.); break;
            case 3: v(0) = sqrt(3./6.); v(1) = sqrt(2./6.); v(2) = sqrt(1./6.);
               break;
         }
         break;
      }
      case 1:
      case 2:
      case 4:
      {
         // Clockwise rotation in 2D around the origin
         const double w = M_PI/2;
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = -w*X(1); v(1) = w*X(0); break;
            case 3: v(0) = -w*X(1); v(1) = w*X(0); v(2) = 0.0; break;
         }
         break;
      }
      case 3:
      {
         // Clockwise twisting rotation in 2D around the origin
         const double w = M_PI/2;
         double d = max((X(0)+1.)*(1.-X(0)),0.) * max((X(1)+1.)*(1.-X(1)),0.);
         d = d*d;
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = d*w*X(1); v(1) = -d*w*X(0); break;
            case 3: v(0) = d*w*X(1); v(1) = -d*w*X(0); v(2) = 0.0; break;
         }
         break;
      }
      case 5:
      {
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = 1.0; v(1) = 1.0; break;
            case 3: v(0) = 1.0; v(1) = 1.0; v(2) = 1.0; break;
         }
         break;
      }
      case 6:
      case 7:
      {
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = x(1); v(1) = -x(0); break;
            case 3: v(0) = x(1); v(1) = -x(0); v(2) = 0.0; break;
         }
         break;
      }
      case 11:
      {
         // Gresho deformation used for mesh motion in remap tests.
         const double r = sqrt(x(0)*x(0) + x(1)*x(1));
         if (r < 0.2)
         {
            v(0) =  5.0 * x(1);
            v(1) = -5.0 * x(0);
         }
         else if (r < 0.4)
         {
            v(0) =  2.0 * x(1) / r - 5.0 * x(1);
            v(1) = -2.0 * x(0) / r + 5.0 * x(0);
         }
         else { v = 0.0; }
         break;
      }
      case 10:
      case 12:
      case 13:
      case 14:
      case 15:
      case 16:
      case 17:
      {
         // Taylor-Green deformation used for mesh motion in remap tests.

         // Map [-1,1] to [0,1].
         for (int d = 0; d < dim; d++) { X(d) = X(d) * 0.5 + 0.5; }

         if (dim == 1) { MFEM_ABORT("Not implemented."); }
         v(0) =  sin(M_PI*X(0)) * cos(M_PI*X(1));
         v(1) = -cos(M_PI*X(0)) * sin(M_PI*X(1));
         if (dim == 3)
         {
            v(0) *= cos(M_PI*X(2));
            v(1) *= cos(M_PI*X(2));
            v(2) = 0.0;
         }
         break;
      }
   }
}

double box(std::pair<double,double> p1, std::pair<double,double> p2,
           double theta,
           std::pair<double,double> origin, double x, double y)
{
   double xmin=p1.first;
   double xmax=p2.first;
   double ymin=p1.second;
   double ymax=p2.second;
   double ox=origin.first;
   double oy=origin.second;

   double pi = M_PI;
   double s=std::sin(theta*pi/180);
   double c=std::cos(theta*pi/180);

   double xn=c*(x-ox)-s*(y-oy)+ox;
   double yn=s*(x-ox)+c*(y-oy)+oy;

   if (xn>xmin && xn<xmax && yn>ymin && yn<ymax)
   {
      return 1.0;
   }
   else
   {
      return 0.0;
   }
}

double box3D(double xmin, double xmax, double ymin, double ymax, double zmin,
             double zmax, double theta, double ox, double oy, double x,
             double y, double z)
{
   double pi = M_PI;
   double s=std::sin(theta*pi/180);
   double c=std::cos(theta*pi/180);

   double xn=c*(x-ox)-s*(y-oy)+ox;
   double yn=s*(x-ox)+c*(y-oy)+oy;

   if (xn>xmin && xn<xmax && yn>ymin && yn<ymax && z>zmin && z<zmax)
   {
      return 1.0;
   }
   else
   {
      return 0.0;
   }
}

double get_cross(double rect1, double rect2)
{
   double intersection=rect1*rect2;
   return rect1+rect2-intersection; //union
}

double ring(double rin, double rout, Vector c, Vector y)
{
   double r = 0.;
   int dim = c.Size();
   if (dim != y.Size())
   {
      mfem_error("Origin vector and variable have to be of the same size.");
   }
   for (int i = 0; i < dim; i++)
   {
      r += pow(y(i)-c(i), 2.);
   }
   r = sqrt(r);
   if (r>rin && r<rout)
   {
      return 1.0;
   }
   else
   {
      return 0.0;
   }
}

// Initial condition: lua function or hard-coded functions
double u0_function(const Vector &x)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   int ProbExec = problem_num % 10;

   switch (ProbExec)
   {
      case 0:
      case 1:
      {
         switch (dim)
         {
            case 1:
               return exp(-40.*pow(X(0)-0.5,2));
            case 2:
            case 3:
            {
               double rx = 0.45, ry = 0.25, cx = 0., cy = -0.2, w = 10.;
               if (dim == 3)
               {
                  const double s = (1. + 0.25*cos(2*M_PI*X(2)));
                  rx *= s;
                  ry *= s;
               }
               return ( erfc(w*(X(0)-cx-rx))*erfc(-w*(X(0)-cx+rx)) *
                        erfc(w*(X(1)-cy-ry))*erfc(-w*(X(1)-cy+ry)) )/16;
            }
         }
      }
      case 2:
      {
         double x_ = X(0), y_ = X(1), rho, phi;
         rho = hypot(x_, y_);
         phi = atan2(y_, x_);
         return pow(sin(M_PI*rho),2)*sin(3*phi);
      }
      case 3:
      {
         return .5*(sin(M_PI*X(0))*sin(M_PI*X(1)) + 1.);
      }
      case 4:
      {
         double scale = 0.0225;
         double coef = (0.5/sqrt(scale));
         double slit = (X(0) <= -0.05) || (X(0) >= 0.05) || (X(1) >= 0.7);
         double cone = coef * sqrt(pow(X(0), 2.) + pow(X(1) + 0.5, 2.));
         double hump = coef * sqrt(pow(X(0) + 0.5, 2.) + pow(X(1), 2.));

         return (slit && ((pow(X(0),2.) + pow(X(1)-.5,2.))<=4.*scale)) ? 1. : 0.
                + (1. - cone) * (pow(X(0), 2.) + pow(X(1)+.5, 2.) <= 4.*scale)
                + .25 * (1. + cos(M_PI*hump))
                * ((pow(X(0)+.5, 2.) + pow(X(1), 2.)) <= 4.*scale);
      }
      case 5:
      {
         Vector y(dim);
         for (int i = 0; i < dim; i++) { y(i) = 50. * (x(i) + 1.); }

         if (dim==1)
         {
            mfem_error("This test is not supported in 1D.");
         }
         else if (dim==2)
         {
            std::pair<double, double> p1;
            std::pair<double, double> p2;
            std::pair<double, double> origin;

            // cross
            p1.first=14.; p1.second=3.;
            p2.first=17.; p2.second=26.;
            origin.first = 15.5;
            origin.second = 11.5;
            double rect1=box(p1,p2,-45.,origin,y(0),y(1));
            p1.first=7.; p1.second=10.;
            p2.first=32.; p2.second=13.;
            double rect2=box(p1,p2,-45.,origin,y(0),y(1));
            double cross=get_cross(rect1,rect2);
            // rings
            Vector c(dim);
            c(0) = 40.; c(1) = 40;
            double ring1 = ring(7., 10., c, y);
            c(1) = 20.;
            double ring2 = ring(3., 7., c, y);

            return cross + ring1 + ring2;
         }
         else
         {
            // cross
            double rect1 = box3D(7.,32.,10.,13.,10.,13.,-45.,15.5,11.5,
                                 y(0),y(1),y(2));
            double rect2 = box3D(14.,17.,3.,26.,10.,13.,-45.,15.5,11.5,
                                 y(0),y(1),y(2));
            double rect3 = box3D(14.,17.,10.,13.,3.,26.,-45.,15.5,11.5,
                                 y(0),y(1),y(2));

            double cross = get_cross(get_cross(rect1, rect2), rect3);

            // rings
            Vector c1(dim), c2(dim);
            c1(0) = 40.; c1(1) = 40; c1(2) = 40.;
            c2(0) = 40.; c2(1) = 20; c2(2) = 20.;

            double shell1 = ring(7., 10., c1, y);
            double shell2 = ring(3., 7., c2, y);

            double dom2 = cross + shell1 + shell2;

            // cross
            rect1 = box3D(2.,27.,30.,33.,30.,33.,0.,0.,0.,y(0),y(1),y(2));
            rect2 = box3D(9.,12.,23.,46.,30.,33.,0.,0.,0.,y(0),y(1),y(2));
            rect3 = box3D(9.,12.,30.,33.,23.,46.,0.,0.,0.,y(0),y(1),y(2));

            cross = get_cross(get_cross(rect1, rect2), rect3);

            double ball1 = ring(0., 7., c1, y);
            double ball2 = ring(0., 3., c2, y);
            double shell3 = ring(7., 10., c2, y);

            double dom3 = cross + ball1 + ball2 + shell3;

            double dom1 = 1. - get_cross(dom2, dom3);

            return dom1 + 2.*dom2 + 3.*dom3;
         }
      }
      case 6:
      {
         double r = x.Norml2();
         if (r >= 0.15 && r < 0.45) { return 1.; }
         else if (r >= 0.55 && r < 0.85)
         {
            return pow(cos(10.*M_PI * (r - 0.7) / 3.), 2.);
         }
         else { return 0.; }
      }
      case 7:
      {
         double r = x.Norml2();
         double a = 0.5, b = 3.e-2, c = 0.1;
         return 0.25*(1.+tanh((r+c-a)/b))*(1.-tanh((r-c-a)/b));
      }
   }
   return 0.0;
}

double s0_function(const Vector &x)
{
   // Simple nonlinear function.
   return 2.0 + sin(2*M_PI * x(0)) * sin(2*M_PI * x(1));
}

double inflow_function(const Vector &x)
{
   double r = x.Norml2();
   if ((problem_num % 10) == 6 && x.Size() == 2)
   {
      if (r >= 0.15 && r < 0.45) { return 1.; }
      else if (r >= 0.55 && r < 0.85)
      {
         return pow(cos(10.*M_PI * (r - 0.7) / 3.), 2.);
      }
      else { return 0.; }
   }
   else if ((problem_num % 10) == 7)
   {
      double a = 0.5, b = 3.e-2, c = 0.1;
      return 0.25*(1.+tanh((r+c-a)/b))*(1.-tanh((r-c-a)/b));
   }
   else { return 0.0; }
}
