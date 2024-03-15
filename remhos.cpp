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

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "remhos_ho.hpp"
#include "remhos_lo.hpp"
#include "remhos_fct.hpp"
#include "remhos_mono.hpp"
#include "remhos_tools.hpp"
#include "remhos_sync.hpp"

using namespace std;
using namespace mfem;

enum class HOSolverType {None, Neumann, CG, LocalInverse};
enum class FCTSolverType {None, FluxBased, ClipScale,
                          NonlinearPenalty, FCTProject};
enum class LOSolverType {None,    DiscrUpwind,    DiscrUpwindPrec,
                         ResDist, ResDistSubcell, MassBased};
enum class MonolithicSolverType {None, ResDistMono, ResDistMonoSubcell};

enum class TimeStepControl {FixedTimeStep, LOBoundsError};

// Choice for the problem setup. The fluid velocity, initial condition and
// inflow boundary condition are chosen based on this parameter.
int problem_num;

// 0 is standard transport.
// 1 is standard remap (mesh moves, solution is fixed).
int exec_mode;

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v);

// Initial condition
double u0_function(const Vector &x);
double s0_function(const Vector &x);
double q0_function(const Vector &x);

// Inflow boundary condition
double inflow_function(const Vector &x);

void blend_global(const Vector &u_b, const Vector &u_s, const Vector &m,
                  const Vector &u_min, const Vector &u_max, Vector &u);
void sharp_product_sync(const Vector &u, const Vector &m,
                        const Vector &s_min, const Vector &s_max,
                        const Vector &us_b, const Array<bool> &active_dofs,
                        Vector &us);

// Mesh bounding box
Vector bb_min, bb_max;

class AdvectionOperator : public TimeDependentOperator
{
private:
   BilinearForm &ml;
   ParBilinearForm &K_sharp, &K_bound;
   ParBilinearForm &M_HO;
   Vector &lumpedM;

   Vector start_mesh_pos, start_submesh_pos;
   GridFunction &mesh_pos, *submesh_pos, &mesh_vel, &submesh_vel;

   mutable ParGridFunction x_gf;
   mutable Vector s_old, q_old;

   double dt;
   TimeStepControl dt_control;
   mutable double dt_est;

   DofInfo &dofs;

   HOSolver *ho_solver_b, *ho_solver_s;
   LOSolver *lo_solver_b, *lo_solver_s;
   FCTSolver *fct_solver_b, *fct_solver_s;
   MonolithicSolver *mono_solver;

   void UpdateTimeStepEstimate(const Vector &x, const Vector &dx,
                               const Vector &x_min, const Vector &x_max) const;

public:
   AdvectionOperator(int size, BilinearForm &_ml,
                     Vector &_lumpedM,
                     ParBilinearForm &K_s, ParBilinearForm &K_b,
                     ParBilinearForm &M_HO_,
                     GridFunction &pos, GridFunction *sub_pos,
                     GridFunction &vel, GridFunction &sub_vel,
                     DofInfo &_dofs,
                     HOSolver *hos_b, HOSolver *hos_s,
                     LOSolver *los_b, LOSolver *los_s,
                     FCTSolver *fct_b, FCTSolver *fct_s,
                     MonolithicSolver *mos);

   bool evolve_sharp = false;

   virtual void Mult(const Vector &x, Vector &y) const;

   void SetTimeStepControl(TimeStepControl tsc)
   {
      if (tsc == TimeStepControl::LOBoundsError)
      {
         MFEM_VERIFY(lo_solver_b,
                     "The selected time step control requires a LO solver.");
      }
      dt_control = tsc;
   }
   void SetDt(double _dt)
   {
      dt = _dt;
      dt_est = std::numeric_limits<double>::infinity();
   }
   double GetTimeStepEstimate() { return dt_est; }

   void SetRemapStartPos(const Vector &m_pos, const Vector &sm_pos)
   {
      start_mesh_pos    = m_pos;
      start_submesh_pos = sm_pos;
   }

   void SetInitialS(Vector &s) { s_old = s; }
   void SetInitialQ(Vector &q) { q_old = q; }

   virtual ~AdvectionOperator() { }
};

int main(int argc, char *argv[])
{
   // Initialize MPI.
   mfem::MPI_Session mpi(argc, argv);
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
   int bounds_type = 0;
   bool sharp = false;
   bool pa = false;
   bool next_gen_full = false;
   int smth_ind_type = 0;
   double t_final = 4.0;
   TimeStepControl dt_control = TimeStepControl::FixedTimeStep;
   double dt = 0.005;
   bool visualization = true;
   bool visit = false;
   bool verify_bounds = false;
   int levels = 1;
   int vis_steps = 100;
   const char *device_config = "cpu";

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
                  "                  4 - Subcell Residual Distribution,\n\t"
                  "                  5 - Mass-Based Element Average.");
   args.AddOption((int*)(&fct_type), "-fct", "--fct-type",
                  "Correction type: 0 - No nonlinear correction,\n\t"
                  "                 1 - Flux-based FCT,\n\t"
                  "                 2 - Local clip + scale,\n\t"
                  "                 3 - Local clip + nonlinear penalization.");
   args.AddOption((int*)(&mono_type), "-mono", "--mono-type",
                  "Monolithic solver: 0 - No monolithic solver,\n\t"
                  "                   1 - Residual Distribution,\n\t"
                  "                   2 - Subcell Residual Distribution.");
   args.AddOption(&bounds_type, "-bt", "--bounds-type",
                  "Bounds stencil type: 0 - overlapping elements,\n\t"
                  "                     1 - matrix sparsity pattern.");
   args.AddOption(&sharp, "-sharp", "--sharp", "-no-sharp", "--no-sharp",
                  "Enable or disable profile sharpening.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly",
                  "Enable or disable partial assembly for the HO solution.");
   args.AddOption(&next_gen_full, "-full", "--next-gen-full", "-no-full",
                  "--no-next-gen-full",
                  "Enable or disable next gen full assembly for the HO solution.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&smth_ind_type, "-si", "--smth_ind",
                  "Smoothness indicator: 0 - no smoothness indicator,\n\t"
                  "                      1 - approx_quadratic,\n\t"
                  "                      2 - exact_quadratic.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption((int*)(&dt_control), "-dtc", "--dt-control",
                  "Time Step Control: 0 - Fixed time step, set with -dt,\n\t"
                  "                   1 - Bounds violation of the LO sltn.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Initial time step size (dt might change based on -dtc).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&verify_bounds, "-vb", "--verify-bounds", "-no-vb",
                  "--no-verify-bounds",
                  "Verify solution bounds after each time step.");
   args.AddOption(&levels, "-lev", "--product-levels",
                  "Levels to synchronize (1 u / 2 us / 3 usq).");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
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

   // Read the serial mesh from the given mesh file on all processors.
   // Refine the mesh in serial to increase the resolution.
   Mesh *mesh = new Mesh(Mesh::LoadFromFile(mesh_file, 1, 1));
   const int dim = mesh->Dimension();
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   mesh->GetBoundingBox(bb_min, bb_max, max(order, 1));

   // Only standard assembly in 1D (some mfem functions just abort in 1D).
   if ((pa || next_gen_full) && dim == 1)
   {
      MFEM_WARNING("Disabling PA / FA for 1D.");
      pa = false;
      next_gen_full = false;
   }

   // Parallel partitioning of the mesh.
   // Refine the mesh further in parallel to increase the resolution.
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < rp_levels; lev++) { pmesh.UniformRefinement(); }
   MPI_Comm comm = pmesh.GetComm();
   const int NE  = pmesh.GetNE();

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

   FiniteElementCollection *mesh_fec;
   if (periodic)
   {
      mesh_fec = new L2_FECollection(mesh_order, dim, BasisType::GaussLobatto);
   }
   else
   {
      mesh_fec = new H1_FECollection(mesh_order, dim, BasisType::GaussLobatto);
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

   // Initial time step estimate (CFL-based).
   if (dt < 0.0)
   {
      dt = std::numeric_limits<double>::infinity();
      Vector vel_e(dim);
      for (int e = 0; e < NE; e++)
      {
         double length_e = pmesh.GetElementSize(e);
         auto Tr = pmesh.GetElementTransformation(e);
         auto ip = Geometries.GetCenter(pmesh.GetElementBaseGeometry(e));
         Tr->SetIntPoint(&ip);
         velocity.Eval(vel_e, *Tr, ip);
         double speed_e = sqrt(vel_e * vel_e + 1e-14);
         dt = fmin(dt, 0.25 * length_e / speed_e);
      }
      MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MIN, comm);
   }

   // Mesh velocity.
   // If remap is on, obtain the mesh velocity by moving the mesh to the final
   // mesh positions, and taking the displacement vector.
   // The mesh motion resembles a time-dependent deformation, e.g., similar to
   // a deformation that is obtained by a Lagrangian simulation.
   GridFunction v_gf(x.FESpace());
   VectorGridFunctionCoefficient v_mesh_coeff(&v_gf);
   if (exec_mode == 1)
   {
      ParGridFunction v(&mesh_pfes);
      v.ProjectCoefficient(velocity);

      double t = 0.0;
      while (t < t_final)
      {
         t += dt;
         // Move the mesh nodes.
         x.Add(std::min(dt, t_final-t), v);
         // Update the node velocities.
         v.ProjectCoefficient(velocity);
      }

      // Pseudotime velocity.
      add(x, -1.0, x0, v_gf);

      // Return the mesh to the initial configuration.
      x = x0;
   }

   H1_FECollection lin_fec(1, dim);
   ParFiniteElementSpace lin_pfes(&pmesh, &lin_fec),
                         lin_pfes_grad(&pmesh, &lin_fec, dim);
   ParGridFunction u_max_bounds(&lin_pfes);
   ParGridFunction u_max_bounds_grad_dir(&lin_pfes_grad);
   u_max_bounds = 1.0;

   ParFiniteElementSpace lin_vec_pfes(&pmesh, &lin_fec, dim);
   ParGridFunction v_new_vis(&lin_vec_pfes);

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
      }
   }
   const bool use_subcell_RD =
      ( lo_type   == LOSolverType::ResDistSubcell ||
        mono_type == MonolithicSolverType::ResDistMonoSubcell );
   if (use_subcell_RD)
   {
      MFEM_VERIFY(order > 1, "Subcell schemes require FE order > 2.");
   }
   if (dt_control == TimeStepControl::LOBoundsError)
   {
      MFEM_VERIFY(bounds_type == 1, "Error: -dtc 1 requires -bt 1.");
   }

   const int prob_size = pfes.GlobalTrueVSize();
   if (myid == 0) { cout << "Number of unknowns: " << prob_size << endl; }

   // Fields related to inflow BC.
   FunctionCoefficient inflow(inflow_function);
   ParGridFunction inflow_gf(&pfes);
   if (problem_num == 7) // Convergence test: use high order projection.
   {
      L2_FECollection l2_fec(order, dim);
      ParFiniteElementSpace l2_fes(&pmesh, &l2_fec);
      ParGridFunction l2_inflow(&l2_fes);
      l2_inflow.ProjectCoefficient(inflow);
      inflow_gf.ProjectGridFunction(l2_inflow);
   }
   else { inflow_gf.ProjectCoefficient(inflow); }

   // Set up the bilinear and linear forms corresponding to the DG
   // discretization.
   ParBilinearForm M_HO(&pfes);
   M_HO.AddDomainIntegrator(new MassIntegrator);

   VelocityCoefficient v_new_coeff_adv(velocity, u_max_bounds,
                                       u_max_bounds_grad_dir, 1.0, 0, false);
   VelocityCoefficient v_new_coeff_rem(v_mesh_coeff, u_max_bounds,
                                       u_max_bounds_grad_dir, 1.0, 1, false);

   VelocityCoefficient v_diff_coeff(v_mesh_coeff, u_max_bounds,
                                    u_max_bounds_grad_dir, 1.0, 1, true);

   ParBilinearForm K_bound(&pfes), K_sharp(&pfes);
   MFEM_VERIFY(exec_mode == 1, "remap only branch");
   K_bound.AddDomainIntegrator(new ConvectionIntegrator(v_mesh_coeff));
   K_sharp.AddDomainIntegrator(new ConvectionIntegrator(v_mesh_coeff));

   auto dgt_i = new DGTraceIntegrator(v_mesh_coeff, -1.0, -0.5);
   auto dgt_b = new DGTraceIntegrator(v_mesh_coeff, -1.0, -0.5);
   K_bound.AddInteriorFaceIntegrator(new TransposeIntegrator(dgt_i));
   K_bound.AddBdrFaceIntegrator(new TransposeIntegrator(dgt_b));
   auto dgt_ii = new DGTraceIntegrator(v_mesh_coeff, -1.0, -0.5);
   auto dgt_bb = new DGTraceIntegrator(v_mesh_coeff, -1.0, -0.5);
   K_sharp.AddInteriorFaceIntegrator(new TransposeIntegrator(dgt_ii));
   K_sharp.AddBdrFaceIntegrator(new TransposeIntegrator(dgt_bb));

   if (sharp == true)
   {
      auto ci  = new ConvectionIntegrator(v_diff_coeff, -1.0);
      auto dgt = new DGTraceIntegrator(v_diff_coeff, 1.0, -0.5);
      K_sharp.AddDomainIntegrator(new TransposeIntegrator(ci));
      K_sharp.AddInteriorFaceIntegrator(dgt);
   }

   K_bound.KeepNbrBlock(true);
   K_sharp.KeepNbrBlock(true);

   MFEM_VERIFY(pa == false, "no pa");

   M_HO.Assemble();
   K_bound.Assemble(0);
   K_sharp.Assemble(0);
   M_HO.Finalize();
   K_bound.Finalize(0);
   K_sharp.Finalize(0);

   // Compute the lumped mass matrix.
   Vector lumpedM;
   ParBilinearForm ml(&pfes);
   ml.AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator));
   ml.Assemble();
   ml.Finalize();
   ml.SpMat().GetDiag(lumpedM);

   int skip_zeros = 0;

   // Store topological dof data.
   DofInfo dof_info(pfes, bounds_type);

   // Precompute data required for high and low order schemes. This could be put
   // into a separate routine. I am using a struct now because the various
   // schemes require quite different information.
   LowOrderMethod lom;
   lom.subcell_scheme = use_subcell_RD;

   lom.pk = NULL;
   if (lo_type == LOSolverType::DiscrUpwind)
   {
      lom.smap = SparseMatrix_Build_smap(K_sharp.SpMat());
      lom.D = K_sharp.SpMat();

      if (exec_mode == 0)
      {
         ComputeDiscreteUpwindingMatrix(K_sharp.SpMat(), lom.smap, lom.D);
      }
   }
   else if (lo_type == LOSolverType::DiscrUpwindPrec)
   {
      lom.pk = new ParBilinearForm(&pfes);
      if (exec_mode == 0)
      {
         lom.pk->AddDomainIntegrator(
            new PrecondConvectionIntegrator(velocity, -1.0) );
      }
      else if (exec_mode == 1)
      {
         lom.pk->AddDomainIntegrator(
            new PrecondConvectionIntegrator(v_mesh_coeff) );
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
   if (exec_mode == 1) { lom.coef = &v_mesh_coeff; }
   else                { lom.coef = &velocity; }

   // Face integration rule.
   const FaceElementTransformations *ft =
      pmesh.GetFaceElementTransformations(0);
   const int el_order = pfes.GetFE(0)->GetOrder();
   int ft_order = ft->Elem1->OrderW() + 2 * el_order;
   if (pfes.GetFE(0)->Space() == FunctionSpace::Pk) { ft_order++; }
   lom.irF = &IntRules.Get(ft->FaceGeom, ft_order);

   DG_FECollection fec0(0, dim, btype);
   DG_FECollection fec1(1, dim, btype);

   ParMesh *subcell_mesh = NULL;
   lom.SubFes0 = NULL;
   lom.SubFes1 = NULL;
   FiniteElementCollection *fec_sub = NULL;
   ParFiniteElementSpace *pfes_sub = NULL;;
   ParGridFunction *xsub = NULL;
   ParGridFunction v_sub_gf;
   VectorGridFunctionCoefficient v_sub_coef;
   Vector x0_sub;

   if (order > 1)
   {
      // The mesh corresponding to Bezier subcells of order p is constructed.
      // NOTE: The mesh is assumed to consist of quads or hexes.
      MFEM_VERIFY(order > 1, "This code should not be entered for order = 1.");

      // Get a uniformly refined mesh.
      const int btype = BasisType::ClosedUniform;
      subcell_mesh = new ParMesh(ParMesh::MakeRefined(pmesh, order, btype));

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

      // Store initial submesh positions.
      x0_sub = *xsub;

      // Integrator on the submesh.
      if (exec_mode == 0)
      {
         lom.subcellCoeff = &velocity;
         lom.VolumeTerms = new MixedConvectionIntegrator(velocity, -1.0);
      }
      else if (exec_mode == 1)
      {
         lom.subcellCoeff = &v_sub_coef;
         lom.VolumeTerms = new MixedConvectionIntegrator(v_sub_coef);
      }
   }
   else { subcell_mesh = &pmesh; }

   Assembly asmbl(dof_info, lom, inflow_gf, pfes, subcell_mesh, exec_mode);

   // Setup the initial conditions.
   const int vsize = pfes.GetVSize();
   Array<int> offset(levels + 1);
   for (int i = 0; i < offset.Size(); i++) { offset[i] = i*vsize; }
   BlockVector S(offset, Device::GetMemoryType());
   // Primary scalar field is u.
   ParGridFunction u(&pfes);
   u.MakeRef(&pfes, S, offset[0]);
   FunctionCoefficient u0(u0_function);
   u.ProjectCoefficient(u0);
   u.SyncAliasMemory(S);
   // For the case of product remap, we also solve for s and u_s.
   ParGridFunction s, us, q, usq;
   Array<bool> u_bool_el, u_bool_dofs;
   if (levels > 1)
   {
      ComputeBoolIndicators(pmesh.GetNE(), u, u_bool_el, u_bool_dofs);

      s.SetSpace(&pfes);
      BoolFunctionCoefficient sc(s0_function, u_bool_el);
      s.ProjectCoefficient(sc);
      us.MakeRef(&pfes, S, offset[1]);
      const double *h_u = u.HostRead();
      double *h_us = us.HostWrite();
      const double *h_s = s.HostRead();
      for (int i = 0; i < s.Size(); i++)
      {
         // Simple - we don't target conservation at initialization.
         h_us[i]  = h_u[i] * h_s[i];
      }
      us.SyncAliasMemory(S);

      if (levels > 2)
      {
         q.SetSpace(&pfes);
         BoolFunctionCoefficient qc(q0_function, u_bool_el);
         q.ProjectCoefficient(qc);
         usq.MakeRef(&pfes, S, offset[2]);
         double *h_usq = usq.HostWrite();
         const double *h_q = q.HostRead();
         // Simple - we don't target conservation at initialization.
         for (int i = 0; i < s.Size(); i++)
         {
            h_usq[i] = h_us[i] * h_q[i];
         }
         usq.SyncAliasMemory(S);
      }
   }

   // Smoothness indicator.
   SmoothnessIndicator *smth_indicator = NULL;
   if (smth_ind_type)
   {
      smth_indicator = new SmoothnessIndicator(smth_ind_type, *subcell_mesh,
                                               pfes, u, dof_info);
   }

   // Setup of the high-order solver (if any).
   HOSolver *ho_solver_b = NULL, *ho_solver_s = NULL;
   if (ho_type == HOSolverType::Neumann)
   {
      ho_solver_b = new NeumannHOSolver(pfes, M_HO, K_sharp, lumpedM, asmbl);
   }
   else if (ho_type == HOSolverType::CG)
   {
      ho_solver_b = new CGHOSolver(pfes, M_HO, K_bound);
   }
   else if (ho_type == HOSolverType::LocalInverse)
   {
      ho_solver_b = new LocalInverseHOSolver(pfes, M_HO, K_bound);
      ho_solver_s= new LocalInverseHOSolver(pfes, M_HO, K_sharp);
   }

   // Setup the low order solver (if any).
   LOSolver *lo_solver_b = NULL, *lo_solver_s = NULL;
   Array<int> lo_smap_b, lo_smap_s;
   const bool time_dep = (exec_mode == 0) ? false : true;
   if (lo_type == LOSolverType::DiscrUpwind)
   {
         lo_smap_b = SparseMatrix_Build_smap(K_bound.SpMat());
         lo_solver_b = new DiscreteUpwind(pfes, K_bound.SpMat(), lo_smap_b,
                                        lumpedM, asmbl, time_dep, false);

         lo_smap_s = SparseMatrix_Build_smap(K_sharp.SpMat());
         lo_solver_s = new DiscreteUpwind(pfes, K_sharp.SpMat(), lo_smap_s,
                                        lumpedM, asmbl, time_dep, false);
   }
   else if (lo_type == LOSolverType::DiscrUpwindPrec)
   {
      lo_smap_b = SparseMatrix_Build_smap(lom.pk->SpMat());
      lo_solver_b = new DiscreteUpwind(pfes, lom.pk->SpMat(), lo_smap_b,
                                     lumpedM, asmbl, time_dep, true);
   }
   else if (lo_type == LOSolverType::ResDist)
   {
      const bool subcell_scheme = false;
      if (pa)
      {
         lo_solver_b = new PAResidualDistribution(pfes, K_sharp, asmbl, lumpedM,
                                                subcell_scheme, time_dep);
         if (exec_mode == 0)
         {
            const PAResidualDistribution *RD_ptr =
               dynamic_cast<const PAResidualDistribution*>(lo_solver_b);
            RD_ptr->SampleVelocity(FaceType::Interior);
            RD_ptr->SampleVelocity(FaceType::Boundary);
            RD_ptr->SetupPA(FaceType::Interior);
            RD_ptr->SetupPA(FaceType::Boundary);
         }
      }
      else
      {
         lo_solver_b = new ResidualDistribution(pfes, K_sharp, asmbl, lumpedM,
                                              subcell_scheme, time_dep);
      }
   }
   else if (lo_type == LOSolverType::ResDistSubcell)
   {
      const bool subcell_scheme = true;
      if (pa)
      {
         lo_solver_b = new PAResidualDistributionSubcell(pfes, K_sharp, asmbl, lumpedM,
                                                       subcell_scheme, time_dep);
         if (exec_mode == 0)
         {
            const PAResidualDistributionSubcell *RD_ptr =
               dynamic_cast<const PAResidualDistributionSubcell*>(lo_solver_b);
            RD_ptr->SampleVelocity(FaceType::Interior);
            RD_ptr->SampleVelocity(FaceType::Boundary);
            RD_ptr->SetupPA(FaceType::Interior);
            RD_ptr->SetupPA(FaceType::Boundary);
         }
      }
      else
      {
         lo_solver_b = new ResidualDistribution(pfes, K_sharp, asmbl, lumpedM,
                                              subcell_scheme, time_dep);
      }
   }
   else if (lo_type == LOSolverType::MassBased)
   {
      MFEM_VERIFY(ho_solver_b != nullptr,
                  "Mass-Based LO solver requires a choice of a HO solver.");
      lo_solver_b = new MassBasedAvg(pfes, *ho_solver_b,
                                   (exec_mode == 1) ? &v_gf : nullptr);
   }

   // Setup of the monolithic solver (if any).
   MonolithicSolver *mono_solver = NULL;
   bool mass_lim = (problem_num != 6 && problem_num != 7) ? true : false;
   if (mono_type == MonolithicSolverType::ResDistMono)
   {
      const bool subcell_scheme = false;
      mono_solver = new MonoRDSolver(pfes, K_sharp.SpMat(), M_HO.SpMat(), lumpedM,
                                     asmbl, smth_indicator, velocity,
                                     subcell_scheme, time_dep, mass_lim);
   }
   else if (mono_type == MonolithicSolverType::ResDistMonoSubcell)
   {
      const bool subcell_scheme = true;
      mono_solver = new MonoRDSolver(pfes, K_sharp.SpMat(), M_HO.SpMat(), lumpedM,
                                     asmbl, smth_indicator, velocity,
                                     subcell_scheme, time_dep, mass_lim);
   }

   // Print the starting meshes and initial condition.
   ofstream meshHO("meshHO_init.mesh");
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
   u.SaveAsOne(sltn);

   // Create data collection for solution output: either VisItDataCollection for
   // ASCII data files, or SidreDataCollection for binary data files.
   DataCollection *dc = NULL;
   if (visit)
   {
      dc = new VisItDataCollection("Remhos", &pmesh);
      dc->SetPrecision(precision);
      dc->RegisterField("solution", &u);
      dc->SetCycle(0);
      dc->SetTime(0.0);
      dc->Save();
   }

   socketstream vis_u, vis_b, vis_v_new, vis_a, vis_s, vis_us, vis_q, vis_usq;
   char vishost[] = "localhost";
   int  visport   = 19916;
   if (visualization)
   {
      // Make sure all MPI ranks have sent their 'v' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh.GetComm());

      vis_u.precision(8);
      vis_s.precision(8);
      vis_us.precision(8);
      vis_q.precision(8);
      vis_usq.precision(8);

      int Wx = 0, Wy = 0; // window position
      const int Ww = 300, Wh = 300; // window size
      u.HostRead();
      s.HostRead();
      q.HostRead();
      VisualizeField(vis_u, vishost, visport, u, "Solution u", Wx, Wy, Ww, Wh);
      if (levels > 1)
      {
         VisualizeField(vis_us, vishost, visport, us, "Solution us",
                        Wx, 400, Ww, Wh);
         VisualizeField(vis_s, vishost, visport, s, "Solution s",
                        Wx + Ww, 400, Ww, Wh);
         if (levels > 2)
         {
            VisualizeField(vis_usq, vishost, visport, usq, "Solution usq",
                           Wx, 750, Ww, Wh);
            VisualizeField(vis_q, vishost, visport, q, "Solution q",
                           Wx + Ww, 750, Ww, Wh);
         }
      }
   }

   // Record the initial mass.
   Vector masses(lumpedM);
   const double mass0_u_loc = lumpedM * u;
   double mass0_u, mass0_us, mass0_usq;
   MPI_Allreduce(&mass0_u_loc, &mass0_u, 1, MPI_DOUBLE, MPI_SUM, comm);
   if (levels > 1)
   {
      const double mass0_us_loc = lumpedM * us;
      MPI_Allreduce(&mass0_us_loc, &mass0_us, 1, MPI_DOUBLE, MPI_SUM, comm);
      if (levels > 2)
      {
         const double mass0_usq_loc = lumpedM * usq;
         MPI_Allreduce(&mass0_usq_loc, &mass0_usq, 1,
                       MPI_DOUBLE, MPI_SUM, comm);
      }
   }

   // Setup of the FCT solver (if any).
   Array<int> K_HO_smap;
   FCTSolver *fct_solver_b = NULL, *fct_solver_s = NULL;
   if (fct_type == FCTSolverType::FluxBased)
   {
      MFEM_VERIFY(pa == false, "Flux-based FCT and PA are incompatible.");
      K_bound.SpMat().HostReadI();
      K_bound.SpMat().HostReadJ();
      K_bound.SpMat().HostReadData();
      K_HO_smap = SparseMatrix_Build_smap(K_bound.SpMat());
      const int fct_iterations = 1;
      fct_solver_b = new FluxBasedFCT(pfes, smth_indicator, dt, K_bound.SpMat(),
                                    K_HO_smap, M_HO.SpMat(), fct_iterations);
   }
   else if (fct_type == FCTSolverType::ClipScale)
   {
      fct_solver_b = new ClipScaleSolver(pfes, smth_indicator, dt);
   }
   else if (fct_type == FCTSolverType::NonlinearPenalty)
   {
      fct_solver_b = new NonlinearPenaltySolver(pfes, smth_indicator, dt);
   }
   else if (fct_type == FCTSolverType::FCTProject)
   {
      fct_solver_b = new ElementFCTProjection(pfes, dt);
      fct_solver_s = new ElementFCTProjection(pfes, dt);
   }

   AdvectionOperator adv(S.Size(), ml, lumpedM, K_sharp, M_HO, K_bound,
                         x, xsub, v_gf, v_sub_gf, dof_info,
                         ho_solver_b, ho_solver_s,
                         lo_solver_b, lo_solver_s,
                         fct_solver_b, fct_solver_s, mono_solver);
   adv.evolve_sharp = sharp;
   if (levels > 1) { adv.SetInitialS(s); }
   if (levels > 2) { adv.SetInitialQ(q); }

   double t = 0.0;
   adv.SetTime(t);
   adv.SetTimeStepControl(dt_control);
   ode_solver->Init(adv);

   double umin, umax;
   GetMinMax(u, umin, umax);

   if (exec_mode == 1)
   {
      adv.SetRemapStartPos(x0, x0_sub);

      // For remap, the pseudo-time always evolves from 0 to 1.
      t_final = 1.0;
   }

   ParGridFunction res = u;
   double residual;
   double s_min_glob = numeric_limits<double>::infinity(),
          s_max_glob = -numeric_limits<double>::infinity();

   // Time-integration (loop over the time iterations, ti, with a time-step dt).
   bool done = false;

   // Data at the beginninng of the time step.
   BlockVector S_old(S);
   ParGridFunction u_old(&pfes);
   u_old.MakeRef(&pfes, S_old, offset[0]);

   int ti_total = 0, ti = 0;
   while (done == false)
   {
      double dt_real = min(dt, t_final - t);

      // This also resets the time step estimate when automatic dt is on.
      adv.SetDt(dt_real);
      if (lo_solver_b)  { lo_solver_b->UpdateTimeStep(dt_real); }
      if (fct_solver_b) { fct_solver_b->UpdateTimeStep(dt_real); }
      if (lo_solver_s)  { lo_solver_s->UpdateTimeStep(dt_real); }
      if (fct_solver_s) { fct_solver_s->UpdateTimeStep(dt_real); }

      if (levels > 1)
      {
         ComputeMinMaxS(pmesh.GetNE(), us, u, s_min_glob, s_max_glob);
#ifdef REMHOS_FCT_PRODUCT_DEBUG
         if (myid == 0)
         {
            std::cout << "   --- Full time step" << std::endl;
            std::cout << "   in:  ";
            std::cout << std::scientific << std::setprecision(5);
            std::cout << "min_s: " << s_min_glob
                      << "; max_s: " << s_max_glob << std::endl;
         }
#endif
      }

      // Needed for velocity modifications.
      dof_info.ComputeLinMaxBound(u, u_max_bounds, u_max_bounds_grad_dir);
      // needed for velocity visualization.
      if (sharp == true)
      {
         if (exec_mode == 0)
         {
            v_new_vis.ProjectCoefficient(v_new_coeff_adv);
         }
         else
         {
            v_new_vis.ProjectCoefficient(v_new_coeff_rem);
         }
      }

      S_old = S;
      ode_solver->Step(S, t, dt_real);
      ti++;
      ti_total++;

      if (dt_control != TimeStepControl::FixedTimeStep)
      {
         double dt_est = adv.GetTimeStepEstimate();
         if (dt_est < dt_real)
         {
            // Repeat with the proper time step.
            if (myid == 0)
            {
               cout << "Repeat / decrease dt: "
                    << dt_real << " --> " << 0.85 * dt << endl;
            }
            ti--;
            t -= dt_real;
            S  = S_old;
            dt = 0.85 * dt;
            if (dt < 1e-12) { MFEM_ABORT("The time step crashed!"); }
            continue;
         }
         else if (dt_est > 1.25 * dt_real)
         {
            if (myid == 0)
            {
               cout << "Increase dt: " << dt << " --> " << 1.02 * dt << endl;
            }
            dt *= 1.02;
         }
      }

      const int size = u.Size();
      Vector new_m(size);
      Array<bool> active_elem, active_dofs;

      if (levels > 1)
      {
//         // Compute s-bounds.
//         ComputeBoolIndicators(NE, u_old, active_elem, active_dofs);
//         ComputeRatio(NE, us_old, u_old, s_old, active_elem, active_dofs);
//         dof_info.ComputeElementsMinMax(s_old, dof_info.xe_min, dof_info.xe_max,
//                                        &active_elem, &active_dofs);
//         dof_info.ComputeBounds(dof_info.xe_min, dof_info.xe_max,
//                                dof_info.xi_min, dof_info.xi_max,
//                                1, &active_elem);

//         Vector us_min(size), us_max(size);
//         ComputeBoolIndicators(NE, u, active_elem, active_dofs);
//         fct_solver_b->ScaleProductBounds(dof_info.xi_min, dof_info.xi_max, u,
//                                          active_elem, active_dofs,
//                                          us_min, us_max);

         // Must always be ok with RK1.
         // check_violation(us, us_min, us_max, "us-full", &active_dofs);
      }

#ifdef REMHOS_FCT_PRODUCT_DEBUG
         ComputeMinMaxS(NE, us, u, s_min_glob, s_max_glob);
         if (myid == 0)
         {
            std::cout << "   out: ";
            std::cout << std::scientific << std::setprecision(5);
            std::cout << "min_s: " << s_min_glob
                      << "; max_s: " << s_max_glob << std::endl;
         }
#endif

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
         x0.HostReadWrite(); v_sub_gf.HostReadWrite();
         x.HostReadWrite();
         add(x0, t, v_gf, x);
         x0_sub.HostReadWrite(); v_sub_gf.HostReadWrite();
         MFEM_VERIFY(xsub != NULL,
                     "xsub == NULL/This code should not be entered for order = 1.");
         xsub->HostReadWrite();
         add(x0_sub, t, v_sub_gf, *xsub);
      }

      if (problem_num != 6 && problem_num != 7 && problem_num != 8)
      {
         done = (t >= t_final - 1.e-8*dt);
      }
      else
      {
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
         else { res = u; }
      }

      if (done || ti % vis_steps == 0)
      {
         if (myid == 0)
         {
            cout << "time step: " << ti << ", time: " << t
                 << ", dt: " << dt << ", residual: " << residual << endl;
         }

         if (visualization)
         {
            ComputeBoolIndicators(NE, u, active_elem, active_dofs);
            L2_FECollection fec_0(0, dim);
            ParFiniteElementSpace fes_0(&pmesh, &fec_0);
            ParGridFunction bool_el_gf(&fes_0); bool_el_gf = 0.0;
            for (int i = 0; i < NE; i++)
            {
               if (active_elem[i]) { bool_el_gf(i) = 1.0; }
            }

            int Wx = 0, Wy = 0; // window position
            int Ww = 300, Wh = 300; // window size
            VisualizeField(vis_u, vishost, visport, u, "Solution u",
                           Wx, Wy, Ww, Wh);
            VisualizeField(vis_b, vishost, visport, u_max_bounds, "Bounds",
                           Wx+Ww, Wy, Ww, Wh);
            if (sharp == true)
            {
               VisualizeField(vis_v_new, vishost, visport, v_new_vis, "Vel",
                              Wx+2*Ww, Wy, Ww, Wh);
            }
            VisualizeField(vis_a, vishost, visport, bool_el_gf, "Active El",
                           Wx+3*Ww, Wy, Ww, Wh);

            if (levels > 1)
            {
               ComputeBoolIndicators(NE, u, u_bool_el, u_bool_dofs);

               // Recompute s = u_s / u.
               VisualizeField(vis_us, vishost, visport, us, "Solution us",
                              Wx, 400, Ww, Wh);
               ComputeRatio(pmesh.GetNE(), us, u, s, u_bool_el, u_bool_dofs);
               VisualizeField(vis_s, vishost, visport, s, "Solution s",
                              Wx + Ww, 400, Ww, Wh);

               // Recompute q = usq / us.
               if (levels > 2)
               {
                  VisualizeField(vis_usq, vishost, visport, usq,
                                 "Solution usq", Wx, 750, Ww, Wh);
                  ComputeRatio(pmesh.GetNE(), usq, us, q,
                               u_bool_el, u_bool_dofs);
                  VisualizeField(vis_q, vishost, visport, q, "Solution q",
                                 Wx + Ww, 750, Ww, Wh);
               }
            }
         }

         if (visit)
         {
            dc->SetCycle(ti);
            dc->SetTime(t);
            dc->Save();
         }
      }
   }

   if (dt_control != TimeStepControl::FixedTimeStep && myid == 0)
   {
      cout << "Total time steps: " << ti_total
           << " (" << ti_total-ti << " repeated)." << endl;
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
   double mass_u_loc = 0.0, mass_us_loc = 0.0, mass_usq_loc = 0.0;
   if (exec_mode == 1)
   {
      ml.BilinearForm::operator=(0.0);
      ml.Assemble();
      lumpedM.HostRead();
      ml.SpMat().GetDiag(lumpedM);
      mass_u_loc = lumpedM * u;
      if (levels > 1)
      {
         mass_us_loc = lumpedM * us;
         if (levels > 2) { mass_usq_loc = lumpedM * usq; }
      }
   }
   else
   {
      mass_u_loc = masses * u;
      if (levels > 1)
      {
         mass_us_loc = masses * us;
         if (levels > 2) { mass_usq_loc = masses * usq; }
      }
   }
   double mass_u, mass_us, mass_usq, s_max;
   MPI_Allreduce(&mass_u_loc, &mass_u, 1, MPI_DOUBLE, MPI_SUM, comm);
   const double umax_loc = u.Max();
   MPI_Allreduce(&umax_loc, &umax, 1, MPI_DOUBLE, MPI_MAX, comm);
   if (levels > 1)
   {
      MPI_Allreduce(&mass_us_loc, &mass_us, 1, MPI_DOUBLE, MPI_SUM, comm);
      if (levels > 2)
      { MPI_Allreduce(&mass_usq_loc, &mass_usq, 1, MPI_DOUBLE, MPI_SUM, comm); }
   }
   if (myid == 0)
   {
      cout << setprecision(10)
           << "Final mass u:   " << mass_u << endl
           << "Max value u:    " << umax << endl << setprecision(6)
           << "Mass loss u:    " << abs(mass0_u - mass_u) << endl;
      if (levels > 1)
      {
         cout << setprecision(10)
              << "Final mass us:  " << mass_us << endl
              << "Mass loss us:   " << abs(mass0_us - mass_us) << endl;

         if (levels > 2)
         {
            cout << setprecision(10)
                 << "Final mass usq: " << mass_usq << endl
                 << "Mass loss usq:  " << abs(mass0_usq - mass_usq) << endl;
         }
      }
   }

   ConstantCoefficient zero(0.0);
   double norm_u = u.ComputeL2Error(zero), norm_us = 0.0, norm_usq = 0.0;
   if (levels > 1)
   {
      norm_us  = us.ComputeL2Error(zero);
      if (levels > 2) { norm_usq = usq.ComputeL2Error(zero); }
   }
   if (myid == 0)
   {
      cout << setprecision(12) << "L2-norm u:   " << norm_u << endl;
      if (levels > 1)
      {
         cout << "L2-norm us:  " << norm_us  << endl;
         if (levels > 2)
         {
            cout << "L2-norm usq: " << norm_usq << endl;
         }
      }
   }

   // Compute errors, if the initial condition is equal to the final solution
   if (problem_num == 4 || problem_num == 18) // solid body rotation
   {
      double err_L1 = u.ComputeLpError(1., u0),
             err_L2 = u.ComputeL2Error(u0);
      if (myid == 0)
      {
         cout << "L1-error: " << err_L1 << endl
              << "L2-error: " << err_L2 << endl;
      }
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
   delete fct_solver_b;
   delete fct_solver_s;
   delete smth_indicator;
   delete ho_solver_b;
   delete ho_solver_s;

   delete ode_solver;
   delete mesh_fec;
   delete lom.pk;
   delete dc;

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

AdvectionOperator::AdvectionOperator(int size,
                                     BilinearForm &_ml, Vector &_lumpedM,
                                     ParBilinearForm &K_s, ParBilinearForm &K_b,
                                     ParBilinearForm &M_HO_,
                                     GridFunction &pos, GridFunction *sub_pos,
                                     GridFunction &vel, GridFunction &sub_vel, DofInfo &_dofs,
                                     HOSolver *hos_b, HOSolver *hos_s,
                                     LOSolver *los_b, LOSolver *los_s,
                                     FCTSolver *fct_b, FCTSolver *fct_s,
                                     MonolithicSolver *mos) :
   TimeDependentOperator(size), ml(_ml),
   K_sharp(K_s), K_bound(K_b), M_HO(M_HO_),
   lumpedM(_lumpedM),
   start_mesh_pos(pos.Size()), start_submesh_pos(sub_vel.Size()),
   mesh_pos(pos), submesh_pos(sub_pos),
   mesh_vel(vel), submesh_vel(sub_vel),
   x_gf(K_sharp.ParFESpace()), dofs(_dofs),
   s_old(0), q_old(0),
   ho_solver_b(hos_b), ho_solver_s(hos_s),
   lo_solver_b(los_b), lo_solver_s(los_s),
   fct_solver_b(fct_b), fct_solver_s(fct_s), mono_solver(mos) { }

void check_violation(const Vector &u_new,
                     const Vector &u_min, const Vector &u_max, string info,
                     double tol, const Array<bool> *active_dofs)
{
   const int size = u_new.Size();
   for (int i = 0; i < size; i++)
   {
      if (active_dofs && (*active_dofs)[i] == false) { continue; }

      if (u_new(i) + tol < u_min(i) || u_new(i) > u_max(i) + tol)
      {
         cout << info << " bounds violation: " << i << " "
              << u_min(i) << " " << u_new(i) << " " << u_max(i) << endl;
         cout << u_max(i) - u_new(i) << " " << u_new(i) - u_min(i) << endl;
         MFEM_ABORT("bounds");
      }
   }
}

void clean_roundoff(const Vector &u_min, const Vector &u_max,
                    Vector &u, const Array<bool> *active_dofs)
{
   const int size = u.Size();
   for (int i = 0; i < size; i++)
   {
      if (active_dofs && (*active_dofs)[i] == false) { u(i) = 0.0; continue; }

      u(i) = fmin(u(i), u_max(i));
      u(i) = fmax(u(i), u_min(i));
   }

}

void blend_global_u(int levels, const Vector &u_b, const Vector &u_s,
                    const Vector &m, Vector &u_min, Vector &u_max,
                    const Array<bool> &active_dofs,
                    const Vector &s_min, const Vector &s_max,
                    const Vector &us_b, double s_glob,
                    const Vector &q_min, const Vector &q_max,
                    const Vector &usq_b, double q_glob,
                    Vector &u)
{
   const double eps = 1e-12;
   const int size = u_b.Size();

   Vector f_clip(size);
   f_clip = 0.0;
   u.SetSize(size);

   // Clip.
   double Sp = 0.0, Sn = 0.0;
   for (int i = 0; i < size; i++)
   {
      if (active_dofs[i] == false) { continue; }

      // eq (3c), right inequality.
      if (levels > 1 && s_max(i) - s_glob < -eps)
      {
         u_max(i) = fmin(u_max(i),
                         (us_b(i) - u_b(i) * s_glob) / (s_max(i) - s_glob));
         u_max(i) = fmax(u_max(i), u_b(i));
      }
      if (levels > 1 && s_max(i) - s_glob > eps)
      {
         u_min(i) = fmax(u_min(i),
                         (us_b(i) - u_b(i) * s_glob) / (s_max(i) - s_glob));
         u_min(i) = fmin(u_min(i), u_b(i));
      }

      // eq (3c), left inequality.
      if (levels > 1 && s_min(i) - s_glob < - eps)
      {
         u_min(i) = fmax(u_min(i),
                         (us_b(i) - u_b(i) * s_glob) / (s_min(i) - s_glob));
         u_min(i) = fmin(u_min(i), u_b(i));
      }
      if (levels > 1 && s_min(i) - s_glob > eps)
      {
         u_max(i) = fmin(u_max(i),
                         (us_b(i) - u_b(i) * s_glob) / (s_min(i) - s_glob));
         u_max(i) = fmax(u_max(i), u_b(i));
      }

      // eq (3d), right inequality.
      if (levels > 2 && s_glob * q_max(i) - s_glob * q_glob < - eps)
      {
         u_max(i) = fmin(u_max(i),
                         u_b(i) + (usq_b(i) - us_b(i) * q_max(i)) /
                                  (s_glob * q_max(i) - s_glob * q_glob));
         u_max(i) = fmax(u_max(i), u_b(i));
      }
      if (levels > 2 && s_glob * q_max(i) - s_glob * q_glob > eps)
      {
         u_min(i) = fmax(u_min(i),
                         u_b(i) + (usq_b(i) - us_b(i) * q_max(i)) /
                                  (s_glob * q_max(i) - s_glob * q_glob));
         u_min(i) = fmin(u_min(i), u_b(i));
      }

      // eq (3d), left inequality.
      if (levels > 2 && s_glob * q_min(i) - s_glob * q_glob < - eps)
      {
         u_min(i) = fmax(u_min(i),
                         u_b(i) + (usq_b(i) - us_b(i) * q_min(i)) /
                                  (s_glob * q_min(i) - s_glob * q_glob));
         u_min(i) = fmin(u_min(i), u_b(i));
      }
      if (levels > 2 && s_glob * q_min(i) - s_glob * q_glob > eps)
      {
         u_max(i) = fmin(u_max(i),
                         u_b(i) + (usq_b(i) - us_b(i) * q_min(i)) /
                                  (s_glob * q_min(i) - s_glob * q_glob));
         u_max(i) = fmax(u_max(i), u_b(i));
      }

      double f_clip_min = u_min(i) - u_b(i),
             f_clip_max = u_max(i) - u_b(i);

      f_clip(i) = u_s(i) - u_b(i);
      f_clip(i) = fmin(f_clip_max, fmax(f_clip_min, f_clip(i)));
      Sp += fmax(m(i) * f_clip(i), 0.0);
      Sn += fmin(m(i) * f_clip(i), 0.0);
   }

   MPI_Allreduce(MPI_IN_PLACE, &Sp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &Sn, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   const double S = Sp + Sn;

   for (int i = 0; i < size; i++)
   {
      if (active_dofs[i] == false) { u(i) = 0.0; continue; }

      if (S > eps && f_clip(i) > 0.0)
      {
         f_clip(i) *= - Sn / Sp;
      }
      if (S < -eps && f_clip(i) < 0.0)
      {
         f_clip(i) *= - Sp / Sn;
      }

      u(i) = u_b(i) + f_clip(i);
   }
}

void blend_global_us(int levels, const Vector &u, const Vector &u_b,
                     const Vector &us_b, const Vector &us_s, const Vector &m,
                     Vector &us_min, Vector &us_max,
                     const Array<bool> &active_dofs,
                     const Vector &q_min, const Vector &q_max,
                     const Vector &usq_b,
                     double s_glob, double q_glob, Vector &us)
{
   const double eps = 1e-12;
   const int size = us_b.Size();

   Vector f_clip(size);
   f_clip = 0.0;
   us.SetSize(size);

   Vector us_LO(size);
   for (int i = 0; i < size; i++)
   {
      us_LO(i) = us_b(i) + (u(i) - u_b(i)) * s_glob;
   }

   // Clip.
   double Sp = 0.0, Sn = 0.0;
   for (int i = 0; i < size; i++)
   {
      if (active_dofs[i] == false) { continue; }

      // eq (4c), right inequality.
      if (levels > 2 && q_max(i) - q_glob < -eps)
      {
         us_max(i) = fmin(us_max(i),
                          (usq_b(i) - us_b(i) * q_glob) / (q_max(i) - q_glob));
         us_max(i) = fmax(us_max(i), us_LO(i));
      }
      if (levels > 2 && q_max(i) - q_glob > eps)
      {
         us_min(i) = fmax(us_min(i),
                          (usq_b(i) - us_b(i) * q_glob) / (q_max(i) - q_glob));
         us_min(i) = fmin(us_min(i), us_LO(i));
      }

      // eq (4c), left inequality.
      if (levels > 2 && q_min(i) - q_glob < -eps)
      {
         us_min(i) = fmax(us_min(i),
                          (usq_b(i) - us_b(i) * q_glob) / (q_min(i) - q_glob));
         us_min(i) = fmin(us_min(i), us_LO(i));
      }
      if (levels > 2 && q_min(i) - q_glob > eps)
      {
         us_max(i) = fmin(us_max(i),
                          (usq_b(i) - us_b(i) * q_glob) / (q_min(i) - q_glob));
         us_max(i) = fmax(us_max(i), us_LO(i));
      }

      double f_clip_min = us_min(i) - us_LO(i);
      double f_clip_max = us_max(i) - us_LO(i);

      f_clip(i) = us_s(i) - us_LO(i);
      f_clip(i) = fmin(f_clip_max, fmax(f_clip_min, f_clip(i)));
      Sp += fmax(m(i) * f_clip(i), 0.0);
      Sn += fmin(m(i) * f_clip(i), 0.0);
   }

   MPI_Allreduce(MPI_IN_PLACE, &Sp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &Sn, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   const double S = Sp + Sn;

   for (int i = 0; i < size; i++)
   {
      if (active_dofs[i] == false) { us(i) = 0.0; continue; }

      if (S > eps && f_clip(i) > 0.0)
      {
         f_clip(i) *= - Sn / Sp;
      }
      if (S < -eps && f_clip(i) < 0.0)
      {
         f_clip(i) *= - Sp / Sn;
      }

      us(i) = us_LO(i) + f_clip(i);
   }
}

void blend_global_usq(const Vector &us, const Vector &us_b,
                      const Vector &usq_b, const Vector &usq_s, const Vector &m,
                      const Vector &usq_min, const Vector &usq_max,
                      const Array<bool> &active_dofs, double q_glob,
                      Vector &usq)
{
   const double eps = 1e-12;
   const int size = usq_b.Size();

   Vector f_clip(size);
   f_clip = 0.0;
   usq.SetSize(size);

   Vector usq_LO(size);
   for (int i = 0; i < size; i++)
   {
      usq_LO(i) = usq_b(i) + (us(i) - us_b(i)) * q_glob;
   }
   check_violation(usq_LO, usq_min, usq_max, "usq-blend-LO",
                   1e-12, &active_dofs);

   // Clip.
   double Sp = 0.0, Sn = 0.0;
   for (int i = 0; i < size; i++)
   {
      if (active_dofs[i] == false) { continue; }

      double f_clip_min = usq_min(i) - usq_LO(i);
      double f_clip_max = usq_max(i) - usq_LO(i);

      f_clip(i) = usq_s(i) - usq_LO(i);
      f_clip(i) = fmin(f_clip_max, fmax(f_clip_min, f_clip(i)));
      Sp += fmax(m(i) * f_clip(i), 0.0);
      Sn += fmin(m(i) * f_clip(i), 0.0);
   }

   MPI_Allreduce(MPI_IN_PLACE, &Sp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &Sn, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   const double S = Sp + Sn;

   for (int i = 0; i < size; i++)
   {
      if (active_dofs[i] == false) { usq(i) = 0.0; continue; }

      if (S > eps && f_clip(i) > 0.0)
      {
         f_clip(i) *= - Sn / Sp;
      }
      if (S < -eps && f_clip(i) < 0.0)
      {
         f_clip(i) *= - Sp / Sn;
      }

      usq(i) = usq_LO(i) + f_clip(i);
   }
}

void sharp_product_sync(const Vector &u, const Vector &m,
                        const Vector &s_min, const Vector &s_max,
                        const Vector &us_b,
                        const Array<bool> &active_dofs_blend,
                        const Array<bool> &active_dofs_bound,
                        Vector &us)
{
   const double eps = 1e-12;
   const int size = u.Size();
   double Sp = 0.0, Sn = 0.0, S_to_min = 0.0, S_to_max = 0.0;
   for (int i = 0; i < size; i++)
   {
      if (active_dofs_blend[i] == false && active_dofs_bound[i] == true)
      {
         us(i) = 0.0;
         Sn -= m(i) * us_b(i);
         continue;
      }
      if (active_dofs_blend[i] == false)  { us(i) = 0.0; continue; }

      us(i) = fmin(u(i) * s_max(i), fmax(us_b(i), u(i) * s_min(i)));
      Sp += m(i) * fmax(us(i) - us_b(i), 0.0);
      Sn += m(i) * fmin(us(i) - us_b(i), 0.0);
      S_to_min += m(i) * (us(i) - u(i) * s_min(i));
      S_to_max += m(i) * (u(i) * s_max(i) - us(i));

      MFEM_VERIFY(m(i) > 0, "m");
   }

   MPI_Allreduce(MPI_IN_PLACE, &Sp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &Sn, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &S_to_min, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &S_to_max, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   const double S = Sp + Sn;
   if (fabs(S) < eps) { return; }

   // Check if restoring the mass is possible.
   if (S > 0.0)
   {
      double S_max_decr = 0.0;
      for (int i = 0; i < size; i++)
      {
         if (active_dofs_blend[i] == false)  { continue; }

         double r = us(i) - u(i) * s_min(i);
         if (r > 0.0) { S_max_decr += m(i) * r; }
      }
      MPI_Allreduce(MPI_IN_PLACE, &S_max_decr, 1, MPI_DOUBLE,
                    MPI_SUM, MPI_COMM_WORLD);

      MFEM_VERIFY(S_max_decr > S, "Gained mass - impossible to fix!!!");
   }
   if (S < 0.0)
   {
      double S_max_incr = 0.0;
      for (int i = 0; i < size; i++)
      {
         if (active_dofs_blend[i] == false)  { continue; }

         double r = u(i) * s_max(i) - us(i);
         if (r > 0.0) { S_max_incr += m(i) * r; }
      }
      MPI_Allreduce(MPI_IN_PLACE, &S_max_incr, 1, MPI_DOUBLE,
                    MPI_SUM, MPI_COMM_WORLD);

      MFEM_VERIFY(S_max_incr > fabs(S), "Lost mass - impossible to fix!!!");
   }

   // Restore the mass.
   for (int i = 0; i < size; i++)
   {
      if (active_dofs_blend[i] == false)  { us(i) = 0.0; continue; }

      if (S > 0.0)
      {
         double r = us(i) - u(i) * s_min(i);
         MFEM_VERIFY(r >= 0, "r");
         us(i) = us(i) - S / S_to_min * r;

         if (us(i) < u(i) * s_min(i))
         {
            cout << "below min: " << us(i) << " " << u(i) * s_min(i) << endl;
            cout << r << " " << us(i) - fabs(Sn / Sp) * r << " " << fabs(Sn / Sp) * r << endl;
            MFEM_ABORT("min bounds");
         }
      }
      if (S < 0.0)
      {
         double r = u(i) * s_max(i) - us(i);
         MFEM_VERIFY(r >= 0, "r");
         us(i) = us(i) + fabs(S / S_to_max) * r;

         if (us(i) > u(i) * s_max(i))
         {
            cout << "above max: " << us(i) << " " << u(i) * s_max(i) << endl;
            cout << r << " " << us(i) + fabs(Sn / Sp) * r << " " << fabs(Sn / Sp) * r << endl;
            MFEM_ABORT("max bounds");
         }
      }
   }

   double m_bound = 0.0, m_blend = 0.0;
   for (int i = 0; i < size; i++)
   {
      if (active_dofs_blend[i] == true) { m_blend += m(i) * us(i); }
      if (active_dofs_bound[i] == true) { m_bound += m(i) * us_b(i); }
   }
   MPI_Allreduce(MPI_IN_PLACE, &m_bound, 1, MPI_DOUBLE,
                 MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &m_blend, 1, MPI_DOUBLE,
                 MPI_SUM, MPI_COMM_WORLD);
   if (fabs(m_bound - m_blend) > eps)
   {
      cout << fabs(m_bound - m_blend) << endl;
      MFEM_ABORT("error in us-sync conservation");
   }
}

void AdvectionOperator::Mult(const Vector &X, Vector &Y) const
{
   if (exec_mode == 1)
   {
      // Move the mesh positions.
      const double t = GetTime();
      add(start_mesh_pos, t, mesh_vel, mesh_pos);
      if (submesh_pos)
      {
         add(start_submesh_pos, t, submesh_vel, *submesh_pos);
      }

      // Reassemble on the new mesh. Element contributions.
      // Currently needed to have the sparse matrices used by the LO methods.
      ml.BilinearForm::operator=(0.0);
      ml.Assemble();
      lumpedM.HostReadWrite();
      ml.SpMat().GetDiag(lumpedM);

      M_HO.FESpace()->GetMesh()->DeleteGeometricFactors();
      M_HO.BilinearForm::operator=(0.0);
      M_HO.Assemble();
      K_bound.BilinearForm::operator=(0.0);
      K_bound.Assemble(0);
      K_sharp.BilinearForm::operator=(0.0);
      K_sharp.Assemble(0);
   }

   const int size = K_sharp.ParFESpace()->GetVSize();
   const int NE   = K_sharp.ParFESpace()->GetNE();

   // Needed because X and Y are allocated on the host by the ODESolver.
   X.Read(); Y.Read();

   Vector *xptr = const_cast<Vector*>(&X);

   MFEM_VERIFY(ho_solver_b && lo_solver_b, "FCT requires HO & LO solvers.");

   int levels = 1;
   if (X.Size() > size)   { levels++; }
   if (X.Size() > 2*size) { levels++; }

   // u.
   Vector u_old, d_u;
   u_old.MakeRef(*xptr, 0, size);
   d_u.MakeRef(Y, 0, size);

   // us.
   Vector us_old, d_us;
   if (levels > 1)
   {
      us_old.MakeRef(*xptr, size, size);
      d_us.MakeRef(Y, size, size);
   }

   // usq.
   Vector usq_old, d_usq;
   if (levels > 2)
   {
      usq_old.MakeRef(*xptr, 2*size, size);
      d_usq.MakeRef(Y, 2*size, size);
   }

   Array<bool> active_elem_old, active_dofs_old,
               active_elem_bound, active_dofs_bound,
               active_elem_blend, active_dofs_blend;

   //
   // Stuff related to s = us_old / u_old.
   //
   // Compute the ratio s = us_old / u_old, on the old active dofs.
   Vector s_min(size), s_max(size);
   ComputeBoolIndicators(NE, u_old, active_elem_old, active_dofs_old);
   if (levels > 1)
   {
      // Bounds for s, based on the old values (and old active dofs).
      // This doesn't consider s values from the old inactive dofs, because
      // there were no bounds restriction on them at the previous time step.
      dofs.ComputeElementsMinMax(s_old, dofs.xe_min, dofs.xe_max,
                                 &active_elem_old, &active_dofs_old);
      dofs.ComputeBounds(dofs.xe_min, dofs.xe_max,
                         s_min, s_max, 1, &active_elem_old);
   }

   //
   // Stuff related to q = usq_old / us_old.
   //
   // Compute the ratio q = usq_old / us_old, on the old active dofs.
   Vector q_min(size), q_max(size);
   ComputeBoolIndicators(NE, u_old, active_elem_old, active_dofs_old);
   if (levels > 2)
   {
      // Bounds for q, based on the old values (and old active dofs).
      // This doesn't consider q values from the old inactive dofs, because
      // there were no bounds restriction on them at the previous time step.
      dofs.ComputeElementsMinMax(q_old, dofs.xe_min, dofs.xe_max,
                                 &active_elem_old, &active_dofs_old);
      dofs.ComputeBounds(dofs.xe_min, dofs.xe_max,
                         q_min, q_max, 1, &active_elem_old);
   }

   //
   // Bounded solution u_b.
   //
   ParGridFunction u_old_gf(K_bound.ParFESpace());
   u_old_gf = u_old; u_old_gf.ExchangeFaceNbrData();
   Vector d_u_b_HO(size), d_u_b_LO(size), u_min(size), u_max(size), d_u_b(size);
   lo_solver_b->CalcLOSolution(u_old, d_u_b_LO);
   ho_solver_b->CalcHOSolution(u_old, d_u_b_HO);
   dofs.ComputeElementsMinMax(u_old, dofs.xe_min, dofs.xe_max, NULL, NULL);
   dofs.ComputeBounds(dofs.xe_min, dofs.xe_max, u_min, u_max);
   fct_solver_b->CalcFCTSolution(u_old_gf, lumpedM, d_u_b_HO, d_u_b_LO,
                                 u_min, u_max, d_u_b);
   if (dt_control == TimeStepControl::LOBoundsError)
   {
      UpdateTimeStepEstimate(u_old, d_u_b_LO, dofs.xi_min, dofs.xi_max);
   }
   // Evolve u_b, get the new active dofs, check violations.
   Vector u_b(size);
   add(1.0, u_old, dt, d_u_b, u_b);
   ComputeBoolIndicators(NE, u_b, active_elem_bound, active_dofs_bound);
   check_violation(u_b, u_min, u_max, "u-b-mult", 1e-12, nullptr);
   clean_roundoff(u_min, u_max, u_b, &active_dofs_bound);

   //
   // Bounded solution us_b.
   //
   ParGridFunction us_old_gf(K_bound.ParFESpace());
   Vector d_us_HO(size), d_us_LO, d_us_b(size);
   Vector us_b(size), us_min(size), us_max(size);
   if (levels > 1)
   {
      us_old_gf = us_old; us_old_gf.ExchangeFaceNbrData();
      if (fct_solver_b->NeedsLOProductInput())
      {
         d_us_LO.SetSize(size);
         lo_solver_b->CalcLOSolution(us_old, d_us_LO);
      }
      ho_solver_b->CalcHOSolution(us_old, d_us_HO);
      fct_solver_b->CalcFCTProduct(us_old_gf, lumpedM, d_us_HO, d_us_LO,
                                   s_min, s_max, u_b,
                                   active_elem_bound, active_dofs_bound, d_us_b);
      // Evolve us_b, check violations.
      add(1.0, us_old, dt, d_us_b, us_b);
      fct_solver_b->ScaleProductBounds(s_min, s_max, u_b,
                                       active_elem_bound, active_dofs_bound,
                                       us_min, us_max);
      check_violation(us_b, us_min, us_max, "us-b-mult",
                      1e-12, &active_dofs_bound);
      clean_roundoff(us_min, us_max, us_b, &active_dofs_bound);
      // Update s_old.
      ComputeRatio(NE, us_b, u_b, s_old, active_elem_bound, active_dofs_bound);
      check_violation(s_old, s_min, s_max, "s-b-mult",
                      1e-12, &active_dofs_bound);
      clean_roundoff(s_min, s_max, s_old, &active_dofs_bound);
   }

   //
   // Bounded solution usq_b.
   //
   ParGridFunction usq_old_gf(K_bound.ParFESpace());
   Vector d_usq_HO(size), d_usq_LO, d_usq_b(size);
   Vector usq_b(size), usq_min(size), usq_max(size);
   if (levels > 2)
   {
      usq_old_gf = usq_old; usq_old_gf.ExchangeFaceNbrData();
      if (fct_solver_b->NeedsLOProductInput())
      {
         d_usq_LO.SetSize(size);
         lo_solver_b->CalcLOSolution(usq_old, d_usq_LO);
      }
      ho_solver_b->CalcHOSolution(usq_old, d_usq_HO);
      fct_solver_b->CalcFCTProduct(usq_old_gf, lumpedM, d_usq_HO, d_usq_LO,
                                   q_min, q_max, us_b,
                                   active_elem_bound,active_dofs_bound,d_usq_b);
      // Evolve usq_b, check violations.
      add(1.0, usq_old, dt, d_usq_b, usq_b);
      fct_solver_b->ScaleProductBounds(q_min, q_max, us_b,
                                       active_elem_bound, active_dofs_bound,
                                       usq_min, usq_max);
      check_violation(usq_b, usq_min, usq_max, "usq-b-mult",
                      1e-12, &active_dofs_bound);
      clean_roundoff(usq_min, usq_max, usq_b, &active_dofs_bound);
      // Update q_old.
      ComputeRatio(NE, usq_b, us_b, q_old, active_elem_bound,active_dofs_bound);
      check_violation(q_old, q_min, q_max, "q-b-mult",
                      1e-12, &active_dofs_bound);
      clean_roundoff(q_min, q_max, q_old, &active_dofs_bound);
   }

   if (evolve_sharp == false)
   { d_u = d_u_b; d_us = d_us_b; d_usq = d_usq_b; return; }

   //
   //  Sharpening starts here.
   //

   //
   // Sharp solution u_s (not in bounds).
   //
   Vector d_u_s(size), d_u_s_LO(size), d_u_s_HO(size), u_s(size);
   lo_solver_s->CalcLOSolution(u_old, d_u_s_LO);
   ho_solver_s->CalcHOSolution(u_old, d_u_s_HO);
   fct_solver_s->CalcFCTSolution(u_old_gf, lumpedM, d_u_s_HO, d_u_s_LO,
                                 u_min, u_max, d_u_s);
   add(1.0, u_old, dt, d_u_s, u_s);

   const bool local_avg = true;
   double Vol = 0.0, Mass = 0.0, Energy = 0.0;
   for (int i = 0; i < size; i++)
   {
      if (active_dofs_bound[i] == false) { continue; }
      if (local_avg && fabs(u_b(i) - u_s(i)) < 1e-12) { continue; }
      Vol    += u_b(i) * lumpedM(i);
      Mass   += us_b(i) * lumpedM(i);
      Energy += usq_b(i) * lumpedM(i);
   }
   MPI_Allreduce(MPI_IN_PLACE, &Vol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &Mass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &Energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   const double s_glob = Mass / Vol,
                q_glob = Energy / Mass;

   //
   // Blended solution u_new (in bounds).
   //
   Vector u_new(size);
   blend_global_u(levels, u_b, u_s, lumpedM,
                  u_min, u_max, active_dofs_bound,
                  s_min, s_max, us_b, s_glob,
                  q_min, q_max, usq_b, q_glob,
                  u_new);
   ComputeBoolIndicators(NE, u_b, active_elem_blend, active_dofs_blend);
   check_violation(u_new, u_min, u_max, "u-blend-mult", 1e-12, nullptr);
   clean_roundoff(u_min, u_max, u_new, &active_dofs_blend);
   for (int i = 0; i < size; i++) { d_u(i) = (u_new(i) - u_old(i)) / dt; }

   // Updated bounds for us.
   if (levels > 1)
   {
      fct_solver_b->ScaleProductBounds(s_min, s_max, u_new,
                                       active_elem_blend, active_dofs_blend,
                                       us_min, us_max);
   }

   //
   // Sharp solution us_s (not in bounds).
   //
   Vector d_us_s(size), us_s(size);
   if (levels > 1)
   {
      if (fct_solver_b->NeedsLOProductInput())
      {
         d_us_LO.SetSize(size);
         lo_solver_s->CalcLOSolution(us_old, d_us_LO);
      }
      ho_solver_s->CalcHOSolution(us_old, d_us_HO);
      fct_solver_s->CalcFCTProduct(us_old_gf, lumpedM, d_us_HO, d_us_LO,
                                   s_min, s_max, u_b,
                                   active_elem_blend, active_dofs_blend, d_us_s);
      add(1.0, us_old, dt, d_us_s, us_s);
   }

   //
   // Blended solution us_new (in bounds).
   //
   Vector us_new(size);
   if (levels > 1)
   {
      blend_global_us(levels, u_new, u_b, us_b, us_s,
                      lumpedM, us_min, us_max, active_dofs_blend,
                      q_min, q_max, usq_b, s_glob, q_glob,
                      us_new);
      check_violation(us_new, us_min, us_max, "us-blend-mult",
                      1e-12, &active_dofs_blend);
      clean_roundoff(us_min, us_max, us_new, &active_dofs_blend);

      // Update s_old.
      ComputeRatio(NE, us_new, u_new, s_old,
                   active_elem_bound, active_dofs_bound);
      check_violation(s_old, s_min, s_max, "s-blend-mult",
                      1e-12, &active_dofs_bound);
      clean_roundoff(s_min, s_max, s_old, &active_dofs_bound);

      for (int i = 0; i < size; i++) { d_us(i) = (us_new(i) - us_old(i))/dt; }
   }

   // Updated bounds for usq.
   if (levels > 2)
   {
      fct_solver_b->ScaleProductBounds(q_min, q_max, us_new,
                                       active_elem_blend, active_dofs_blend,
                                       usq_min, usq_max);
   }

   //
   // Sharp solution usq_s (not in bounds).
   //
   Vector d_usq_s(size), usq_s(size);
   if (levels > 2)
   {
      if (fct_solver_b->NeedsLOProductInput())
      {
         d_usq_LO.SetSize(size);
         lo_solver_s->CalcLOSolution(usq_old, d_usq_LO);
      }
      ho_solver_s->CalcHOSolution(usq_old, d_usq_HO);
      fct_solver_s->CalcFCTProduct(usq_old_gf, lumpedM, d_usq_HO, d_usq_LO,
                                   q_min, q_max, us_b,
                                   active_elem_blend, active_dofs_blend, d_usq_s);
      add(1.0, usq_old, dt, d_usq_s, usq_s);
   }

   //
   // Blended solution usq_new (in bounds).
   //
   Vector usq_new(size);
   if (levels > 2)
   {
      blend_global_usq(us_new, us_b, usq_b, usq_s, lumpedM, usq_min, usq_max,
                       active_dofs_blend, q_glob, usq_new);
      check_violation(usq_new, usq_min, usq_max, "usq-blend-mult",
                      1e-12, &active_dofs_blend);
      clean_roundoff(usq_min, usq_max, usq_new, &active_dofs_blend);

      // Update q_old.
      ComputeRatio(NE, usq_new, us_new, q_old,
                   active_elem_bound, active_dofs_bound);
      check_violation(q_old, q_min, q_max, "q-blend-mult",
                      1e-12, &active_dofs_bound);
      clean_roundoff(q_min, q_max, q_old, &active_dofs_bound);

      for (int i = 0; i < size; i++) { d_usq(i) = (usq_new(i) - usq_old(i))/dt; }
   }
}

void AdvectionOperator::UpdateTimeStepEstimate(const Vector &x,
                                               const Vector &dx,
                                               const Vector &x_min,
                                               const Vector &x_max) const
{
   if (dt_control == TimeStepControl::FixedTimeStep) { return; }

   // x_min <= x + dt * dx <= x_max.
   int n = x.Size();
   const double eps = 1e-12;

   double dt = numeric_limits<double>::infinity();

   for (int i = 0; i < n; i++)
   {
      if (dx(i) > eps)
      {
         dt = fmin(dt, (x_max(i) - x(i)) / dx(i) );
      }
      else if (dx(i) < -eps)
      {
         dt = fmin(dt, (x_min(i) - x(i)) / dx(i) );
      }
   }

   MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MIN,
                 K_sharp.ParFESpace()->GetComm());

   dt_est = fmin(dt_est, dt);
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
      case 18:
      case 19:
      {
         // Taylor-Green deformation used for mesh motion in remap tests.

         // Map [-1,1] to [0,1].
         for (int d = 0; d < dim; d++) { X(d) = X(d) * 0.5 + 0.5; }

         if (dim == 1)
         {
            v(0) = sin(M_PI*X(0));
         }
         else
         {
            if (problem_num == 18)
            {
               v(0) = 0.25 * sin(M_PI*X(0));
               v(1) = 0.25 * sin(M_PI*X(1));
               return;
            }
            if (problem_num == 19)
            {
               v(0) =  0.25 * sin(M_PI*X(0)) * cos(M_PI*X(1));
               v(1) = -0.25 * cos(M_PI*X(0)) * sin(M_PI*X(1));
               return;
            }

            v(0) =  sin(M_PI*X(0)) * cos(M_PI*X(1));
            v(1) = -cos(M_PI*X(0)) * sin(M_PI*X(1));
            if (dim == 3)
            {
               v(0) *= cos(M_PI*X(2));
               v(1) *= cos(M_PI*X(2));
               v(2) = 0.0;
            }
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

// Initial condition: hard-coded functions
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
      case 8:
      case 9:
      {
         switch (dim)
         {
            case 1:
               return (x(0) > 0.3 && x(0) < 0.7) ? 1.0 : 0.0;
            case 2:
            {
               if (problem_num == 18)
               {
                  double rad = std::sqrt((x(0)-0.2) * (x(0)-0.2) +
                                         (x(1)-0.2) * (x(1)-0.2));
                  return (rad <= 0.3) ? 1.0 : 0.0;
               }
               if (problem_num == 19)
               {
                  double rad = std::sqrt(x(0) * x(0) + x(1) * x(1));
                  return (rad <= 0.3) ? 1.0 : 0.0;
               }
               return 0.0;
            }
            case 3: MFEM_ABORT("not setup"); return 0.0;
         }
      }
   }
   return 0.0;
}

double s0_function(const Vector &x)
{
   // Simple nonlinear function.
   return 7.0;
   //return 2.0 + sin(2*M_PI * x(0)) * sin(2*M_PI * x(1));
}

double q0_function(const Vector &x)
{
   return 11.0;
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
