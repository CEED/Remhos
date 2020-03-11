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
//
//    Using lua problem definition file
//    ./remhos -p balls-and-jacks.lua -r 4 -dt 0.001 -tf 5.0
//
//    Transport mode:
//    ./remhos -m ./data/periodic-segment.mesh -p 0 -r 2 -dt 0.005
//    ./remhos -m ./data/periodic-square.mesh -p 0 -r 2 -dt 0.01 -tf 10
//    ./remhos -m ./data/periodic-hexagon.mesh -p 0 -r 2 -dt 0.01 -tf 10
//    ./remhos -m ./data/periodic-square.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ./remhos -m ./data/periodic-hexagon.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ./remhos -m ./data/star-q3.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ./remhos -m ./data/disc-nurbs.mesh -p 1 -r 3 -dt 0.005 -tf 9
//    ./remhos -m ./data/disc-nurbs.mesh -p 2 -r 3 -dt 0.005 -tf 9
//    ./remhos -m ./data/periodic-square.mesh -p 3 -r 4 -dt 0.0025 -tf 9 -vs 20
//    ./remhos -m ./data/periodic-cube.mesh -p 0 -r 2 -o 2 -dt 0.02 -tf 8
//    ./remhos -m ./data/periodic-square.mesh -p 4 -r 4 -dt 0.001 -o 2 -mt 3
//    ./remhos -m ./data/periodic-square.mesh -p 3 -r 2 -dt 0.0025 -o 15 -tf 9 -mt 4
//    ./remhos -m ./data/periodic-square.mesh -p 5 -r 4 -dt 0.002 -o 2 -tf 0.8 -mt 4
//    ./remhos -m ./data/periodic-cube.mesh -p 5 -r 5 -dt 0.0001 -o 1 -tf 0.8 -mt 4
//
//    Remap mode:
//    ./remhos -m ./data/periodic-square.mesh -p 10 -r 3 -dt 0.005 -tf 0.5 -mt 4 -vs 10
//    ./remhos -m ./data/periodic-square.mesh -p 14 -r 3 -dt 0.005 -tf 0.5 -mt 4 -vs 10
//
// Description:  This example code solves the time-dependent advection equation
//               du/dt + v.grad(u) = 0, where v is a given fluid velocity, and
//               u0(x)=u(0,x) is a given initial condition.
//
//               The example demonstrates the use of Discontinuous Galerkin (DG)
//               bilinear forms in MFEM (face integrators), the use of explicit
//               ODE time integrators, the definition of periodic boundary
//               conditions through periodic meshes, as well as the use of GLVis
//               for persistent visualization of a time-evolving solution. The
//               saving of time-dependent data files for external visualization
//               with VisIt (visit.llnl.gov) is also illustrated.

#include "mfem.hpp"
#include "miniapps/common/mfem-common.hpp"
#include <fstream>
#include <iostream>
#include "remhos_ho.hpp"
#include "remhos_lo.hpp"
#include "remhos_fct.hpp"
#include "remhos_mono.hpp"
#include "remhos_tools.hpp"

using namespace std;
using namespace mfem;

#ifdef USE_LUA
#include "lua.hpp"
lua_State* L;
#endif

enum class HOSolverType {Neumann, CG, LocalInverse};
enum class LOSolverType {None, DiscrUpwind, ResidDist};
enum class FCTSolverType {FluxBased, ClipScale, NonlinearPenalty};
enum class MonolithicSolverType {None, RDMonolithic};

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

// Inflow boundary condition
double inflow_function(const Vector &x);

// Mesh bounding box
Vector bb_min, bb_max;

void GetMinMax(const ParGridFunction &g, double &min, double &max);

// Utility function to build a map to the offset of the symmetric entry in a
// sparse matrix.
Array<int> SparseMatrix_Build_smap(const SparseMatrix &A)
{
   // Assuming that A is finalized
   const int *I = A.GetI(), *J = A.GetJ(), n = A.Size();
   Array<int> smap(I[n]);

   for (int row = 0, j = 0; row < n; row++)
   {
      for (int end = I[row+1]; j < end; j++)
      {
         int col = J[j];
         // Find the offset, _j, of the (col,row) entry and store it in smap[j].
         for (int _j = I[col], _end = I[col+1]; true; _j++)
         {
            MFEM_VERIFY(_j != _end, "Can't find the symmetric entry!");

            if (J[_j] == row) { smap[j] = _j; break; }
         }
      }
   }
   return smap;
}

// Given a matrix K, matrix D (initialized with same sparsity as K) is computed,
// such that (K+D)_ij >= 0 for i != j.
void ComputeDiscreteUpwindingMatrix(const SparseMatrix &K,
                                    Array<int> smap, SparseMatrix& D)
{
   const int *Ip = K.GetI(), *Jp = K.GetJ(), n = K.Size();
   const double *Kp = K.GetData();

   double *Dp = D.GetData();

   for (int i = 0, k = 0; i < n; i++)
   {
      double rowsum = 0.;
      for (int end = Ip[i+1]; k < end; k++)
      {
         int j = Jp[k];
         double kij = Kp[k];
         double kji = Kp[smap[k]];
         double dij = fmax(fmax(0.0,-kij),-kji);
         Dp[k] = kij + dij;
         Dp[smap[k]] = kji + dij;
         if (i != j) { rowsum += dij; }
      }
      D(i,i) = K(i,i) -rowsum;
   }
}

// Appropriate quadrature rule for faces according to DGTraceIntegrator.
const IntegrationRule *GetFaceIntRule(FiniteElementSpace *fes)
{
   int i, order;
   // Use the first mesh face and element as indicator.
   const FaceElementTransformations *Trans =
      fes->GetMesh()->GetFaceElementTransformations(0);
   const FiniteElement *el = fes->GetFE(0);

   if (Trans->Elem2No >= 0)
   {
      order = min(Trans->Elem1->OrderW(), Trans->Elem2->OrderW()) + 2*el->GetOrder();
   }
   else
   {
      order = Trans->Elem1->OrderW() + 2*el->GetOrder();
   }
   if (el->Space() == FunctionSpace::Pk)
   {
      order++;
   }
   return &IntRules.Get(Trans->FaceGeom, order);
}

// Class for local assembly of M_L M_C^-1 K, where M_L and M_C are the lumped
// and consistent mass matrices and K is the convection matrix. The spaces are
// assumed to be L2 conforming.
class PrecondConvectionIntegrator: public BilinearFormIntegrator
{
private:
#ifndef MFEM_THREAD_SAFE
   DenseMatrix dshape, adjJ, Q_ir;
   Vector shape, vec2, BdFidxT;
#endif
   VectorCoefficient &Q;
   double alpha;

public:
   PrecondConvectionIntegrator(VectorCoefficient &q, double a = 1.0)
      : Q(q) { alpha = a; }
   virtual void AssembleElementMatrix(const FiniteElement &,
                                      ElementTransformation &,
                                      DenseMatrix &);
};

// alpha (q . grad u, v)
class MixedConvectionIntegrator : public BilinearFormIntegrator
{
private:
#ifndef MFEM_THREAD_SAFE
   DenseMatrix dshape, adjJ, Q_ir;
   Vector shape, vec2, BdFidxT;
#endif
   VectorCoefficient &Q;
   double alpha;

public:
   MixedConvectionIntegrator(VectorCoefficient &q, double a = 1.0)
      : Q(q) { alpha = a; }
   virtual void AssembleElementMatrix2(const FiniteElement &tr_el,
                                       const FiniteElement &te_el,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat)
   {
      int tr_nd = tr_el.GetDof();
      int te_nd = te_el.GetDof();
      int dim = te_el.GetDim(); // Using test geometry.

#ifdef MFEM_THREAD_SAFE
      DenseMatrix dshape, adjJ, Q_ir;
      Vector shape, vec2, BdFidxT;
#endif
      elmat.SetSize(te_nd, tr_nd);
      dshape.SetSize(tr_nd,dim);
      adjJ.SetSize(dim);
      shape.SetSize(te_nd);
      vec2.SetSize(dim);
      BdFidxT.SetSize(tr_nd);

      Vector vec1;

      // Using midpoint rule and test geometry.
      const IntegrationRule *ir = &IntRules.Get(te_el.GetGeomType(), 1);

      Q.Eval(Q_ir, Trans, *ir);

      elmat = 0.0;
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         tr_el.CalcDShape(ip, dshape);
         te_el.CalcShape(ip, shape);

         Trans.SetIntPoint(&ip);
         CalcAdjugate(Trans.Jacobian(), adjJ);
         Q_ir.GetColumnReference(i, vec1);
         vec1 *= alpha * ip.weight;

         adjJ.Mult(vec1, vec2);
         dshape.Mult(vec2, BdFidxT);

         AddMultVWt(shape, BdFidxT, elmat);
      }
   }
};

/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form of du/dt = -v.grad(u) is M du/dt = K u + b, where M and K are the mass
    and advection matrices, and b describes the flow on the boundary. This can
    be written as a general ODE, du/dt = M^{-1} (K u + b), and this class is
    used to evaluate the right-hand side. */
class FE_Evolution : public TimeDependentOperator
{
private:
   BilinearForm &Mbf, &ml;
   ParBilinearForm &Kbf;
   SparseMatrix &M, &K;
   ParBilinearForm &M_HO, &K_HO;
   Vector &lumpedM;
   const GridFunction &inflow_gf;
   const Vector &b;

   Vector start_mesh_pos, start_submesh_pos;
   GridFunction &mesh_pos, *submesh_pos, &mesh_vel, &submesh_vel;

   mutable Vector z;
   mutable ParGridFunction x_gf;

   double dt;
   Assembly &asmbl;

   LowOrderMethod &lom;
   SmoothnessIndicator *smth_indicator;
   DofInfo &dofs;

   HOSolver &ho_solver;
   LOSolver *lo_solver;
   FCTSolver *fct_solver;
   MonolithicSolver *mono_solver;

public:
   FE_Evolution(BilinearForm &Mbf_, SparseMatrix &_M, BilinearForm &_ml,
                Vector &_lumpedM,
                ParBilinearForm &Kbf_, SparseMatrix &_K,
                ParBilinearForm &M_HO_, ParBilinearForm &K_HO_,
                const Vector &_b, const GridFunction &inflow,
                GridFunction &pos, GridFunction *sub_pos,
                GridFunction &vel, GridFunction &sub_vel,
                Assembly &_asmbl, LowOrderMethod &_lom, DofInfo &_dofs,
                SmoothnessIndicator *si,
                HOSolver &hos, LOSolver *los, FCTSolver *fct, MonolithicSolver *mos);

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual void SetDt(double _dt) { dt = _dt; }
   void SetRemapStartPos(const Vector &m_pos, const Vector &sm_pos)
   {
      start_mesh_pos    = m_pos;
      start_submesh_pos = sm_pos;
   }

   virtual ~FE_Evolution() { }
};

FE_Evolution* adv;

int main(int argc, char *argv[])
{
   // Initialize MPI.
   MPI_Session mpi(argc, argv);
   const int myid = mpi.WorldRank();

   // Parse command-line options.
#ifdef USE_LUA
   L = luaL_newstate();
   luaL_openlibs(L);
   const char* problem_file = "problem.lua";
#else
   problem_num = 4;
#endif
   const char *mesh_file = "data/periodic-square.mesh";
   int rs_levels = 2;
   int rp_levels = 0;
   int order = 3;
   int mesh_order = 2;
   int ode_solver_type = 3;
   MONOTYPE MonoType = MONOTYPE::None;
   HOSolverType ho_type   = HOSolverType::Neumann;
   LOSolverType lo_type   = LOSolverType::None;
   FCTSolverType fct_type = FCTSolverType::ClipScale;
   MonolithicSolverType mono_type = MonolithicSolverType::None;
   bool pa = false;
   bool OptScheme = true;
   int smth_ind_type = 0;
   double t_final = 4.0;
   double dt = 0.005;
   bool visualization = true;
   bool visit = false;
   bool binary = false;
   int vis_steps = 100;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
#ifdef USE_LUA
   args.AddOption(&problem_file, "-p", "--problem",
                  "lua problem definition file.");
#else
   args.AddOption(&problem_num, "-p", "--problem",
                  "Problem setup to use. See options in velocity_function().");
#endif
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
   args.AddOption((int*)(&MonoType), "-mt", "--MonoType",
                  "Monotonicity scheme: 0 - no monotonicity treatment,\n\t"
                  "                     1 - discrete upwinding - LO,\n\t"
                  "                     2 - discrete upwinding - FCT,\n\t"
                  "                     3 - residual distribution - LO,\n\t"
                  "                     4 - residual distribution - FCT,n\t"
                  "                     5 - residual distribution - monolithic.");
   args.AddOption((int*)(&ho_type), "-ho", "--ho-type",
                  "High-Order Solver: 0 - Neumann iteration,\n\t"
                  "                   1 - CG solver,\n\t"
                  "                   2 - Local inverse.");
   args.AddOption((int*)(&lo_type), "-lo", "--lo-type",
                  "Low-Order Solver: 0 - None,\n\t"
                  "                  1 - Discrete Upwind,\n\t"
                  "                  2 - Residual Distribution.");
   args.AddOption((int*)(&fct_type), "-fct", "--fct-type",
                  "Correction type: 0 - Flux-based FCT,\n\t"
                  "                 1 - Local clip + scale,\n\t"
                  "                 2 - Local clip + nonlinear penalization.");
   args.AddOption((int*)(&mono_type), "-mono", "--mono-type",
                  "Monolithic solver: 0 - No monolithic solver,\n\t"
                  "                   1 - Residual distribution.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly",
                  "Enable or disable partial assembly for the HO solution.");
   args.AddOption(&OptScheme, "-sc", "--subcell", "-el", "--element",
                  "Optimized low order scheme: PDU / RDS VS DU / RD.");
   args.AddOption(&smth_ind_type, "-si", "--smth_ind",
                  "Smoothness indicator: 0 - no smoothness indicator,\n\t"
                  "                      1 - approx_quadratic,\n\t"
                  "                      2 - exact_quadratic.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&binary, "-binary", "--binary-datafiles", "-ascii",
                  "--ascii-datafiles",
                  "Use binary (Sidre) or ascii format for VisIt data files.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   // When not using lua, exec mode is derived from problem number convention
   if (problem_num < 10)      { exec_mode = 0; }
   else if (problem_num < 20) { exec_mode = 1; }
   else { MFEM_ABORT("Unspecified execution mode."); }

#ifdef USE_LUA
   // When using lua, exec mode is read from lua file
   if (luaL_dofile(L, problem_file))
   {
      printf("Error opening lua file: %s\n",problem_file);
      exit(1);
   }

   lua_getglobal(L, "exec_mode");
   if (!lua_isnumber(L, -1))
   {
      printf("Did not find exec_mode in lua input.\n");
      return 1;
   }
   exec_mode = (int)lua_tonumber(L, -1);
#endif

   // TODO remove MONOTYPE and use the other enums.
   if (MonoType == MONOTYPE::DiscUpw || MonoType == MONOTYPE::DiscUpw_FCT)
   {
      lo_type = LOSolverType::DiscrUpwind;
   }
   else if (MonoType == MONOTYPE::ResDist || MonoType == MONOTYPE::ResDist_FCT)
   {
      lo_type = LOSolverType::ResidDist;
   }
   if (MonoType == MONOTYPE::ResDist_Monolithic)
   {
      mono_type = MonolithicSolverType::RDMonolithic;
   }

   // Read the serial mesh from the given mesh file on all processors.
   // Refine the mesh in serial to increase the resolution.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   const int dim = mesh->Dimension();
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   mesh->GetBoundingBox(bb_min, bb_max, max(order, 1));

   // Parallel partitioning of the mesh.
   // Refine the mesh further in parallel to increase the resolution.
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < rp_levels; lev++) { pmesh.UniformRefinement(); }

   // 3. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.
   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK2Solver(1.0); break;
      case 3: ode_solver = new RK3SSPSolver; break;
      case 4: ode_solver = new RK4Solver; break;
      case 6: ode_solver = new RK6Solver; break;
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

   // Mesh velocity.
   GridFunction v_gf(x.FESpace());
   VectorGridFunctionCoefficient v_coef(&v_gf);

   // If remap is on, obtain the mesh velocity by moving the mesh to the final
   // mesh positions, and taking the displacement vector.
   // The mesh motion resembles a time-dependent deformation, e.g., similar to
   // a deformation that is obtained by a Lagrangian simulation.
   if (exec_mode == 1)
   {
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
      x = x0;
   }

   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   const int btype = BasisType::Positive;
   DG_FECollection fec(order, dim, btype);
   ParFiniteElementSpace pfes(&pmesh, &fec);

   // The min and max bounds are represented as CG functions of the same order
   // as the solution, thus having 1:1 dof correspondence inside each element.
   H1_FECollection fec_bounds(max(order, 1), dim, BasisType::GaussLobatto);
   ParFiniteElementSpace pfes_bounds(&pmesh, &fec_bounds);

   // Check for meaningful combinations of parameters.
   bool fail = false;
   if (MonoType != MONOTYPE::None)
   {
      if (((int)MonoType != MonoType) || (MonoType < 0) || (MonoType > 5))
      {
         if (myid == 0) { cout << "Unsupported option for monotonicity treatment." << endl; }
         fail = true;
      }
      if (btype != 2)
      {
         if (myid == 0) { cout << "Monotonicity treatment requires Bernstein basis." << endl; }
         fail = true;
      }
      if (order == 0)
      {
         // Disable monotonicity treatment for piecewise constants.
         if (myid == 0) { mfem_warning("For -o 0, monotonicity treatment is disabled."); }
         MonoType = MONOTYPE::None;
         OptScheme = false;
      }
   }
   else { OptScheme = false; }

   if ((MonoType > 2) && (order==1) && OptScheme)
   {
      // Avoid subcell methods for linear elements.
      if (myid == 0) { mfem_warning("For -o 1, subcell scheme is disabled."); }
      OptScheme = false;
   }

   if (fail)
   {
      delete ode_solver;
      return 5;
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
   ParBilinearForm m(&pfes);
   m.AddDomainIntegrator(new MassIntegrator);

   ParBilinearForm M_HO(&pfes);
   ConstantCoefficient one(1.0);
   M_HO.AddDomainIntegrator(new MassIntegrator(one));

   ParBilinearForm k(&pfes);
   ParBilinearForm K_HO(&pfes);
   if (exec_mode == 0)
   {
      k.AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
      K_HO.AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
   }
   else if (exec_mode == 1)
   {
      k.AddDomainIntegrator(new ConvectionIntegrator(v_coef));
      K_HO.AddDomainIntegrator(new ConvectionIntegrator(v_coef));
   }

   // TODO: decide what to do with assembly.LinearFluxLumping.
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
         DGTraceIntegrator *dgt_i = new DGTraceIntegrator(v_coef, -1.0, -0.5);
         DGTraceIntegrator *dgt_b = new DGTraceIntegrator(v_coef, -1.0, -0.5);
         K_HO.AddInteriorFaceIntegrator(new TransposeIntegrator(dgt_i));
         K_HO.AddBdrFaceIntegrator(new TransposeIntegrator(dgt_b));
      }

      K_HO.KeepNbrBlock(true);
   }

   if (pa)
   {
      M_HO.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      K_HO.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }

   M_HO.Assemble();
   K_HO.Assemble(0);

   if (pa == false)
   {
      M_HO.Finalize();
      K_HO.Finalize(0);
   }

   ParLinearForm b(&pfes);
   b.AddBdrFaceIntegrator(
      new BoundaryFlowIntegrator(inflow, v_coef, -1.0, -0.5));

   // Compute the lumped mass matrix.
   Vector lumpedM;
   ParBilinearForm ml(&pfes);
   ml.AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator));
   ml.Assemble();
   ml.Finalize();
   ml.SpMat().GetDiag(lumpedM);

   m.Assemble();
   m.Finalize();
   int skip_zeros = 0;
   k.Assemble(skip_zeros);
   k.Finalize(skip_zeros);
   b.Assemble();

   // Store topological dof data.
   DofInfo dofs(&pfes, &pfes_bounds);

   // Precompute data required for high and low order schemes. This could be put
   // into a separate routine. I am using a struct now because the various
   // schemes require quite different information.
   LowOrderMethod lom;
   lom.MonoType = MonoType;
   lom.OptScheme = OptScheme;
   lom.fes = &pfes;

   lom.pk = NULL;
   if (lom.MonoType == DiscUpw || lom.MonoType == DiscUpw_FCT)
   {
      if (!lom.OptScheme)
      {
         lom.smap = SparseMatrix_Build_smap(k.SpMat());
         lom.D = k.SpMat();

         if (exec_mode == 0)
         {
            ComputeDiscreteUpwindingMatrix(k.SpMat(), lom.smap, lom.D);
         }
      }
      else
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
   }
   if (exec_mode == 1) { lom.coef = &v_coef; }
   else                { lom.coef = &velocity; }

   lom.irF = GetFaceIntRule(&pfes);

   DG_FECollection fec0(0, dim, btype);
   DG_FECollection fec1(1, dim, btype);

   lom.subcell_mesh = NULL;
   lom.SubFes0 = NULL;
   lom.SubFes1 = NULL;
   FiniteElementCollection *fec_sub;
   ParFiniteElementSpace *pfes_sub;
   ParGridFunction *xsub;
   ParGridFunction v_sub_gf;
   VectorGridFunctionCoefficient v_sub_coef;
   Vector x0_sub;

   if (order > 1)
   {
      // The mesh corresponding to Bezier subcells of order p is constructed.
      // NOTE: The mesh is assumed to consist of quads or hexes.
      MFEM_VERIFY(order > 1, "This code should not be entered for order = 1.");

      // Get a uniformly refined mesh.
      lom.subcell_mesh = new ParMesh(&pmesh, order, BasisType::ClosedUniform);

      // Check if the mesh is periodic.
      const L2_FECollection *L2_coll = dynamic_cast<const L2_FECollection *>
                                       (pmesh.GetNodes()->FESpace()->FEColl());
      if (L2_coll == NULL)
      {
         // Standard non-periodic mesh.
         // Note that the fine mesh is always linear.
         fec_sub = new H1_FECollection(1, dim, BasisType::ClosedUniform);
         pfes_sub = new ParFiniteElementSpace(lom.subcell_mesh, fec_sub, dim);
         xsub = new ParGridFunction(pfes_sub);
         lom.subcell_mesh->SetCurvature(1);
         lom.subcell_mesh->SetNodalGridFunction(xsub);
      }
      else
      {
         // Periodic mesh - the node positions must be corrected after the call
         // to the above Mesh constructor. Note that the fine mesh is always
         // linear.
         const bool disc_nodes = true;
         lom.subcell_mesh->SetCurvature(1, disc_nodes);

         fec_sub = new L2_FECollection(1, dim, BasisType::ClosedUniform);
         pfes_sub = new ParFiniteElementSpace(lom.subcell_mesh, fec_sub, dim);
         xsub = new ParGridFunction(pfes_sub);
         lom.subcell_mesh->SetNodalGridFunction(xsub);

         GridFunction *coarse = pmesh.GetNodes();
         InterpolationGridTransfer transf(*coarse->FESpace(), *pfes_sub);
         transf.ForwardOperator().Mult(*coarse, *xsub);
      }

      lom.SubFes0 = new FiniteElementSpace(lom.subcell_mesh, &fec0);
      lom.SubFes1 = new FiniteElementSpace(lom.subcell_mesh, &fec1);

      // Submesh velocity.
      v_sub_gf.SetSpace(pfes_sub);
      v_sub_gf.ProjectCoefficient(velocity);

      // Zero it out on boundaries (not moving boundaries).
      Array<int> ess_bdr, ess_vdofs;
      if (lom.subcell_mesh->bdr_attributes.Size() > 0)
      {
         ess_bdr.SetSize(lom.subcell_mesh->bdr_attributes.Max());
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
         lom.VolumeTerms = new MixedConvectionIntegrator(velocity, -1.0);
      }
      else if (exec_mode == 1)
      {
         lom.VolumeTerms = new MixedConvectionIntegrator(v_sub_coef);
      }
   }
   else { lom.subcell_mesh = &pmesh; }

   Assembly asmbl(dofs, lom, inflow_gf, pfes, exec_mode);
   const int ne = pmesh.GetNE();

   // Monolithic limiting correction factors.
   if (lom.MonoType == ResDist_Monolithic)
   {
      lom.scale.SetSize(ne);

      for (int e = 0; e < ne; e++)
      {
         const FiniteElement* el = pfes.GetFE(e);
         DenseMatrix velEval;
         Vector vval;
         double vmax = 0.;
         ElementTransformation *tr = pmesh.GetElementTransformation(e);
         int qOrdE = tr->OrderW() + 2*el->GetOrder() + 2*max(tr->OrderGrad(el), 0);
         const IntegrationRule *irE = &IntRules.Get(el->GetGeomType(), qOrdE);
         velocity.Eval(velEval, *tr, *irE);

         for (int l = 0; l < irE->GetNPoints(); l++)
         {
            velEval.GetColumnReference(l, vval);
            vmax = max(vmax, vval.Norml2());
         }
         lom.scale(e) = vmax / (2. * (sqrt(dim) * pmesh.GetElementSize(e) / order));
      }
   }

   LOSolver *lo_solver = NULL;
   Array<int> lo_smap;
   if (lo_type == LOSolverType::DiscrUpwind)
   {
      SparseMatrix *lo_spmat = (OptScheme) ? &lom.pk->SpMat() : &k.SpMat();
      lo_smap = SparseMatrix_Build_smap(*lo_spmat);
      bool update_D = (exec_mode == 0) ? false : true;
      lo_solver = new DiscreteUpwind(pfes, *lo_spmat, lo_smap,
                                     lumpedM, asmbl, update_D);
   }
   else if (lo_type == LOSolverType::ResidDist)
   {
      bool subcell = (OptScheme) ? true : false;
      bool dynamic = (exec_mode == 0) ? false : true;
      lo_solver = new ResidualDistribution(pfes, k, asmbl, lumpedM,
                                           subcell, dynamic);
   }

   // Initial condition.
   ParGridFunction u(&pfes);
   FunctionCoefficient u0(u0_function);
   u.ProjectCoefficient(u0);

   // Smoothness indicator.
   H1_FECollection H1fec(1, dim, btype);
   ParFiniteElementSpace H1fes(lom.subcell_mesh, &H1fec);
   SmoothnessIndicator *smth_indicator = NULL;
   if (smth_ind_type)
   {
      smth_indicator = new SmoothnessIndicator(smth_ind_type, H1fes,
                                               pfes, u, dofs);
   }

   HOSolver *ho_solver;
   if (ho_type == HOSolverType::Neumann)
   {
      ho_solver = new NeumannHOSolver(pfes, m, k, lumpedM, asmbl);
   }
   else if (ho_type == HOSolverType::CG)
   {
      ho_solver = new CGHOSolver(pfes, M_HO, K_HO);
   }
   else if (ho_type == HOSolverType::LocalInverse)
   {
      ho_solver = new LocalInverseHOSolver(pfes, M_HO, K_HO);
   }
   else { MFEM_ABORT("Wrong high-order solver type specification."); }


   MonolithicSolver *mono_solver = NULL;
   if (mono_type == MonolithicSolverType::RDMonolithic)
   {
      bool subcell = (OptScheme) ? true : false;
      bool dynamic = (exec_mode == 0) ? false : true;
      bool mass_lim = (problem_num != 6 && problem_num != 7) ? true : false;
      mono_solver = new MonoRDSolver(pfes, k.SpMat(), m.SpMat(), lumpedM,
                                     asmbl, smth_indicator, lom.scale,
                                     subcell, dynamic, mass_lim);
   }

   // Print the starting meshes and initial condition.
   {
      ofstream meshHO("meshHO_init.mesh");
      meshHO.precision(precision);
      pmesh.PrintAsOne(meshHO);
      if (lom.subcell_mesh)
      {
         ofstream meshLO("meshLO_init.mesh");
         meshLO.precision(precision);
         lom.subcell_mesh->PrintAsOne(meshLO);
      }
      ofstream sltn("sltn_init.gf");
      sltn.precision(precision);
      u.SaveAsOne(sltn);
   }

   // Create data collection for solution output: either VisItDataCollection for
   // ascii data files, or SidreDataCollection for binary data files.
   DataCollection *dc = NULL;
   if (visit)
   {
      if (binary)
      {
#ifdef MFEM_USE_SIDRE
         dc = new SidreDataCollection("Example9", &pmesh);
#else
         MFEM_ABORT("Must build with MFEM_USE_SIDRE=YES for binary output.");
#endif
      }
      else
      {
         dc = new VisItDataCollection("Example9", &pmesh);
         dc->SetPrecision(precision);
      }
      dc->RegisterField("solution", &u);
      dc->SetCycle(0);
      dc->SetTime(0.0);
      dc->Save();
   }

   socketstream sout;
   char vishost[] = "localhost";
   int  visport   = 19916;
   if (visualization)
   {
      // Make sure all MPI ranks have sent their 'v' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh.GetComm());

      sout.precision(8);

      int Wx = 0, Wy = 0; // window position
      const int Ww = 350, Wh = 350; // window size
      common::VisualizeField(sout, vishost, visport, u,
                             "Solution", Wx, Wy, Ww, Wh);
   }

   // check for conservation
   Vector masses(lumpedM);
   const double initialMass_loc = lumpedM * u;
   double initialMass;
   MPI_Allreduce(&initialMass_loc, &initialMass, 1, MPI_DOUBLE, MPI_SUM,
                 pmesh.GetComm());

   // 8. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).

   Array<int> K_HO_smap;
   FCTSolver *fct_solver = NULL;
   if (MonoType == DiscUpw_FCT || MonoType == ResDist_FCT)
   {
      if (fct_type == FCTSolverType::FluxBased)
      {
         MFEM_VERIFY(pa == false, "Flux-based FCT and PA are incompatible.");

         K_HO_smap = SparseMatrix_Build_smap(K_HO.SpMat());
         const int fct_iterations = 1;
         fct_solver = new FluxBasedFCT(pfes, smth_indicator, dt, K_HO.SpMat(),
                                       K_HO_smap, M_HO.SpMat(), fct_iterations);
      }
      else if (fct_type == FCTSolverType::ClipScale)
      {
         fct_solver = new ClipScaleSolver(pfes, smth_indicator, dt);
      }
      else if (fct_type == FCTSolverType::NonlinearPenalty)
      {
         fct_solver = new NonlinearPenaltySolver(pfes, smth_indicator, dt);
      }
   }

   FE_Evolution* adv = new FE_Evolution(m, m.SpMat(), ml, lumpedM,
                                        k, k.SpMat(), M_HO, K_HO,
                                        b, inflow_gf, x, xsub, v_gf, v_sub_gf,
                                        asmbl, lom, dofs, smth_indicator,
                                        *ho_solver, lo_solver,
                                        fct_solver, mono_solver);

   double t = 0.0;
   adv->SetTime(t);
   ode_solver->Init(*adv);

   double umin, umax;
   GetMinMax(u, umin, umax);

   if (exec_mode == 1)
   {
      adv->SetRemapStartPos(x0, x0_sub);

      // For remap, the pseudotime always evolves from 0 to 1.
      t_final = 1.0;
   }

   ParGridFunction res = u;
   double residual;

   bool done = false;
   for (int ti = 0; !done; )
   {
      double dt_real = min(dt, t_final - t);

      adv->SetDt(dt_real);

      ode_solver->Step(u, t, dt_real);
      ti++;

      // Monotonicity check for debug purposes mainly.
      if (MonoType != MONOTYPE::None && smth_indicator == NULL)
      {
         double umin_new, umax_new;         
         GetMinMax(u, umin_new, umax_new);
         if (problem_num % 10 != 6 && problem_num % 10 != 7)
         {
            MFEM_VERIFY(umin_new > umin - 1e-12,
                        "Undershoot of " << umin - umin_new);
            MFEM_VERIFY(umax_new < umax + 1e-12,
                        "Overshoot of " << umax_new - umax);
            umin = umin_new;
            umax = umax_new;
         }
         else
         {
            MFEM_VERIFY(umin_new > 0.0 - 1e-12,
                        "Undershoot of " << 0.0 - umin_new);
            MFEM_VERIFY(umax_new < 1.0 + 1e-12,
                        "Overshoot of " << umax_new - 1.0);
         }
      }

      if (exec_mode == 1)
      {
         add(x0, t, v_gf, x);
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
         for (int i = 0; i < res.Size(); i++)
         {
            res_loc += pow( (lumpedM(i) * u(i) / dt) - (lumpedM(i) * res(i) / dt), 2. );
         }
         MPI_Allreduce(&res_loc, &residual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

         residual = sqrt(residual);
         if (residual < 1.e-12 && t >= 1.) { done = true; u = res; }
         else { res = u; }
      }

      if (done || ti % vis_steps == 0)
      {
         if (myid == 0)
         {
            cout << "time step: " << ti << ", time: " << t << ", residual: "
                 << residual << endl;
         }

         if (visualization)
         {
            int Wx = 0, Wy = 0; // window position
            int Ww = 350, Wh = 350; // window size
            common::VisualizeField(sout, vishost, visport,
                                   u, "Solution", Wx, Wy, Ww, Wh);
         }
         if (visit)
         {
            dc->SetCycle(ti);
            dc->SetTime(t);
            dc->Save();
         }
      }
   }

   // Print the final meshes and solution.
   {
      ofstream meshHO("meshHO_final.mesh");
      meshHO.precision(precision);
      pmesh.PrintAsOne(meshHO);
      if (asmbl.subcell_mesh)
      {
         ofstream meshLO("meshLO_final.mesh");
         meshLO.precision(precision);
         asmbl.subcell_mesh->Print(meshLO);
      }
      ofstream sltn("sltn_final.gf");
      sltn.precision(precision);
      u.SaveAsOne(sltn);
   }

   // Check for mass conservation.
   double finalMass_loc(0);
   if (exec_mode == 1)
   {
      ml.BilinearForm::operator=(0.0);
      ml.Assemble();
      ml.SpMat().GetDiag(lumpedM);
      finalMass_loc = lumpedM * u;
   }
   else { finalMass_loc = masses * u; }
   double finalMass;
   MPI_Allreduce(&finalMass_loc, &finalMass, 1, MPI_DOUBLE, MPI_SUM,
                 pmesh.GetComm());
   const double umax_loc = u.Max();
   MPI_Allreduce(&umax_loc, &umax, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
   if (myid == 0)
   {
      cout << setprecision(10)
           << "Final mass: " << finalMass << endl
           << "Max value:  " << umax << endl << setprecision(6)
           << "Mass loss:  " << abs(initialMass - finalMass) << endl;
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

   // 10. Free the used memory.
   delete adv;
   delete mono_solver;
   delete fct_solver;
   delete smth_indicator;
   delete ho_solver;

   delete ode_solver;
   delete mesh_fec;
   delete lom.pk;
   delete dc;

   if (order > 1)
   {
      delete lom.subcell_mesh;
      delete fec_sub;
      delete pfes_sub;
      delete xsub;
      delete lom.SubFes0;
      delete lom.SubFes1;
      delete lom.VolumeTerms;
   }

   return 0;
}

void Assembly::NonlinFluxLumping(const int k, const int nd,
                                 const int BdrID, const Vector &x,
                                 Vector &y, const Vector &x_nd,
                                 const Vector &alpha) const
{
   int i, j, dofInd;
   double xNeighbor, SumCorrP = 0., SumCorrN = 0., eps = 1.E-15;
   const int size_x = x.Size();
   Vector xDiff(dofs.numFaceDofs), BdrTermCorr(dofs.numFaceDofs);
   BdrTermCorr = 0.;

   for (j = 0; j < dofs.numFaceDofs; j++)
   {
      dofInd = k*nd+dofs.BdrDofs(j,BdrID);
      const int nbr_dof_id = dofs.NbrDof(k, BdrID, j);
      // Note that if the boundary is outflow, we have bdrInt = 0 by definition,
      // s.t. this value will not matter.
      if (nbr_dof_id < 0) { xNeighbor = inflow_gf(dofInd); }
      else
      {
         xNeighbor = (nbr_dof_id < size_x) ? x(nbr_dof_id)
                     : x_nd(nbr_dof_id - size_x);
      }
      xDiff(j) = xNeighbor - x(dofInd);
   }

   for (i = 0; i < dofs.numFaceDofs; i++)
   {
      dofInd = k*nd+dofs.BdrDofs(i,BdrID);
      for (j = 0; j < dofs.numFaceDofs; j++)
      {
         y(dofInd) += bdrInt(k, BdrID, i*dofs.numFaceDofs + j) * xDiff(i);
         BdrTermCorr(i) += bdrInt(k, BdrID,
                                  i*dofs.numFaceDofs + j) * (xDiff(j)-xDiff(i));
      }
      BdrTermCorr(i) *= alpha(dofs.BdrDofs(i,BdrID));
      SumCorrP += max(0., BdrTermCorr(i));
      SumCorrN += min(0., BdrTermCorr(i));
   }

   for (i = 0; i < dofs.numFaceDofs; i++)
   {
      dofInd = k*nd+dofs.BdrDofs(i,BdrID);
      if (SumCorrP + SumCorrN > eps)
      {
         BdrTermCorr(i) = min(0., BdrTermCorr(i)) -
                          max(0., BdrTermCorr(i)) * SumCorrN / SumCorrP;
      }
      else if (SumCorrP + SumCorrN < -eps)
      {
         BdrTermCorr(i) = max(0., BdrTermCorr(i)) -
                          min(0., BdrTermCorr(i)) * SumCorrP / SumCorrN;
      }
      y(dofInd) += BdrTermCorr(i);
   }
}

// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(BilinearForm &Mbf_, SparseMatrix &_M,
                           BilinearForm &_ml, Vector &_lumpedM,
                           ParBilinearForm &Kbf_, SparseMatrix &_K,
                           ParBilinearForm &M_HO_, ParBilinearForm &K_HO_,
                           const Vector &_b, const GridFunction &inflow,
                           GridFunction &pos, GridFunction *sub_pos,
                           GridFunction &vel, GridFunction &sub_vel,
                           Assembly &_asmbl,
                           LowOrderMethod &_lom, DofInfo &_dofs,
                           SmoothnessIndicator *si,
                           HOSolver &hos, LOSolver *los, FCTSolver *fct,
                           MonolithicSolver *mos) :
   TimeDependentOperator(_M.Size()), Mbf(Mbf_), Kbf(Kbf_), ml(_ml),
   M(_M), K(_K), lumpedM(_lumpedM),
   M_HO(M_HO_), K_HO(K_HO_),
   inflow_gf(inflow), b(_b),
   start_mesh_pos(pos.Size()), start_submesh_pos(sub_vel.Size()),
   mesh_pos(pos), submesh_pos(sub_pos),
   mesh_vel(vel), submesh_vel(sub_vel),
   z(_M.Size()), x_gf(Kbf.ParFESpace()),
   asmbl(_asmbl), lom(_lom), dofs(_dofs), smth_indicator(si),
   ho_solver(hos), lo_solver(los), fct_solver(fct), mono_solver(mos) { }

void FE_Evolution::Mult(const Vector &x, Vector &y) const
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
      // Reset precomputed geometric data.
      Mbf.GetFES()->GetMesh()->DeleteGeometricFactors();

      // Reassemble on the new mesh. Element contributions.
      // TODO: remove these.
      // Currently needed to have the sparse matrices used by the LO methods.
      Mbf.BilinearForm::operator=(0.0);
      Mbf.Assemble();
      Kbf.BilinearForm::operator=(0.0);
      Kbf.Assemble(0);
      ml.BilinearForm::operator=(0.0);
      ml.Assemble();
      ml.SpMat().GetDiag(lumpedM);

      M_HO.BilinearForm::operator=(0.0);
      M_HO.Assemble();
      K_HO.BilinearForm::operator=(0.0);
      K_HO.Assemble(0);

      if (lom.pk)
      {
         lom.pk->BilinearForm::operator=(0.0);
         lom.pk->Assemble();
      }

      // Face contributions.
      asmbl.bdrInt = 0.;
      Mesh *mesh = lom.fes->GetMesh();
      const int dim = mesh->Dimension(), ne = lom.fes->GetNE();
      Array<int> bdrs, orientation;
      FaceElementTransformations *Trans;

      for (int k = 0; k < ne; k++)
      {
         if (dim == 1)      { mesh->GetElementVertices(k, bdrs); }
         else if (dim == 2) { mesh->GetElementEdges(k, bdrs, orientation); }
         else if (dim == 3) { mesh->GetElementFaces(k, bdrs, orientation); }

         for (int i = 0; i < dofs.numBdrs; i++)
         {
            Trans = mesh->GetFaceElementTransformations(bdrs[i]);
            asmbl.ComputeFluxTerms(k, i, Trans, lom);
         }
      }
   }

   x_gf = x;
   x_gf.ExchangeFaceNbrData();

   if (mono_solver)
   {
      mono_solver->CalcSolution(x, y);
      return;
   }

   if (lom.MonoType == 0)
   {
      ho_solver.CalcHOSolution(x, y);
   }
   else
   {
      if (lom.MonoType % 2 == 1)
      {
         lo_solver->CalcLOSolution(x, y);
      }
      else if (lom.MonoType % 2 == 0)
      {
         Vector yH(x.Size()), yL(x.Size());

         lo_solver->CalcLOSolution(x, yL);

         ho_solver.CalcHOSolution(x, yH);

         dofs.ComputeBounds();
         fct_solver->CalcFCTSolution(x_gf, lumpedM, yH, yL,
                                     dofs.xi_min, dofs.xi_max, y);
      }
   }
}

#ifdef USE_LUA
void lua_velocity_function(const Vector &x, Vector &v)
{
   lua_getglobal(L, "velocity_function");
   int dim = x.Size();

   lua_pushnumber(L, x(0));
   if (dim > 1)
   {
      lua_pushnumber(L, x(1));
   }
   if (dim > 2)
   {
      lua_pushnumber(L, x(2));
   }

   double v0 = 0;
   double v1 = 0;
   double v2 = 0;
   lua_call(L, dim, dim);
   v0 = (double)lua_tonumber(L, -1);
   lua_pop(L, 1);
   if (dim > 1)
   {
      v1 = (double)lua_tonumber(L, -1);
      lua_pop(L, 1);
   }
   if (dim > 2)
   {
      v2 = (double)lua_tonumber(L, -1);
      lua_pop(L, 1);
   }

   v(0) = v0;
   if (dim > 1)
   {
      v(0) = v1;
      v(1) = v0;
   }
   if (dim > 2)
   {
      v(0) = v2;
      v(1) = v1;
      v(2) = v0;
   }
}
#endif

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
#ifdef USE_LUA
   lua_velocity_function(x, v);
   return;
#endif

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
      case 12:
      case 13:
      case 14:
      case 15:
      case 16:
      case 10:
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

void Assembly::LinearFluxLumping(const int k, const int nd, const int BdrID,
                                 const Vector &x, Vector &y, const Vector &x_nd,
                                 const Vector &alpha) const
{
   int i, j, dofInd;
   double xNeighbor;
   Vector xDiff(dofs.numFaceDofs);
   const int size_x = x.Size();

   for (j = 0; j < dofs.numFaceDofs; j++)
   {
      dofInd = k*nd+dofs.BdrDofs(j,BdrID);
      const int nbr_dof_id = dofs.NbrDof(k, BdrID, j);
      // Note that if the boundary is outflow, we have bdrInt = 0 by definition,
      // s.t. this value will not matter.
      if (nbr_dof_id < 0) { xNeighbor = inflow_gf(dofInd); }
      else
      {
         xNeighbor = (nbr_dof_id < size_x) ? x(nbr_dof_id)
                                           : x_nd(nbr_dof_id - size_x);
      }
      xDiff(j) = xNeighbor - x(dofInd);
   }

   for (i = 0; i < dofs.numFaceDofs; i++)
   {
      dofInd = k*nd+dofs.BdrDofs(i,BdrID);
      for (j = 0; j < dofs.numFaceDofs; j++)
      {
         // alpha=0 is the low order solution, alpha=1, the Galerkin solution.
         // 0 < alpha < 1 can be used for limiting within the low order method.
         y(dofInd) += bdrInt(k, BdrID, i*dofs.numFaceDofs + j) *
                      (xDiff(i) + (xDiff(j)-xDiff(i)) *
                       alpha(dofs.BdrDofs(i,BdrID)) *
                       alpha(dofs.BdrDofs(j,BdrID)));
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

// Initial condition as defined by lua function
#ifdef USE_LUA
double lua_u0_function(const Vector &x)
{
   lua_getglobal(L, "initial_function");
   int dim = x.Size();

   lua_pushnumber(L, x(0));
   if (dim > 1)
   {
      lua_pushnumber(L, x(1));
   }
   if (dim > 2)
   {
      lua_pushnumber(L, x(2));
   }

   lua_call(L, dim, 1);
   double u = (double)lua_tonumber(L, -1);
   lua_pop(L, 1);

   return u;
}
#endif

// Initial condition: lua function or hard-coded functions
double u0_function(const Vector &x)
{
#ifdef USE_LUA
   return lua_u0_function(x);
#endif

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

#ifdef USE_LUA
double lua_inflow_function(const Vector& x)
{
   lua_getglobal(L, "boundary_condition");

   int dim = x.Size();

   double t;
   adv ? t = adv->GetTime() : t = 0.0;

   for (int d = 0; d < dim; d++)
   {
      lua_pushnumber(L, x(d));
   }
   lua_pushnumber(L, t);

   lua_call(L, dim+1, 1);
   double u = (double)lua_tonumber(L, -1);
   lua_pop(L, 1);

   return u;
}
#endif

double inflow_function(const Vector &x)
{
#ifdef USE_LUA
   return lua_inflow_function(x);
#endif

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

void PrecondConvectionIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
   int i, nd = el.GetDof(), dim = el.GetDim();

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape, adjJ, Q_ir;
   Vector shape, vec2, BdFidxT;
#endif
   elmat.SetSize(nd);
   dshape.SetSize(nd,dim);
   adjJ.SetSize(dim);
   shape.SetSize(nd);
   vec2.SetSize(dim);
   BdFidxT.SetSize(nd);

   double w;
   Vector vec1;
   DenseMatrix mass(nd,nd), conv(nd,nd), lumpedM(nd,nd), tmp(nd,nd);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = Trans.OrderGrad(&el) + Trans.Order() + el.GetOrder();
      order = max(order, 2 * el.GetOrder() + Trans.OrderW());
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   Q.Eval(Q_ir, Trans, *ir);

   conv = mass = 0.0;
   for (i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);
      el.CalcShape(ip, shape);

      Trans.SetIntPoint(&ip);
      CalcAdjugate(Trans.Jacobian(), adjJ);
      Q_ir.GetColumnReference(i, vec1);
      vec1 *= alpha * ip.weight;

      adjJ.Mult(vec1, vec2);
      dshape.Mult(vec2, BdFidxT);

      AddMultVWt(shape, BdFidxT, conv);

      w = Trans.Weight() * ip.weight;
      AddMult_a_VVt(w, shape, mass);
   }
   lumpedM = mass;
   lumpedM.Lump();
   mass.Invert();

   MultABt(mass, lumpedM, tmp);
   MultAtB(tmp, conv, elmat); // using symmetry of mass matrix
}

void GetMinMax(const ParGridFunction &g, double &min, double &max)
{
   double min_loc = g.Min(), max_loc = g.Max();
   MPI_Allreduce(&min_loc, &min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   MPI_Allreduce(&max_loc, &max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
}
