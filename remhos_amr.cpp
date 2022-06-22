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

#define MFEM_DEBUG_COLOR 51
#include "debug.hpp"

#include <unistd.h>

#include "mfem.hpp"
#include "remhos_amr.hpp"
#include "linalg/sparsemat.hpp"

using namespace std;
using namespace mfem;

namespace mfem
{

namespace amr
{

/// ESTIMATOR ///
static const char *EstimatorName(const int est)
{
   switch (static_cast<amr::Estimator>(est))
   {
      case amr::Estimator::ZZ: return "ZZ";
      case amr::Estimator::L2ZZ: return "L2ZZ";
      case amr::Estimator::JJt: return "JJt";
      case amr::Estimator::Custom: return "Custom";
      case amr::Estimator::DRL4AMR: return "DRL4AMR";
      default: MFEM_ABORT("Unknown estimator!");
   }
   return nullptr;
}

EstimatorIntegrator::EstimatorIntegrator(ParMesh &pmesh,
                                         const Options &opt,
                                         const FluxMode mode):
   DiffusionIntegrator(one),
   NE(pmesh.GetNE()),
   e2(0),
   pmesh(pmesh),
   mode(mode),
   opt(opt) { dbg(); }

void EstimatorIntegrator::Reset() { e2 = 0; NE = pmesh.GetNE(); }

double EstimatorIntegrator::ComputeFluxEnergy(const FiniteElement &el,
                                              ElementTransformation &Tr,
                                              Vector &flux, Vector *d_energy)
{
   if (mode == FluxMode::diffusion)
   {
      return DiffusionIntegrator::ComputeFluxEnergy(el, Tr, flux, d_energy);
   }
   // Not implemented for other modes
   MFEM_ABORT("Not implemented!");
   return 0.0;
}

void EstimatorIntegrator::ComputeElementFlux1(const FiniteElement &el,
                                              ElementTransformation &Tr,
                                              const Vector &u,
                                              const FiniteElement &fluxelem,
                                              Vector &flux)
{
   const int dof = el.GetDof();
   const int dim = el.GetDim();
   const int sdim = Tr.GetSpaceDim();

   DenseMatrix dshape(dof, dim);
   DenseMatrix invdfdx(dim, sdim);
   Vector vec(dim), pointflux(sdim);

   const IntegrationRule &ir = fluxelem.GetNodes();
   const int NQ = ir.GetNPoints();
   flux.SetSize(NQ * sdim);

   for (int q = 0; q < NQ; q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      el.CalcDShape(ip, dshape);
      dshape.MultTranspose(u, vec);

      Tr.SetIntPoint (&ip);
      CalcInverse(Tr.Jacobian(), invdfdx);
      invdfdx.MultTranspose(vec, pointflux);

      for (int d = 0; d < sdim; d++)
      {
         flux(NQ*d+q) = pointflux(d);
      }
   }
}

void EstimatorIntegrator::ComputeElementFlux2(const int e,
                                              const FiniteElement &el,
                                              ElementTransformation &Tr,
                                              const FiniteElement &fluxelem,
                                              Vector &flux)
{
   const int dim = el.GetDim();
   const int sdim = Tr.GetSpaceDim();

   DenseMatrix Jadjt(dim, sdim), Jadj(dim, sdim);

   const IntegrationRule &ir = fluxelem.GetNodes();
   const int NQ = ir.GetNPoints();
   flux.SetSize(NQ * sdim);

   constexpr double NL_DMAX = std::numeric_limits<double>::max();
   double minW = +NL_DMAX;
   double maxW = -NL_DMAX;

   const int depth = pmesh.pncmesh->GetElementDepth(e);

   for (int q = 0; q < NQ; q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      Tr.SetIntPoint(&ip);
      CalcAdjugate(Tr.Jacobian(), Jadj);
      Jadjt = Jadj;
      Jadjt.Transpose();
      const double w = Jadjt.Weight();
      minW = std::fmin(minW, w);
      maxW = std::fmax(maxW, w);
      MFEM_VERIFY(std::fabs(maxW) > 1e-13, "");
      const double rho = minW / maxW;
      MFEM_VERIFY(rho <= 1.0, "");
      for (int d = 0; d < sdim; d++)
      {
         const int iq = NQ*d + q;
         flux(iq) = 1.0 - rho;
         if (rho > opt.ref_threshold) { continue; }
         if (depth > opt.max_level) { continue; }
         flux(iq) = rho;
      }
   }
}

void EstimatorIntegrator::ComputeElementFlux(const FiniteElement &el,
                                             ElementTransformation &Trans,
                                             Vector &u,
                                             const FiniteElement &fluxelem,
                                             Vector &flux,
                                             bool with_coef,
                                             const IntegrationRule */*ir*/)
{
   // ZZ comes with with_coef set to true
   switch (mode)
   {
      case FluxMode::diffusion:
      {
         DiffusionIntegrator::ComputeElementFlux(el, Trans, u,
                                                 fluxelem, flux, with_coef);
         break;
      }
      case FluxMode::one:
      {
         ComputeElementFlux1(el, Trans, u, fluxelem, flux);
         break;
      }
      case FluxMode::two:
      {
         MFEM_VERIFY(NE == pmesh.GetNE(), "");
         ComputeElementFlux2(e2++, el, Trans, fluxelem, flux);
         break;
      }
      default: MFEM_ABORT("Unknown mode!");
   }
}

/// AMR Update for ZZ/L2ZZ Estimator
void Operator::ApplyZZ()
{
   if (refiner->Apply(pmesh))
   {
      mesh_refined = true;
      dbg("Refined!");
      return;
   }

   if (derefiner->Apply(pmesh))
   {
      mesh_derefined = true;
      dbg("DeRefined!");
      return;
   }
   dbg("Nothing to do");
}

/// AMR OPERATOR
Operator::Operator(ParFiniteElementSpace &pfes,
                   ParMesh &pmesh,
                   ParGridFunction &u,
                   const Options &opt):
   pfes(pfes),
   pmesh(pmesh),
   u(u),
   myid(pmesh.GetMyRank()),
   dim(pmesh.Dimension()),
   sdim(pmesh.SpaceDimension()),
   opt(opt),
   h1_fec(new H1_FECollection(opt.mesh_order, dim)),
   l2_fec(new L2_FECollection(opt.mesh_order, dim)),
   flux_fec(opt.estimator == Estimator::ZZ ? h1_fec : l2_fec),
   flux_fes(new ParFiniteElementSpace(&pmesh, flux_fec, dim)),
   quad_JJt(Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL))
{
   constexpr int SDIM = 3;
   constexpr bool discont = false;
   constexpr int ordering = Ordering::byVDIM;
   quad_JJt.SetCurvature(lorder_JJt, discont, SDIM, ordering);

   dbg("AMR Setup: order:%d, mesh_order:%d", opt.order, opt.mesh_order);
   if (myid == 0)
   {
      std::cout << "AMR setup with "
                << amr::EstimatorName(opt.estimator) << " estimator"
                << std::endl;
   }

   if (opt.estimator == Estimator::ZZ || opt.estimator == Estimator::L2ZZ)
   {
      integ = new EstimatorIntegrator(pmesh, opt);
   }

   if (opt.estimator == Estimator::ZZ)
   {
      estimator = new ZienkiewiczZhuEstimator(*integ, u, *flux_fes);
   }

   if (opt.estimator == Estimator::L2ZZ)
   {
      smooth_flux_fec = new RT_FECollection(opt.order-1, dim);
      auto smooth_flux_fes = new ParFiniteElementSpace(&pmesh, smooth_flux_fec);
      estimator = new L2ZienkiewiczZhuEstimator(*integ, u, flux_fes,
                                                smooth_flux_fes);
   }

   if (estimator)
   {
      refiner = new ThresholdRefiner(*estimator);
      refiner->SetTotalErrorFraction(0.0); // use purely local threshold
      refiner->SetLocalErrorGoal(opt.ref_threshold);
      refiner->PreferConformingRefinement();
      refiner->SetNCLimit(opt.nc_limit);

      derefiner = new ThresholdDerefiner(*estimator);
      //derefiner->SetOp(1); // 0:min, 1:sum, 2:max
      derefiner->SetThreshold(opt.deref_threshold);
      derefiner->SetNCLimit(opt.nc_limit);

      refiner->Reset();
      derefiner->Reset();
   }
}

Operator::~Operator()
{
   dbg();
   delete h1_fec;
   delete l2_fec;
   delete flux_fes;
   delete smooth_flux_fec;
   delete estimator;
   delete refiner;
   delete derefiner;
   delete integ;
}

/// RESET
void Operator::Reset()
{
   dbg();
   mesh_refined = false;
   mesh_derefined = false;
   if (integ) { integ->Reset(); }
   if (refiner) { refiner->Reset(); }
   if (derefiner) { derefiner->Reset(); }
}

/// APPLY
void Operator::Apply(Array<Refinement> input_refs)
{
   dbg("%s", EstimatorName(opt.estimator));

   mesh_refined = false;
   mesh_derefined = false;

   refs.DeleteAll();
   derefs.SetSize(pfes.GetNE());
   derefs = 0.0; // tie to 0.0 to trig the bool tests in Update

   switch (opt.estimator)
   {
      case Estimator::ZZ:
      case Estimator::L2ZZ:    { ApplyZZ(); break; }
      case Estimator::JJt:     { ApplyJJt(); break; }
      case Estimator::Custom:  { ApplyCustom(); break; }
      case Estimator::DRL4AMR: { ApplyDRL4AMR(input_refs); break; }
      default: MFEM_ABORT("Unknown AMR estimator!");
   }

   const bool use_ref_array = opt.estimator == Estimator::JJt ||
                              opt.estimator == Estimator::DRL4AMR ;
   if (!use_ref_array) { dbg("mesh_refined/derefined set by ZZ"); return; }

   const int nref = pmesh.ReduceInt(refs.Size());
   mesh_refined = nref > 0;
   dbg("mesh_refined: %s", mesh_refined ? "yes" : "no");

   MPI_Comm comm = pmesh.GetComm();
   double derefs_max_loc = derefs.Max();
   MPI_Allreduce(&derefs_max_loc, &derefs_max, 1, MPI_DOUBLE, MPI_MAX, comm);

   mesh_derefined = !mesh_refined &&
                    pmesh.GetLastOperation() == Mesh::REFINE &&
                    derefs_max > 0.0;
   dbg("use_ref_array: %s, derefs.Max():%f",
       use_ref_array ? "yes" : "no",
       derefs_max);
   dbg("mesh_derefined: %s", mesh_derefined ? "yes" : "no");
}

/// Refined
bool Operator::Refined()
{
   dbg("%s!", mesh_refined ? "yes" : "no");
   return mesh_refined;
}

/// DeRefined
bool Operator::DeRefined()
{
   dbg("%s!", mesh_derefined ? "yes" : "no");
   return mesh_derefined;
}

/// Main AMR Update Entry Point
void Operator::Update(AdvectionOperator &adv,
                      ODESolver *ode_solver,
                      BlockVector &S,
                      Array<int> &offset,
                      LowOrderMethod &lom,
                      ParMesh *subcell_mesh,
                      ParFiniteElementSpace *pfes_sub,
                      ParGridFunction *xsub,
                      ParGridFunction &v_sub_gf,
                      const double mass0_u,
                      ParGridFunction &inflow_gf,
                      FunctionCoefficient &inflow)
{
   assert(mesh_refined || mesh_derefined);

   // custom uses refs, ZZs will set mesh_refined
   const int nref = pmesh.ReduceInt(refs.Size());
   dbg("nref:%d",nref);

   if (nref > 0 && mesh_refined)
   {
      dbg("Using 'refs' array for GeneralRefinement");
      constexpr int non_conforming = 1;
      pmesh.GetNodes()->HostReadWrite();
      pmesh.GeneralRefinement(refs, non_conforming, opt.nc_limit);
      mesh_refined = true;
      if (myid == 0)
      {
         std::cout << "\033[32mRefined " << nref
                   << " elements.\033[m" << std::endl;
      }
   }
   else if (mesh_derefined && derefs_max > 0.0)
   {
      dbg("Got at least one (%f), NE:%d!", derefs_max, pmesh.GetNE());
      const int NE = pmesh.GetNE();
      const double threshold = opt.deref_threshold;

      Vector local_err(NE);
      local_err = threshold;
      for (int e = 0; e < NE; e++)
      {
         const double rho = derefs(e);
         local_err(e) = 1.0 - rho;
      }
      const int op = 2; // 0:min, 1:sum, 2:max
      mesh_derefined =
         pmesh.DerefineByError(local_err, threshold, 0, op);

      if (myid == 0 && mesh_derefined)
      {
         std::cout << "\033[31m DE-Refined.\033[m" << std::endl;
         if (opt.nc_limit) { MFEM_VERIFY(mesh_derefined,""); }
      }
   }
   else if ((opt.estimator == amr::Estimator::ZZ ||
             opt.estimator == amr::Estimator::L2ZZ) && mesh_derefined)
   {
      dbg("ZZ derefinement");
   }
   else { dbg("Nothing done before real Update"); }

   UpdateAndRebalance(S, offset, lom, subcell_mesh, pfes_sub, xsub, v_sub_gf);

   if (opt.estimator == Estimator::JJt &&
       ((derefs_max > 0.0) && !mesh_derefined && !mesh_refined))
   {
      dbg("Updates aborted!");
      derefs = 0.0;
      return;
   }

   inflow_gf.Update();
   inflow_gf.ProjectCoefficient(inflow);

   //pmesh.Rebalance();

   adv.AMRUpdate(S, u, mass0_u);

   ode_solver->Init(adv);
}

/// Internall AMR update
void Operator::UpdateAndRebalance(BlockVector &S,
                                  Array<int> &offset,
                                  LowOrderMethod &/*lom*/,
                                  ParMesh */*subcell_mesh*/,
                                  ParFiniteElementSpace */*pfes_sub*/,
                                  ParGridFunction *xsub,
                                  ParGridFunction &/*v_sub_gf*/)
{
   dbg("AMR Operator Update");
   Vector tmp;
   u.GetTrueDofs(tmp);
   u.SetFromTrueDofs(tmp);

   if (mesh_refined)
   {
      dbg("REFINE");
      pfes.Update();

      const int vsize = pfes.GetVSize();
      MFEM_VERIFY(offset.Size() == 2, "!product_sync vs offset size error!");
      offset[0] = 0;
      offset[1] = vsize;

      BlockVector S_bkp(S);
      S.Update(offset, Device::GetMemoryType());

      const mfem::Operator *R = pfes.GetUpdateOperator();

      R->Mult(S_bkp.GetBlock(0), S.GetBlock(0));

      u.MakeRef(&pfes, S, offset[0]);
      MFEM_VERIFY(u.Size() == vsize,"");
      u.SyncMemory(S);
   }
   else if (derefs_max > 0.0 || mesh_derefined)
   {
      if (!mesh_derefined && opt.estimator == Estimator::JJt)
      {
         dbg("Derefinement aborted!");
         return;
      }

      dbg("DEREFINE");
      ParGridFunction U = u;

      pfes.Update();

      const int vsize = pfes.GetVSize();
      MFEM_VERIFY(offset.Size() == 2, "!product_sync vs offset size error!");
      offset[0] = 0;
      offset[1] = vsize;


      S.Update(offset, Device::GetMemoryType());

      const mfem::Operator *R = pfes.GetUpdateOperator();
      assert(R);

      OperatorHandle Th;
      pfes.GetUpdateOperator(Th);
      SparseMatrix *Pth = Th.As<SparseMatrix>();
      assert(Pth);

      u.MakeRef(&pfes, S, offset[0]);
      MFEM_VERIFY(u.Size() == vsize,"");
      u.SyncMemory(S);

      R->Mult(U, u);
   }
   else { assert(false); }

   u.GetTrueDofs(tmp);
   u.SetFromTrueDofs(tmp);

   if (pfes.GetProlongationMatrix())
   {
      dbg("Constrain slave nodes!");
      Vector y(pfes.GetTrueVSize());
      pfes.GetRestrictionMatrix()->Mult(u, y);
      pfes.GetProlongationMatrix()->Mult(y, u);
   }

   if (xsub)
   {
      dbg("\033[31mXSUB!!");
      xsub->ParFESpace()->Update();
      xsub->Update();
   }
}

/// AMR Update for DRL4AMR Estimator
void Operator::ApplyDRL4AMR(Array<Refinement> &input_refs)
{
   dbg("refs <= input_refs");
   refs = input_refs;
}

/// AMR Update for Custom Estimator
void Operator::ApplyCustom()
{
   double umin_new, umax_new;
   GetMinMax(u, umin_new, umax_new);
   const double range = umax_new - umin_new;

   //dbg("u: [%f,%f], range: %f", umin_new, umax_new, range);
   MFEM_VERIFY(range > 0.0, "Range error!");
   //dbg("opt.max_level: %d",opt.max_level);
   Vector u_max, u_min;
   GetPerElementMinMax(u, u_min, u_max);
   const double ref_threshold = opt.ref_threshold;
   const double deref_threshold = opt.deref_threshold;
   for (int e = 0; e < pfes.GetNE(); e++)
   {
      const int depth = pmesh.pncmesh->GetElementDepth(e);
      //dbg("#%d (@%d) in [%f,%f]", e, depth, u_min(e), u_max(e));
      const double delta = (u_max(e) - u_min(e)) / range;
      dbg("#%d (@%d) %.2f [%.2f, %.2f]", e, depth, delta,
          ref_threshold, deref_threshold);
      if ((delta > ref_threshold) && depth < opt.max_level )
      {
         dbg("\033[32mRefinement #%d",e);
         refs.Append(Refinement(e));
      }
      if ((delta < deref_threshold) && depth > 0 )
      {
         dbg("\033[31mDeRefinement #%d",e);
         derefs(e) = 1.0;
      }
   }
}

/// AMR Update for JJt Estimator
void Operator::ApplyJJt()
{
   dbg("JJt");
   const bool vis = false;
   const int horder = opt.order;
   // The refinement factor, an integer > 1
   const int ref_factor = 2;
   // Specify the positions of the new vertices.
   const int ref_type = BasisType::ClosedUniform; // ClosedUniform || GaussLobatto

   dbg("order:%d, ref_factor:%d, lorder:%d", horder, ref_factor, lorder_JJt);

   // Create the low-order refined mesh
   ParMesh mesh_lor(ParMesh::MakeRefined(pmesh, ref_factor, ref_type));

   L2_FECollection fec_ho(horder, dim);
   L2_FECollection fec_lo(lorder_JJt, dim);
   L2_FECollection fec0(0, dim);

   ParFiniteElementSpace fes_ho(&pmesh, &fec_ho);
   ParFiniteElementSpace fes_lo(&mesh_lor, &fec_lo);
   ParFiniteElementSpace fes_rf(&pmesh, &fec0);

   ParGridFunction rho_ho(&fes_ho);
   ParGridFunction rho_lo(&fes_lo);
   ParGridFunction rho_rf(&fes_rf);

   // this is what we need: the 'solution' coming back is of size 'NE'
   assert(rho_rf.Size() == pmesh.GetNE());
   dbg("pmesh.GetNE:%d, u:%d, rho_ho:%d", pmesh.GetNE(), u.Size(), rho_ho.Size());

   const bool L2projection = true;

   const IntegrationRule &ir_ho = IntRules.Get(Geometry::SQUARE, horder + 1);
   const IntegrationRule &ir_lo = IntRules.Get(Geometry::SQUARE, lorder_JJt + 1);

   if (L2projection)
   {
      dbg("L2 projection");
      rho_ho = 0.0;
      ParBilinearForm a(&fes_ho);
      a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      a.AddDomainIntegrator(new MassIntegrator(&ir_ho));
      a.Assemble();

      ParLinearForm b(&fes_ho);
      GridFunctionCoefficient gf_coeff(&u);
      b.AddDomainIntegrator(new DomainLFIntegrator(gf_coeff, &ir_ho));
      b.Assemble();

      Vector B, X;
      OperatorPtr A;
      Array<int> no_bc;
      a.FormLinearSystem(no_bc, rho_ho, b, A, X, B);
      OperatorJacobiSmoother M(a, no_bc);
      const int print_iter = 0;
      const int max_num_iter = 200;
      const double RTOL = 1e-18, ATOL = 0.0;
      PCG(*A, M, B, X, print_iter, max_num_iter, RTOL, ATOL);
      a.RecoverFEMSolution(X, b, rho_ho);
   }
   else
   {
      rho_ho.ProjectGridFunction(u);
   }
   if (vis) VisualizeField(amr_vis[0], host, port, rho_ho, "rho_ho",
                              Wx, Wy, Ww, Wh, keys);

   L2ProjectionGridTransfer forward(fes_ho, fes_lo);
   L2ProjectionGridTransfer backward(fes_rf, fes_lo);
   const mfem::Operator &R = forward.ForwardOperator();
   const mfem::Operator &P = backward.BackwardOperator();

   R.Mult(rho_ho, rho_lo);
   if (vis) VisualizeField(amr_vis[1], host, port, rho_lo, "rho_lo",
                              Wx+Ww, Wy, Ww, Wh, keys);

   // now work on LOR mesh
   Array<int> dofs;
   Vector vals, elemvect, coords;
   GridFunction &nodes_lor = *mesh_lor.GetNodes();
   const int nip = ir_lo.GetNPoints();
   assert(nip == 4);

   constexpr int DIM = 2;
   constexpr int SDIM = 3;
   DenseMatrix Jadjt, Jadj(DIM, SDIM);

   GridFunction &q_nodes = *quad_JJt.GetNodes();
   constexpr double NL_DMAX = std::numeric_limits<double>::max();

   for (int e = 0; e < mesh_lor.GetNE(); e++)
   {
      rho_lo.GetValues(e, ir_lo, vals);
      fes_lo.GetElementVDofs(e, dofs);
      elemvect.SetSize(dofs.Size());
      assert(elemvect.Size() == 4);
      assert(vals.Size() == 4);

      for (int q = 0; q < nip; q++)
      {
         const IntegrationPoint &ip = ir_lo.IntPoint(q);
         ElementTransformation *eTr = mesh_lor.GetElementTransformation(e);
         eTr->SetIntPoint(&ip);
         nodes_lor.GetVectorValue(e, ip, coords);
         const double z = rho_lo.GetValue(e, ip);
         assert(coords.Size() == 2);
         q_nodes(3*q + 0) = coords[0];
         q_nodes(3*q + 1) = coords[1];
         q_nodes(3*q + 2) = z;
      }

      // Focus now on the quad mesh
      const int qe = 0;
      double minW = +NL_DMAX;
      double maxW = -NL_DMAX;
      ElementTransformation *eTr = quad_JJt.GetElementTransformation(qe);
      for (int q = 0; q < nip; q++) // nip == 4
      {
         eTr->SetIntPoint(&ir_lo.IntPoint(q));
         const DenseMatrix &J = eTr->Jacobian();
         CalcAdjugate(J, Jadj);
         Jadjt = Jadj;
         Jadjt.Transpose();
         const double w = Jadjt.Weight();
         minW = std::fmin(minW, w);
         maxW = std::fmax(maxW, w);
      }
      assert(std::fabs(maxW) > 0.0);
      const double rho = minW / maxW;
      //dbg("#%d rho:%f", e, rho);
      assert(rho > 0.0 && rho <= 1.0);
      elemvect = rho;
      rho_lo.SetSubVector(dofs, elemvect);
   }

   P.Mult(rho_lo, rho_rf);
   if (vis) VisualizeField(amr_vis[2], host, port, rho_rf, "rho_rf",
                              Wx + 2*Ww, Wy, Ww, Wh, keys);

   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      const int depth = pmesh.pncmesh->GetElementDepth(e);
      const double rho = rho_rf(e);
      dbg("#%d %.4e @ %d/%d",e, rho, depth, opt.max_level);

      if ((rho < opt.ref_threshold) && depth < opt.max_level )
      {
         dbg("\033[32mRefining #%d",e);
         refs.Append(Refinement(e));
      }
      if ((rho > opt.deref_threshold) && depth > 0 )
      {
         dbg("\033[31mTag for de-refinement #%d",e);
         derefs(e) = rho;
      }
   }
}

} // namespace amr

} // namespace mfem
