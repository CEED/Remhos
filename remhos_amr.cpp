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

#include "remhos.hpp"

using namespace std;
using namespace mfem;

namespace mfem
{

namespace amr
{

static const char *EstimatorName(const int est)
{
   switch (static_cast<amr::estimator>(est))
   {
      case amr::estimator::custom: return "Custom";
      case amr::estimator::zz: return "ZZ";
      case amr::estimator::kelly: return "Kelly";
      default: MFEM_ABORT("Unknown estimator!");
   }
   return nullptr;
}

EstimatorIntegrator::EstimatorIntegrator(ParMesh &pmesh,
                                         const int max_level,
                                         const double jac_threshold,
                                         const mode flux_mode):
   DiffusionIntegrator(one),
   NE(pmesh.GetNE()),
   pmesh(pmesh),
   flux_mode(flux_mode),
   max_level(max_level),
   jac_threshold(jac_threshold) { dbg(); }

void EstimatorIntegrator::Reset() { e = 0; NE = pmesh.GetNE(); }

double EstimatorIntegrator::ComputeFluxEnergy(const FiniteElement &fluxelem,
                                              ElementTransformation &Trans,
                                              Vector &flux, Vector *d_energy)
{
   if (flux_mode == mode::diffusion)
   {
      return DiffusionIntegrator::ComputeFluxEnergy(fluxelem, Trans, flux, d_energy);
   }
   // Not implemented for other modes
   MFEM_ABORT("Not implemented!");
   return 0.0;
}

void EstimatorIntegrator::ComputeElementFlux1(const FiniteElement &el,
                                              ElementTransformation &Trans,
                                              const Vector &u,
                                              const FiniteElement &fluxelem,
                                              Vector &flux)
{
   const int dof = el.GetDof();
   const int dim = el.GetDim();
   const int sdim = Trans.GetSpaceDim();

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

      Trans.SetIntPoint (&ip);
      CalcInverse(Trans.Jacobian(), invdfdx);
      invdfdx.MultTranspose(vec, pointflux);

      for (int d = 0; d < sdim; d++)
      {
         flux(NQ*d+q) = pointflux(d);
      }
   }
}

void EstimatorIntegrator::ComputeElementFlux2(const int e,
                                              const FiniteElement &el,
                                              ElementTransformation &Trans,
                                              const FiniteElement &fluxelem,
                                              Vector &flux)
{
   const int dim = el.GetDim();
   const int sdim = Trans.GetSpaceDim();

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
      Trans.SetIntPoint(&ip);
      CalcAdjugate(Trans.Jacobian(), Jadj);
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
         if (rho > jac_threshold) { continue; }
         if (depth > max_level) { continue; }
         flux(iq) = rho;
      }
   }
}

void EstimatorIntegrator::ComputeElementFlux(const FiniteElement &el,
                                             ElementTransformation &Trans,
                                             Vector &u,
                                             const FiniteElement &fluxelem,
                                             Vector &flux,
                                             bool with_coef)
{
   MFEM_VERIFY(NE == pmesh.GetNE(), "");
   // ZZ comes with with_coef set to true, not Kelly
   switch (flux_mode)
   {
      case mode::diffusion:
      {
         DiffusionIntegrator::ComputeElementFlux(el, Trans, u,
                                                 fluxelem, flux, with_coef);
         break;
      }
      case mode::one:
      {
         ComputeElementFlux1(el, Trans, u, fluxelem, flux);
         break;
      }
      case mode::two:
      {
         ComputeElementFlux2(e++, el, Trans, fluxelem, flux);
         break;
      }
      default: MFEM_ABORT("Unknown mode!");
   }
}

static void GetPerElementMinMax(const ParGridFunction &gf,
                                Vector &elem_min, Vector &elem_max,
                                int int_order = -1)
{
   const FiniteElementSpace *fes = gf.FESpace();
   const int ne = fes->GetNE();
   elem_min.SetSize(ne);
   elem_max.SetSize(ne);
   Vector vals;

   if (int_order < 0) { int_order = fes->GetOrder(0) + 1; }

   for (int e = 0; e < ne; e++)
   {
      const int geom = fes->GetFE(e)->GetGeomType();
      const IntegrationRule &ir = IntRules.Get(geom, int_order);
      gf.GetValues(e, ir, vals);
      elem_min(e) = vals.Min();
      elem_max(e) = vals.Max();
      //dbg("#%d %f [%f,%f], vals:",e, gf(e), elem_min(e), elem_max(e)); vals.Print();
   }
}

Operator::Operator(ParFiniteElementSpace &pfes,
                   ParFiniteElementSpace &mesh_pfes,
                   ParMesh &pmesh,
                   ParGridFunction &x,
                   ParGridFunction &xsub,
                   ParGridFunction &sol,
                   int est,
                   double ref_t, double jac_t, double deref_t,
                   int max_level, int nc_limit):

   pmesh(pmesh),
   x(x),
   xsub(xsub),
   sol(sol),
   pfes(pfes),
   mesh_pfes(mesh_pfes),
   myid(pmesh.GetMyRank()),
   dim(pmesh.Dimension()),
   sdim(pmesh.SpaceDimension()),
   flux_fec(order, dim),
   flux_fes(&pmesh, &flux_fec, sdim),
   opt( {est, ref_t, jac_t, deref_t, max_level, nc_limit})
{
   dbg("AMR Setup");
   if (myid == 0)
   {
      std::cout << "AMR setup with "
                << amr::EstimatorName(opt.estimator) << " estimator"
                << std::endl;
   }

   if (opt.estimator == amr::estimator::zz)
   {
      integ = new amr::EstimatorIntegrator(pmesh, opt.max_level,
                                           opt.jac_threshold);
      smooth_flux_fec = new RT_FECollection(order-1, dim);
      auto smooth_flux_fes = new ParFiniteElementSpace(&pmesh, smooth_flux_fec);
      estimator = new L2ZienkiewiczZhuEstimator(*integ, sol, &flux_fes,
                                                smooth_flux_fes);
   }

   if (opt.estimator == amr::estimator::kelly)
   {
      integ = new amr::EstimatorIntegrator(pmesh, opt.max_level,
                                           opt.jac_threshold);
      estimator = new KellyErrorEstimator(*integ, sol, flux_fes);
   }

   if (estimator)
   {
      const double hysteresis = 0.25;
      const double max_elem_error = 1.0e-6;
      refiner = new ThresholdRefiner(*estimator);
      refiner->SetTotalErrorFraction(0.0);
      refiner->SetLocalErrorGoal(max_elem_error);
      refiner->PreferConformingRefinement();
      refiner->SetNCLimit(opt.nc_limit);

      derefiner = new ThresholdDerefiner(*estimator);
      derefiner->SetOp(2); // 0:min, 1:sum, 2:max
      derefiner->SetThreshold(hysteresis * max_elem_error);
      derefiner->SetNCLimit(opt.nc_limit);
   }
}

Operator::~Operator() { dbg(); }

void Operator::Reset()
{
   dbg();
   if (integ) { integ->Reset(); }
   if (refiner) { refiner->Reset(); }
   if (derefiner) { derefiner->Reset(); }
}

void Operator::Update(AdvectionOperator &adv,
                      ODESolver *ode_solver,
                      BlockVector &S,
                      Array<int> &offset,
                      ParGridFunction &u)
{
   dbg();
   double umin_new, umax_new;
   GetMinMax(u, umin_new, umax_new);
   const double range = umax_new - umin_new;
   //dbg("u: [%f,%f], range: %f", umin_new, umax_new, range);
   MFEM_VERIFY(range > 0.0, "Range error!");

   Array<Refinement> refs;
   bool mesh_refined = false;
   const FiniteElementSpace *fes = u.FESpace();
   const int NE = fes->GetNE();
   //ParFiniteElementSpace &pfes = *u.ParFESpace();
   //constexpr double NL_DMAX = std::numeric_limits<double>::max();

   switch (opt.estimator)
   {
      case amr::estimator::custom:
      {
         Vector u_max, u_min;
         GetPerElementMinMax(u, u_min, u_max);
         const double threshold = opt.ref_threshold;
         for (int e = 0; e < NE; e++)
         {
            //dbg("u(%d) in [%f,%f]", e, u_max(e), u_min(e));
            const double delta = (u_max(e) - u_min(e)) / range;
            //dbg("delta: %f, threshold: %f", delta, threshold);
            if (delta > opt.ref_threshold &&
                pmesh.pncmesh->GetElementDepth(e) < opt.max_level )
            {
               dbg("\033[32mRefinement #%d",e);
               refs.Append(Refinement(e));
            }
         }
         break;
      }

      case amr::estimator::zz:
      case amr::estimator::kelly:
      {
         refiner->Apply(pmesh);
         if (refiner->Refined()) { mesh_refined = true; }
         MFEM_VERIFY(!refiner->Derefined(),"");
         MFEM_VERIFY(!refiner->Rebalanced(),"");
         break;
      }
      default: MFEM_ABORT("Unknown AMR estimator!");
   }

   // custom uses refs, ZZ and Kelly will set mesh_refined
   const int nref = pmesh.ReduceInt(refs.Size());

   if (nref && !mesh_refined)
   {
      constexpr int non_conforming = 1;
      pmesh.GetNodes()->HostReadWrite();
      dbg("pmesh.GetNodes():%d",pmesh.GetNodes()->Size());
      pmesh.GeneralRefinement(refs, non_conforming, opt.nc_limit);
      dbg("pmesh.GetNodes():%d",pmesh.GetNodes()->Size());
      mesh_refined = true;
      if (myid == 0)
      {
         std::cout << "Refined " << nref << " elements." << std::endl;
      }
   }
   else if (opt.estimator == amr::estimator::custom &&
            opt.deref_threshold >= 0.0 && !mesh_refined)
   {
      // no derefinement
      //MFEM_VERIFY(false, "Not yet implemented!");
   }
   else if ((opt.estimator == amr::estimator::zz ||
             opt.estimator == amr::estimator::kelly) && !mesh_refined)
   {
      MFEM_VERIFY(derefiner,"");
      if (derefiner->Apply(pmesh))
      {
         if (myid == 0)
         {
            std::cout << "\nDerefined elements." << std::endl;
         }
      }
      if (derefiner->Derefined()) { mesh_refined = true; }
   }
   else { /* nothing to do */ }

   if (mesh_refined)
   {
      dbg();

      AMRUpdate(S, offset, u);
      pmesh.GetNodes()->FESpace()->Update();
      pmesh.GetNodes()->Update();
      dbg("pmesh.GetNodes():%d",pmesh.GetNodes()->Size());

      //dbg("Rebalance");
      //pmesh.Rebalance();

      //mesh_pfes.Update();
      //x.ParFESpace()->Update();
      //x.Update();
      //pmesh.SetNodalGridFunction(&x);

      //xsub.ParFESpace()->Update();
      //xsub.Update();

      //AMRUpdate(S, offset, u);
      adv.AMRUpdate(S);

      ode_solver->Init(adv);
   }
}

void Operator::AMRUpdate(BlockVector &S,
                         Array<int> &offset,
                         ParGridFunction &u)
{
   dbg();
   pfes.Update();
   //pfes.PrintPartitionStats();

   const int vsize = pfes.GetVSize();
   MFEM_VERIFY(offset.Size() == 2, "!product_sync vs offset size error!");
   offset[0] = 0;
   offset[1] = vsize;

   BlockVector S_bkp(S);
   S.Update(offset, Device::GetMemoryType());
   const mfem::Operator* update_op = pfes.GetUpdateOperator();
   MFEM_VERIFY(update_op,"");
   update_op->Mult(S_bkp.GetBlock(0), S.GetBlock(0));

   u.MakeRef(&pfes, S, offset[0]);
   u.SyncMemory(S);
   MFEM_VERIFY(u.Size() == vsize,"");
}

} // namespace amr

} // namespace mfem
