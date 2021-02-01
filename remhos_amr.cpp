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
      case amr::estimator::jjt: return "JJt";
      case amr::estimator::zz: return "ZZ";
      case amr::estimator::kelly: return "Kelly";
      default: MFEM_ABORT("Unknown estimator!");
   }
   return nullptr;
}

EstimatorIntegrator::EstimatorIntegrator(ParMesh &pmesh,
                                         const int max_level,
                                         const double jjt_threshold,
                                         const mode flux_mode):
   DiffusionIntegrator(one),
   NE(pmesh.GetNE()),
   e2(0),
   pmesh(pmesh),
   flux_mode(flux_mode),
   max_level(max_level),
   jjt_threshold(jjt_threshold) { dbg(); }

void EstimatorIntegrator::Reset() { e2 = 0; NE = pmesh.GetNE(); }

double EstimatorIntegrator::ComputeFluxEnergy(const FiniteElement &el,
                                              ElementTransformation &Tr,
                                              Vector &flux, Vector *d_energy)
{
   if (flux_mode == mode::diffusion)
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
         if (rho > jjt_threshold) { continue; }
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
         ComputeElementFlux2(e2++, el, Trans, fluxelem, flux);
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
                   ParGridFunction &u,
                   int order, int mesh_order,
                   int est,
                   double ref_t, double jjt_t, double deref_t,
                   int max_level, int nc_limit):

   pmesh(pmesh),
   u(u),
   pfes(pfes),
   mesh_pfes(mesh_pfes),
   myid(pmesh.GetMyRank()),
   dim(pmesh.Dimension()),
   sdim(pmesh.SpaceDimension()),
   flux_fec(order, dim),
   flux_fes(&pmesh, &flux_fec, sdim),
   opt( {order, mesh_order, est, ref_t, jjt_t, deref_t, max_level, nc_limit})
{
   dbg("AMR Setup");
   amr_vis.precision(8);

   if (myid == 0)
   {
      std::cout << "AMR setup with "
                << amr::EstimatorName(opt.estimator) << " estimator"
                << std::endl;
   }

   if (opt.estimator == amr::estimator::zz)
   {
      integ = new amr::EstimatorIntegrator(pmesh,
                                           opt.max_level,
                                           opt.jjt_threshold);
      smooth_flux_fec = new RT_FECollection(order-1, dim);
      auto smooth_flux_fes = new ParFiniteElementSpace(&pmesh, smooth_flux_fec);
      estimator = new L2ZienkiewiczZhuEstimator(*integ, u, &flux_fes,
                                                smooth_flux_fes);
   }

   if (opt.estimator == amr::estimator::kelly)
   {
      integ = new amr::EstimatorIntegrator(pmesh, opt.max_level,
                                           opt.jjt_threshold);
      estimator = new KellyErrorEstimator(*integ, u, flux_fes);
   }

   if (estimator)
   {
      const double hysteresis = 0.15;
      const double max_elem_error = 5.0e-3;
      refiner = new ThresholdRefiner(*estimator);
      refiner->SetTotalErrorFraction(0.0); // use purely local threshold
      refiner->SetLocalErrorGoal(max_elem_error);
      refiner->PreferConformingRefinement();
      refiner->SetNCLimit(opt.nc_limit);

      derefiner = new ThresholdDerefiner(*estimator);
      //derefiner->SetOp(2); // 0:min, 1:sum, 2:max
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

/// AMR Update for Custom Estimator
void Operator::AMRUpdateEstimatorCustom(Array<Refinement> &refs,
                                        Vector &derefs)
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
void Operator::AMRUpdateEstimatorJJt(Array<Refinement> &refs, Vector &derefs)
{
   dbg("JJt-LOR");
   const int horder = opt.order;
   // The refinement factor, an integer > 1
   const int ref_factor = 2;
   // Specify the positions of the new vertices.
   const int ref_type = BasisType::GaussLobatto;
   const int lorder = 1; // LOR space order
   dbg("order:%d, ref_factor:%d, lorder:%d", horder, ref_factor, lorder);

   // Create the low-order refined mesh
   ParMesh mesh_lor(&pmesh, ref_factor, ref_type);

   L2_FECollection fec_ho(horder, dim);
   L2_FECollection fec_lo(lorder, dim);
   L2_FECollection fec0(0, dim);

   ParFiniteElementSpace fes_ho(&pmesh, &fec_ho);
   ParFiniteElementSpace fes_lo(&mesh_lor, &fec_lo);
   ParFiniteElementSpace fes_rf(&pmesh, &fec0);

   ParGridFunction rho_ho(&fes_ho);
   ParGridFunction rho_lo(&fes_lo);
   ParGridFunction rho_rf(&fes_rf);

   // this is what we need: the 'solution' coming back is of size 'NE'
   assert(rho_rf.Size() == pmesh.GetNE());

   rho_ho.ProjectGridFunction(u);

   L2ProjectionGridTransfer forward(fes_ho, fes_lo);
   L2ProjectionGridTransfer backward(fes_rf, fes_lo);
   const mfem::Operator &R = forward.ForwardOperator();
   const mfem::Operator &P = backward.BackwardOperator();

   R.Mult(rho_ho, rho_lo);
   // VisualizeField(amr_vis, host, port, rho_lo, "rho_lo", Wx, Wy, Ww, Wh, keys);

   // now work on LOR mesh
   Array<int> dofs;
   Vector vals, elemvect, coords;
   const int geom = fes_lo.GetFE(0)->GetGeomType();
   const IntegrationRule &ir = IntRules.Get(geom, lorder+1);
   GridFunction &nodes_lor = *mesh_lor.GetNodes();
   const int nip = ir.GetNPoints();
   assert(nip == 4);

   constexpr int DIM = 2;
   constexpr int SDIM = 3;
   constexpr int VDIM = 3;
   constexpr bool discont = false;
   Mesh quad(1, 1, Element::QUADRILATERAL);
   constexpr int ordering = Ordering::byVDIM;
   quad.SetCurvature(lorder, discont, SDIM, ordering);
   FiniteElementSpace *qfes =
      const_cast<FiniteElementSpace*>(quad.GetNodalFESpace());
   assert(quad.GetNE() == 1);
   assert(qfes->GetVDim() == VDIM);
   assert(quad.SpaceDimension() == SDIM);
   GridFunction &q_nodes = *quad.GetNodes();
   DenseMatrix Jadjt, Jadj(DIM, SDIM);
   constexpr double NL_DMAX = std::numeric_limits<double>::max();

   for (int e = 0; e < mesh_lor.GetNE(); e++)
   {
      rho_lo.GetValues(e,ir,vals);
      fes_lo.GetElementVDofs(e, dofs);
      elemvect.SetSize(dofs.Size());
      assert(elemvect.Size() == 4);
      assert(vals.Size() == 4);

      for (int q = 0; q < nip; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         ElementTransformation *eTr = mesh_lor.GetElementTransformation(e);
         eTr->SetIntPoint(&ip);
         nodes_lor.GetVectorValue(e, ip, coords);
         assert(coords.Size() == 2);
         q_nodes(3*q+0) = coords[0];
         q_nodes(3*q+1) = coords[1];
         q_nodes(3*q+2) = rho_lo.GetValue(e, ip);
      }

      // Focus now on the quad mesh
      const int qe = 0;
      double minW = +NL_DMAX;
      double maxW = -NL_DMAX;
      ElementTransformation *eTr = quad.GetElementTransformation(qe);
      for (int q = 0; q < nip; q++) // nip == 4
      {
         eTr->SetIntPoint(&ir.IntPoint(q));
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
      assert(rho>0.0 && rho <= 1.0);
      elemvect = rho;
      rho_lo.SetSubVector(dofs, elemvect);
   }

   P.Mult(rho_lo, rho_rf);
   //VisualizeField(amr_vis,host,port,rho_rf,"AMR rho",Wx,Wy,Ww,Wh,keys);

   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      const int depth = pmesh.pncmesh->GetElementDepth(e);
      const double rho = rho_rf(e);
      dbg("#%d %.8e @%d",e,rho,depth);
      if ((rho < opt.jjt_threshold) && depth < opt.max_level )
      {
         dbg("\033[32mRefining #%d",e);
         refs.Append(Refinement(e));
      }
      if ((fabs(1.0-rho)<1e-6) && depth > 0 )
      {
         dbg("\033[31mDeRefinement #%d",e);
         derefs(e) = 1.0;
      }
   }
}

/// AMR Update for ZZ or Kelly Estimator
void Operator::AMRUpdateEstimatorZZKelly(bool &mesh_refined)
{
   refiner->Apply(pmesh);
   if (!refiner->Refined()) { return; }
   dbg("ZZ/Kelly refined!");
   mesh_refined = true;
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
                      ParGridFunction &v_sub_gf)
{
   dbg();
   Array<Refinement> refs;
   bool mesh_update = false;
   const int NE = pfes.GetNE();
   Vector derefs(NE), one(NE);
   derefs = 0.0;
   one = 1.0;

   switch (opt.estimator)
   {
      case amr::estimator::custom:
      {
         AMRUpdateEstimatorCustom(refs, derefs);
         break;
      }
      case amr::estimator::jjt:
      {
         AMRUpdateEstimatorJJt(refs, derefs);
         break;
      }
      case amr::estimator::zz:
      case amr::estimator::kelly:
      {
         AMRUpdateEstimatorZZKelly(mesh_update);
         break;
      }
      default: MFEM_ABORT("Unknown AMR estimator!");
   }

   // custom uses refs, ZZ and Kelly will set mesh_refined
   const int nref = pmesh.ReduceInt(refs.Size());

   if (nref > 0 && !mesh_update)
   {
      dbg("GeneralRefinement");
      constexpr int non_conforming = 1;
      pmesh.GetNodes()->HostReadWrite();
      pmesh.GeneralRefinement(refs, non_conforming, opt.nc_limit);
      mesh_update = true;
      if (myid == 0)
      {
         std::cout << "Refined " << nref << " elements." << std::endl;
      }
   }
   /// deref only for custom for now
   //else if (opt.estimator == amr::estimator::custom &&
   //         opt.deref_threshold >= 0.0 && !mesh_refined)
   /// JJt deref
   else if (opt.estimator == amr::estimator::jjt && !mesh_update)
   {
      const int nderef = derefs * one;
      if (nderef > 0)
      {
         if (myid == 0)
         {
            std::cout << "DE-Refining " << nderef << " elements." << std::endl;
         }
         const int op = 1; // maximum value of fine elements
         mesh_update = pmesh.DerefineByError(derefs, 1.0, opt.nc_limit, op);
         MFEM_VERIFY(mesh_update,"");
      }
   }
   else if ((opt.estimator == amr::estimator::zz ||
             opt.estimator == amr::estimator::kelly) && !mesh_update)
   {
      MFEM_VERIFY(derefiner,"");
      if (derefiner->Apply(pmesh))
      {
         mesh_update = true;
         if (myid == 0)
         {
            std::cout << "\nDerefined elements." << std::endl;
         }
      }
   }
   else { /* nothing to do */ }

   if (mesh_update)
   {
      dbg();
      AMRUpdate(S, offset, lom, subcell_mesh, pfes_sub, xsub, v_sub_gf);
      //pmesh.Rebalance();
      adv.AMRUpdate(S);
      ode_solver->Init(adv);
   }
}

void Operator::AMRUpdate(BlockVector &S,
                         Array<int> &offset,
                         LowOrderMethod &lom,
                         ParMesh *subcell_mesh,
                         ParFiniteElementSpace *pfes_sub,
                         ParGridFunction *xsub,
                         ParGridFunction &v_sub_gf)
{
   dbg();
   pfes.Update();

   const int vsize = pfes.GetVSize();
   MFEM_VERIFY(offset.Size() == 2, "!product_sync vs offset size error!");
   offset[0] = 0;
   offset[1] = vsize;

   BlockVector S_bkp(S);
   S.Update(offset, Device::GetMemoryType());
   const mfem::Operator *update_op = pfes.GetUpdateOperator();
   MFEM_VERIFY(update_op,"");
   update_op->Mult(S_bkp.GetBlock(0), S.GetBlock(0));

   u.MakeRef(&pfes, S, offset[0]);
   u.SyncAliasMemory(S);
   MFEM_VERIFY(u.Size() == vsize,"");

   if (xsub)
   {
      xsub->ParFESpace()->Update();
      xsub->Update();
   }
}

} // namespace amr

} // namespace mfem
