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

#include "remhos.hpp"
#include "linalg/sparsemat.hpp"

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
      case amr::estimator::l2zz: return "L2ZZ";
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
                   double ref_t,
                   double deref_t,
                   double jjt_ref_t,
                   double jjt_deref_t,
                   int max_level,
                   int nc_limit):

   pmesh(pmesh),
   u(u),
   pfes(pfes),
   //mesh_pfes(mesh_pfes),
   myid(pmesh.GetMyRank()),
   dim(pmesh.Dimension()),
   sdim(pmesh.SpaceDimension()),
   fec(order, dim),
   flux_fec(order, dim),
   flux_fes(&pmesh,
            est == amr::estimator::kelly ?
            static_cast<FiniteElementCollection*>(&flux_fec) :
            static_cast<FiniteElementCollection*>(&fec),
            sdim),
   opt(
{
   order, mesh_order, est,
          ref_t, deref_t,
          jjt_ref_t, jjt_deref_t,
          max_level, nc_limit
})
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
      MFEM_VERIFY(false, "Not yet fully implemented!");
      integ = new amr::EstimatorIntegrator(pmesh,
                                           opt.max_level,
                                           opt.jjt_ref_threshold);
      estimator = new ZienkiewiczZhuEstimator(*integ, u, &flux_fes);
   }

   if (opt.estimator == amr::estimator::l2zz)
   {
      integ = new amr::EstimatorIntegrator(pmesh,
                                           opt.max_level,
                                           opt.jjt_ref_threshold);
      smooth_flux_fec = new RT_FECollection(order-1, dim);
      auto smooth_flux_fes = new ParFiniteElementSpace(&pmesh, smooth_flux_fec);
      estimator = new L2ZienkiewiczZhuEstimator(*integ, u,
                                                &flux_fes,smooth_flux_fes);
   }

   if (opt.estimator == amr::estimator::kelly)
   {
      integ = new amr::EstimatorIntegrator(pmesh,
                                           opt.max_level,
                                           opt.jjt_ref_threshold);
      estimator = new KellyErrorEstimator(*integ, u, flux_fes);
   }

   if (estimator)
   {
      const double hysteresis = 0.1;
      const double max_elem_error = 5.0e-3;
      refiner = new ThresholdRefiner(*estimator);
      refiner->SetTotalErrorFraction(0.0); // use purely local threshold
      refiner->SetLocalErrorGoal(max_elem_error);
      refiner->PreferConformingRefinement();
      refiner->SetNCLimit(opt.nc_limit);

      derefiner = new ThresholdDerefiner(*estimator);
      derefiner->SetOp(1); // 0:min, 1:sum, 2:max
      derefiner->SetThreshold(hysteresis * max_elem_error);
      derefiner->SetNCLimit(opt.nc_limit);

      refiner->Reset();
      derefiner->Reset();
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
void Operator::AMRUpdateEstimatorJJt(Array<Refinement> &refs,
                                     Vector &deref_tags)
{
   dbg("JJt");
   const bool vis = false;
   const int horder = opt.order;
   // The refinement factor, an integer > 1
   const int ref_factor = 2;
   // Specify the positions of the new vertices.
   const int ref_type = BasisType::ClosedUniform; // ClosedUniform || GaussLobatto
   const int lorder = 1; // LOR space order
   dbg("order:%d, ref_factor:%d, lorder:%d", horder, ref_factor, lorder);
   //dbg("pmesh.GetNE:%d", pmesh.GetNE());

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
   dbg("pmesh.GetNE:%d, u:%d, rho_ho:%d", pmesh.GetNE(), u.Size(), rho_ho.Size());

   const bool L2projection = true;

   const IntegrationRule &ir_ho = IntRules.Get(Geometry::SQUARE, horder + 1);
   const IntegrationRule &ir_lo = IntRules.Get(Geometry::SQUARE, lorder + 1);

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
      Array<int> ess_tdof_list;
      a.FormLinearSystem(ess_tdof_list, rho_ho, b, A, X, B);
      OperatorJacobiSmoother M(a, ess_tdof_list);
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
   constexpr bool discont = false;
   DenseMatrix Jadjt, Jadj(DIM, SDIM);
   Mesh quad(1, 1, Element::QUADRILATERAL);
   constexpr int ordering = Ordering::byVDIM;
   quad.SetCurvature(lorder, discont, SDIM, ordering);
   GridFunction &q_nodes = *quad.GetNodes();
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
      ElementTransformation *eTr = quad.GetElementTransformation(qe);
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
         //elemvect[q] = w;
      }
      assert(std::fabs(maxW) > 0.0);
      const double rho = minW / maxW;
      //dbg("#%d rho:%f", e, rho);
      assert(rho > 0.0 && rho <= 1.0);
      /*for (int q = 0; q < nip; q++) // nip == 4
      {
         const double w = elemvect[q];
         elemvect[q] = 1.0 - w / rho;
      }*/
      elemvect = rho;
      rho_lo.SetSubVector(dofs, elemvect);
      /*if (vis && e == 0)
      {
         static bool newly_opened = false;
         if (myid == 0)
         {
            if (!amr_vis[3].is_open() || !amr_vis[3])
            {
               amr_vis[3].open(host, port);
               amr_vis[3].precision(8);
            }
         }
         newly_opened = true;
         amr_vis[3] << "mesh\n";
         amr_vis[3] << quad;
         //amr_vis[3] << "keys gmA\n";
         amr_vis[3] << std::endl;

         const unsigned int microseconds = 1000000;
         usleep(microseconds);
      }*/
   }

   P.Mult(rho_lo, rho_rf);
   if (vis) VisualizeField(amr_vis[2], host, port, rho_rf, "rho_rf",
                              Wx + 2*Ww, Wy, Ww, Wh, keys);

   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      const int depth = pmesh.pncmesh->GetElementDepth(e);
      const double rho = rho_rf(e);
      dbg("#%d %.4e @ %d/%d",e, rho, depth, opt.max_level);

      if ((rho < opt.jjt_ref_threshold) && depth < opt.max_level )
      {
         dbg("\033[32mRefining #%d",e);
         refs.Append(Refinement(e));
      }
      if ((rho >= opt.jjt_deref_threshold) && depth > 0 )
      {
         dbg("\033[31mTag for de-refinement #%d",e);
         deref_tags(e) = rho;
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
                      ParGridFunction &v_sub_gf,
                      Vector &lumpedM,
                      const double mass0_u,
                      ParGridFunction &inflow_gf,
                      FunctionCoefficient &inflow)
{
   dbg();
   Array<Refinement> refs;
   bool mesh_refine = false;
   bool mesh_derefine = false;
   const int NE = pfes.GetNE();
   Vector deref_tags(NE);
   deref_tags = 0.0; // tie to 0.0 to trig the bool tests below

   switch (opt.estimator)
   {
      case amr::estimator::custom:
      {
         AMRUpdateEstimatorCustom(refs, deref_tags);
         break;
      }
      case amr::estimator::jjt:
      {
         AMRUpdateEstimatorJJt(refs, deref_tags);
         break;
      }
      case amr::estimator::zz:
      case amr::estimator::l2zz:
      case amr::estimator::kelly:
      {
         AMRUpdateEstimatorZZKelly(mesh_refine);
         mesh_derefine = mesh_refine;
         break;
      }
      default: MFEM_ABORT("Unknown AMR estimator!");
   }

   // custom uses refs, ZZ and Kelly will set mesh_refined
   const int nref = pmesh.ReduceInt(refs.Size());

   if (nref > 0 && !mesh_refine)
   {
      dbg("GeneralRefinement");
      constexpr int non_conforming = 1;
      pmesh.GetNodes()->HostReadWrite();
      pmesh.GeneralRefinement(refs, non_conforming, opt.nc_limit);
      mesh_refine = true;
      if (myid == 0)
      {
         std::cout << "\033[32mRefined " << nref
                   << " elements.\033[m" << std::endl;
      }
   }
   /// deref only for custom for now
   //else if (opt.estimator == amr::estimator::custom &&
   //         opt.deref_threshold >= 0.0 && !mesh_refined)
   /// JJt deref
   else if (opt.estimator == amr::estimator::jjt
            && !mesh_refine
            && pmesh.GetLastOperation() == Mesh::REFINE)
   {
      const bool deref_tagged = deref_tags.Max() > 0.0;

      if (deref_tagged)
      {
         dbg("Got at least one (%f), NE:%d!", deref_tags.Max(), pmesh.GetNE());

         Table coarse_to_fine;
         Table ref_type_to_matrix;
         Array<int> coarse_to_ref_type;
         Array<Geometry::Type> ref_type_to_geom;
         const CoarseFineTransformations &rtrans =
            pmesh.GetRefinementTransforms();
         rtrans.GetCoarseToFineMap(pmesh,
                                   coarse_to_fine,
                                   coarse_to_ref_type,
                                   ref_type_to_matrix,
                                   ref_type_to_geom);
         Array<int> tabrow;
         const double threshold = 1.0;

         Vector local_err(pmesh.GetNE());
         local_err = 2.0 * threshold; // 8.0 in aggregated error

         const int op = 1; // 0:min, 1:sum, 2:max
         int n_derefs = 0;

         assert(deref_tags.Size() == pmesh.GetNE());
         assert(local_err.Size() == pmesh.GetNE());

         dbg("coarse_to_fine:%d",coarse_to_fine.Size());

         for (int coarse_e = 0; coarse_e < coarse_to_fine.Size(); coarse_e++)
         {
            coarse_to_fine.GetRow(coarse_e, tabrow);
            const int tabsz = tabrow.Size();
            dbg("Scanning element #%d (%d)", coarse_e, tabsz);
            //dbg("tabrow:"); tabrow.Print();
            if (tabsz != 4) { continue; }
            bool all_four = true;
            for (int j = 0; j < tabrow.Size(); j++)
            {
               const int fine_e = tabrow[j];
               const double rho = deref_tags(fine_e);
               dbg("\t#%d -> %d: %.4e", coarse_e, fine_e, rho);
               all_four &= rho > opt.jjt_deref_threshold;
            }

            if (!all_four) { continue; }
            dbg("\033[31mDERFINE #%d",coarse_e);
            for (int j = 0; j < tabrow.Size(); j++)
            {
               const int fine_e = tabrow[j];
               local_err(fine_e) = 0.0;
            }
            n_derefs += 1;
         }
         mesh_derefine =
            pmesh.DerefineByError(local_err, threshold, opt.nc_limit, op);

         if (myid == 0 && n_derefs > 0)
         {
            std::cout << "\033[31m DE-Refined " << n_derefs
                      << " elements.\033[m" << std::endl;
            if (opt.nc_limit) { MFEM_VERIFY(mesh_derefine,""); }
         }
      }
   }
   else if ((opt.estimator == amr::estimator::zz ||
             opt.estimator == amr::estimator::l2zz ||
             opt.estimator == amr::estimator::kelly) && !mesh_refine)
   {
      MFEM_VERIFY(derefiner,"");
      if (derefiner->Apply(pmesh))
      {
         mesh_refine = true;
         if (myid == 0)
         {
            std::cout << "\n\033[31mDerefined elements.\033[m" << std::endl;
         }
      }
   }
   else { /* nothing to do */ }

   if (mesh_refine || mesh_derefine)
   {
      dbg("mesh_refine:%s, mesh_derefine:%s",
          mesh_refine?"yes":"no",
          mesh_derefine?"yes":"no");
      AMRUpdate(mesh_derefine,
                S, offset, lom, subcell_mesh, pfes_sub, xsub, v_sub_gf);
      inflow_gf.Update();
      inflow_gf.ProjectCoefficient(inflow);
      //pmesh.Rebalance();

      adv.AMRUpdate(S, u, mass0_u);

      ode_solver->Init(adv);
   }
}

SparseMatrix *IdentityMatrix(int n)
{
   SparseMatrix *I = new SparseMatrix(n);
   for (int i=0; i<n; ++i)
   {
      I->Set(i, i, 1.0);
   }
   I->Finalize();
   return I;
}

void Operator::AMRUpdate(const bool derefine,
                         BlockVector &S,
                         Array<int> &offset,
                         LowOrderMethod &lom,
                         ParMesh *subcell_mesh,
                         ParFiniteElementSpace *pfes_sub,
                         ParGridFunction *xsub,
                         ParGridFunction &v_sub_gf)
{
   dbg("AMR Operator Update");
   dbg("refined pfes:%d", pfes.GetVSize());


   if (!derefine)
   {
      Vector tmp;
      u.GetTrueDofs(tmp);
      u.SetFromTrueDofs(tmp);

      dbg("pfes.Update");
      pfes.Update();
      dbg("updated pfes:%d", pfes.GetVSize());

      const int vsize = pfes.GetVSize();
      MFEM_VERIFY(offset.Size() == 2, "!product_sync vs offset size error!");
      offset[0] = 0;
      offset[1] = vsize;

      dbg("S_bkp");
      BlockVector S_bkp(S);
      S.Update(offset, Device::GetMemoryType());

      dbg("GetUpdateOperator");
      const mfem::Operator *R = pfes.GetUpdateOperator();
      dbg("R: %dx%d", R->Width(), R->Height());

      dbg("R->Mult");
      R->Mult(S_bkp.GetBlock(0), S.GetBlock(0));

      u.MakeRef(&pfes, S, offset[0]);
      MFEM_VERIFY(u.Size() == vsize,"");
      u.SyncMemory(S);
   }
   else
   {
      dbg("DEREFINE");
      Vector tmp;
      u.GetTrueDofs(tmp);
      u.SetFromTrueDofs(tmp);
      ParGridFunction U = u;

      //pfes.SetUpdateOperatorType(mfem::Operator::Type::Hypre_ParCSR);

      //Mass M_orig(pfes, false);
      GridFunction R_one(&pfes);
      GridFunction R_gf(&pfes);
      GridFunction M_R_gf(&pfes);

      Mass M_refine(pfes);

      dbg("pfes:%d", pfes.GetVSize());
      dbg("Update");
      pfes.Update();
      dbg("updated pfes:%d", pfes.GetVSize());

      Mass M_coarse(pfes);

      const int vsize = pfes.GetVSize();
      MFEM_VERIFY(offset.Size() == 2, "!product_sync vs offset size error!");
      offset[0] = 0;
      offset[1] = vsize;

      dbg("S:%d",S.Size());
      BlockVector S_bkp(S);
      dbg("S_bkp:%d",S_bkp.Size());

      S.Update(offset, Device::GetMemoryType());


      const mfem::Operator *R = pfes.GetUpdateOperator();
      assert(R);
      dbg(" R: %dx%d", R->Width(), R->Height());
      //R->PrintMatlab(std::cout);

      SparseMatrix Rt(R->Width(), R->Height());
      dbg("Rt: %dx%d", Rt.Width(), Rt.Height());
      {
         const int n = R->Width();
         const int m = R->Height();
         Vector x(n), y(m);
         x = 0.0;

         //std::out << setiosflags(ios::scientific | ios::showpos);
         for (int i = 0; i < n; i++)
         {
            x(i) = 1.0;
            R->Mult(x, y);
            for (int j = 0; j < m; j++)
            {
               if (y(j))
               {
                  //dbg("[%d,%d] %f",j,i,y(j));
                  Rt.Set(i,j,y(j));
               }
            }
            x(i) = 0.0;
         }
      }
      dbg("Finalize");
      Rt.Finalize();
      dbg("PrintMatlab");
      //Rt.PrintMatlab(std::cout);
      //assert(false);

      //OperatorHandle Th;
      //pfes.GetUpdateOperator(Th);
      //HypreParMatrix *Rth = Th.As<HypreParMatrix>();
      //assert(Rth);
      //dbg("Rth: %dx%d", Rth->Width(), Rth->Height());
      //HypreParMatrix *Rtt = Rth->Transpose();
      //assert(Rtt);
      //Rtt->Print("Rtt");
      //fflush(0);
      //assert(false);
      //dbg("Rtt: %dx%d", Rtt->Width(), Rtt->Height());
      //assert(false);

      dbg("U size:%d", U.Size());
      dbg("S.GetBlock(0) size:%d", S.GetBlock(0).Size());
      dbg("S_bkp.GetBlock(0) size:%d", S_bkp.GetBlock(0).Size());

      dbg("R->Mult");
      R->Mult(U, S.GetBlock(0));

      dbg("u.MakeRef");
      u.MakeRef(&pfes, S, offset[0]);
      MFEM_VERIFY(u.Size() == vsize,"");
      u.SyncMemory(S);
      //u.GetTrueDofs(tmp);
      //u.SetFromTrueDofs(tmp);

      dbg("AMR_P");
      dbg(" R: %dx%d", R->Width(), R->Height());
      dbg("Rt: %dx%d", Rt.Width(), Rt.Height());
      AMR_P P(M_refine,
              M_coarse,
              *R, Rt);
      dbg("u = 0.0;");
      u = 0.0;
      dbg("refined_gf:%d, coarse_gf:%d", U.Size(),  u.Size());
      P.Mult(U, u);
   }

   if (xsub)
   {
      dbg("\033[31mXSUB!!");
      xsub->ParFESpace()->Update();
      xsub->Update();
   }
}

Mass::Mass(ParFiniteElementSpace &fes_, bool pa)
   : mfem::Operator(fes_.GetTrueVSize()),
     fes(fes_),
     m(&fes),
     cg(MPI_COMM_WORLD)
{
   dbg("fes_.GetVSize():%d",fes_.GetVSize());
   m.AddDomainIntegrator(new MassIntegrator);
   //if (pa) { m.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   m.Assemble();
   if (!pa) { m.Finalize(); }
   m.FormSystemMatrix(empty, M);
   /*
      if (pa) { prec.reset(new OperatorJacobiSmoother(m, empty)); }
      else
      {
         if (M.Type() == mfem::Operator::Type::Hypre_ParCSR)
         {
            prec.reset(new HypreSmoother(*M.As<HypreParMatrix>()));
         }
         else if (M.Type() == mfem::Operator::Type::MFEM_SPARSEMAT)
         {
            prec.reset(new DSmoother(*M.As<SparseMatrix>()));
         }
      }*/
}

void Mass::Mult(const Vector &x, Vector &y) const
{
   dbg("[Mass]");
   //const SparseMatrix *R = fes.GetRestrictionMatrix();
   //const Operator *P = fes.GetProlongationMatrix();
   //if (!R)
   {
      dbg("!R");
      M->Mult(x, y);
   }
   /*else
   {
      assert(P);
      dbg("[Mass] R");
      z1.SetSize(R->Height());
      z2.SetSize(R->Height());

      dbg("[Mass] R->Mult");
      P->MultTranspose(x, z1);
      dbg("[Mass] M->Mult");
      M->Mult(z1, z2);
      dbg("[Mass] R->MultTranspose");
      R->MultTranspose(z2, y);
      dbg("[Mass] done");
   }*/
}


AMR_P::AMR_P(Mass &M_refine,
             Mass &M_coarse,
             const mfem::Operator &R,
             const mfem::Operator &Rt)
   : M_refine(M_refine),
     M_coarse(M_coarse),
     R(R),
     Rt(Rt),
     rap(Rt, M_refine, Rt),
     cg(MPI_COMM_WORLD)
{
   cg.SetRelTol(1e-14);
   cg.SetAbsTol(0.0);
   cg.SetMaxIter(1000);
   cg.SetPrintLevel(1);
   cg.SetOperator(rap);
   //cg.SetPreconditioner(*M_coarse.prec);
}

void AMR_P::Mult(const Vector &x, Vector &y) const
{
   dbg("[AMR_P::Mult]");
   z1.SetSize(x.Size());
   z2.SetSize(Rt.Width());

   dbg("[AMR_P::Mult] M_refine.Mult");
   M_refine.Mult(x, z1);

   dbg("[AMR_P::Mult] R->MultTranspose");
   Rt.MultTranspose(z1, z2);

   dbg("[AMR_P::Mult] cg.Mult");
   cg.Mult(z2, y);

   dbg("AMR_P::Mult done");
}

} // namespace amr

} // namespace mfem
