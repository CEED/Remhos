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

static void Eval2D(const ParGridFunction &sol, const double threshold,
                   ParMesh &pmesh, Array<Refinement> &refs)
{
   Vector val;
   IntegrationPoint ip;
   constexpr int z = 0.0;
   constexpr double w = 1.0;
   const int NE = pmesh.GetNE();

   for (int e = 0; e < NE; e++)
   {
      for (int x = 0; x <= 1; x++)
      {
         for (int y = 0; y <= 1; y++)
         {
            ip.Set(x, y, z, w);
            sol.GridFunction::GetVectorValue(e, ip, val);
            const double l2_norm = val.Norml2();
            if (l2_norm > threshold) { refs.Append(Refinement(e)); }

         }
      }
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
   Vector derefs(NE), one(NE);
   derefs = 0.0;
   one = 1.0;

   switch (opt.estimator)
   {
      case amr::estimator::custom:
      {
         //dbg("opt.max_level: %d",opt.max_level);
         Vector u_max, u_min;
         GetPerElementMinMax(u, u_min, u_max);
         const double ref_threshold = opt.ref_threshold;
         const double deref_threshold = opt.deref_threshold;
         for (int e = 0; e < NE; e++)
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
         break;
      }

      case amr::estimator::jjt:
      {
         dbg("pfes order:%d", pfes.GetOrder(0));
         const int order = pfes.GetOrder(0);


         constexpr int DIM = 2;
         constexpr int SDIM = 3;
         constexpr int VDIM = 3;
         MFEM_VERIFY(pmesh.SpaceDimension() == 2,"");

         constexpr bool discont = false;
         //constexpr bool generate_edges = false;
         constexpr int ordering = Ordering::byVDIM;
         constexpr Element::Type QUAD = Element::QUADRILATERAL;
         //const double sx = 1.0, sy = 1.0;
         //const  bool sfc = true;
         Mesh quad(1, 1, QUAD);//, generate_edges, sx, sy, sfc);
         if (false)
         {
            quad.SetCurvature(order, discont, SDIM, ordering);
         }
         else
         {
            FiniteElementCollection *nfec =
               new H1_FECollection(order, DIM, BasisType::Positive);
            FiniteElementSpace* nfes =
               new FiniteElementSpace(&quad, nfec, SDIM, ordering);
            quad.SetNodalFESpace(nfes);
            quad.GetNodes()->MakeOwner(nfec);
         }
         MFEM_VERIFY(quad.GetNE() == 1,"");
         MFEM_VERIFY(quad.SpaceDimension() == SDIM,"");

         FiniteElementSpace *fes =
            const_cast<FiniteElementSpace*>(quad.GetNodalFESpace());
         MFEM_VERIFY(fes->GetVDim() == VDIM,"")

         dbg("u.Size():%d", u.Size());

         const int geom = pfes.GetFE(0)->GetGeomType();
         MFEM_VERIFY(geom == Geometry::SQUARE,"");
         const IntegrationRule &ir = IntRules.Get(geom, order + 2);
         const int nip = ir.GetNPoints();

         GridFunction &u_nodes = *pmesh.GetNodes();
         dbg("u_nodes.Size():%d", u_nodes.Size());

         GridFunction &q_nodes = *quad.GetNodes();
         dbg("q_nodes.Size():%d", q_nodes.Size());

         //MFEM_VERIFY(q_nodes.Size()/3 == 4, "");

         Vector vals;
         Vector coords;
         IntegrationPoint ip;
         constexpr int z = 0.0;
         constexpr double w = 1.0;
         DenseMatrix Jadjt, Jadj(DIM, SDIM);
         constexpr double NL_DMAX = std::numeric_limits<double>::max();

         for (int e = 0; e < pfes.GetNE(); e++)
         {
            const int depth = pmesh.pncmesh->GetElementDepth(e);

            u.GetValues(e, ir, vals);
            dbg("vals:"); vals.Print();

            MFEM_VERIFY(vals.Size() == (q_nodes.Size()/SDIM),"");

            for (int q=0, n=0, x=0; x <= 1; x++)
            {
               for (int y=0; y <= 1; y++)
               {
                  ip.Set(x, y, z, w);
                  u_nodes.GetVectorValue(e, ip, coords);
                  dbg("coords:"); coords.Print();
                  q_nodes(q+0) = coords[0];
                  q_nodes(q+1) = coords[1];
                  q_nodes(q+2) = vals(n++);
                  q+=3;
               }
            }

            for (int q=0; q < q_nodes.Size(); q+=3)
            {
               dbg("q: [%f, %f, %f]", q_nodes(q), q_nodes(q+1), q_nodes(q+2));
            }

            socketstream glvis("localhost", 19916);
            glvis.precision(8);
            glvis << "mesh\n" << quad << std::flush;

            MFEM_VERIFY(quad.GetNE() == 1, "");
            const int eq = 0;
            {
               double minW = +NL_DMAX;
               double maxW = -NL_DMAX;
               ElementTransformation *eTr = quad.GetElementTransformation(eq);
               //const Geometry::Type &type = quad.GetElement(e)->GetGeometryType();
               //const IntegrationRule *ir = &IntRules.Get(type, 2);
               const int NQ = ir.GetNPoints();
               //dbg("NQ:%d", NQ);
               for (int q = 0; q < NQ; q++)
               {
                  eTr->SetIntPoint(&ir.IntPoint(q));
                  const DenseMatrix &J = eTr->Jacobian();
                  CalcAdjugate(J, Jadj);
                  Jadjt = Jadj;
                  Jadjt.Transpose();
                  const double w = Jadjt.Weight();
                  //dbg("#%d w:%f",q,w);
                  minW = std::fmin(minW, w);
                  maxW = std::fmax(maxW, w);
               }
               assert(std::fabs(maxW) > 0.0);
               {
                  const double rho = minW / maxW;
                  //dbg("#%d rho:%f", e, rho);
                  MFEM_VERIFY(rho <= 1.0, "");
                  if (rho < opt.jac_threshold && depth < opt.max_level)
                  {
                     refs.Append(Refinement(e));
                  }
               }
            }
            assert(false);
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

   if (nref /*&& opt.ref_threshold >= 0.0*/ && !mesh_refined)
   {
      dbg("GeneralRefinement");
      constexpr int non_conforming = 1;
      pmesh.GetNodes()->HostReadWrite();
      pmesh.GeneralRefinement(refs, non_conforming, opt.nc_limit);
      mesh_refined = true;
      if (myid == 0)
      {
         std::cout << "Refined " << nref << " elements." << std::endl;
      }
   }
   /// deref only for custom for now
   else if (opt.estimator == amr::estimator::custom &&
            opt.deref_threshold >= 0.0 && !mesh_refined)
   {
      const int nderef = derefs * one;
      if (nderef > 0)
      {
         if (myid == 0)
         {
            std::cout << "DE-Refining " << nderef << " elements." << std::endl;
         }
         const int op = 1; // maximum value of fine elements
         mesh_refined = pmesh.DerefineByError(derefs, 2.0, opt.nc_limit, op);
         MFEM_VERIFY(mesh_refined,"");
      }
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
      pmesh.Rebalance();
      //xsub.ParFESpace()->Update();
      //xsub.Update();
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
