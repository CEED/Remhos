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

/// ESTIMATOR ///
static const char *EstimatorName(const int est)
{
   switch (static_cast<amr::Estimator>(est))
   {
      case amr::Estimator::ZZ: return "ZZ";
      case amr::Estimator::L2ZZ: return "L2ZZ";
      case amr::Estimator::JJt: return "JJt";
      case amr::Estimator::Custom: return "Custom";
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
                                             bool with_coef)
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
   flux_fes(new ParFiniteElementSpace(&pmesh, flux_fec, dim))
{
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
void Operator::Apply()
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
      case Estimator::L2ZZ:   { ApplyZZ(); break; }
      case Estimator::JJt:    { ApplyJJt(); break; }
      case Estimator::Custom: { ApplyCustom(); break; }
      default: MFEM_ABORT("Unknown AMR estimator!");
   }

   const bool JJt = opt.estimator == Estimator::JJt;
   if (!JJt) { dbg("mesh_refined/derefined set by ZZ"); return; }

   const int nref = pmesh.ReduceInt(refs.Size());
   mesh_refined = nref > 0;
   dbg("mesh_refined: %s", mesh_refined ? "yes" : "no");

   mesh_derefined = !mesh_refined &&
                    pmesh.GetLastOperation() == Mesh::REFINE &&
                    derefs.Max() > 0.0;
   dbg("JJt: %s, derefs.Max():%f", JJt ? "yes" : "no", derefs.Max());
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
                      Vector &lumpedM,
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
      dbg("JJt GeneralRefinement");
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
   /// deref only for custom for now
   //else if (opt.estimator == amr::estimator::custom &&
   //         opt.deref_threshold >= 0.0 && !mesh_refined)
   /// JJt deref
   else if (mesh_derefined && derefs.Max() > 0.0)
   {
      dbg("Got at least one (%f), NE:%d!", derefs.Max(), pmesh.GetNE());

      ///////////////////////////////////////////////////////////////////////
      /*{
         // Only serial ?
         assert(pmesh.GetNRanks() == 1);

         Table coarse_to_fine;
         Table ref_type_to_matrix;
         Array<int> coarse_to_ref_type;
         Array<Geometry::Type> ref_type_to_geom;

         const CoarseFineTransformations &rtrans = pmesh.GetRefinementTransforms();
         rtrans.GetCoarseToFineMap(pmesh,
                                   coarse_to_fine,
                                   coarse_to_ref_type,
                                   ref_type_to_matrix,
                                   ref_type_to_geom);
         Array<int> tabrow;
         const double threshold = 1.0;

         Vector local_err(pmesh.GetNE());
         local_err = 2.0 * threshold; // 8.0 in aggregated error

         //const int op = 1; // 0:min, 1:sum, 2:max
         int n_derefs = 0;

         assert(derefs.Size() == pmesh.GetNE());
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
               const double rho = derefs(fine_e);
               dbg("\t#%d -> %d: %.4e", coarse_e, fine_e, rho);
               all_four &= rho > opt.deref_threshold;
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
         if (myid == 0 && n_derefs > 0)
         {
            std::cout << "\033[31m DE-Refined " << n_derefs
                      << " elements.\033[m" << std::endl;
            if (opt.nc_limit) { MFEM_VERIFY(mesh_derefined,""); }
         }
      }*/
      ///////////////////////////////////////////////////////////////////////
      {
         const int NE = pmesh.GetNE();
         const double threshold = opt.deref_threshold;

         Vector local_err(NE);
         local_err = threshold;

         for (int e = 0; e < NE; e++)
         {
            const double rho = derefs(e);
            dbg("#%d derefs: %f", e, derefs(e));
            local_err(e) = 1.0 - rho;
            dbg("#%d local_err: %f", e, local_err(e));
         }
         //dbg("local_err: "); local_err.Print();
         dbg("threshold: %f",threshold);
         const int op = 2; // 0:min, 1:sum, 2:max
         mesh_derefined =
            pmesh.DerefineByError(local_err, threshold, 0, op);

         if (myid == 0 && mesh_derefined)
         {
            std::cout << "\033[31m DE-Refined.\033[m" << std::endl;
            if (opt.nc_limit) { MFEM_VERIFY(mesh_derefined,""); }
         }
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
       ((derefs.Max() > 0.0) && !mesh_derefined && !mesh_refined))
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
   dbg("done");

}

/// Internall AMR update
void Operator::UpdateAndRebalance(BlockVector &S,
                                  Array<int> &offset,
                                  LowOrderMethod &lom,
                                  ParMesh *subcell_mesh,
                                  ParFiniteElementSpace *pfes_sub,
                                  ParGridFunction *xsub,
                                  ParGridFunction &v_sub_gf)
{
   dbg("AMR Operator Update");
   dbg("pfes VSize:%d", pfes.GetVSize());

   Vector tmp;
   u.GetTrueDofs(tmp);
   u.SetFromTrueDofs(tmp);

   if (mesh_refined)
   {
      dbg("\033[32mREFINE");
      pfes.Update();

      const int vsize = pfes.GetVSize();
      MFEM_VERIFY(offset.Size() == 2, "!product_sync vs offset size error!");
      offset[0] = 0;
      offset[1] = vsize;

      BlockVector S_bkp(S);
      S.Update(offset, Device::GetMemoryType());

      const mfem::Operator *R = pfes.GetUpdateOperator();
      dbg("R: %dx%d", R->Width(), R->Height());

      R->Mult(S_bkp.GetBlock(0), S.GetBlock(0));

      u.MakeRef(&pfes, S, offset[0]);
      MFEM_VERIFY(u.Size() == vsize,"");
      u.SyncMemory(S);
   }
   else if (derefs.Max() > 0.0 || mesh_derefined)
   {
      if (!mesh_derefined && opt.estimator == Estimator::JJt)
      {
         dbg("Derefinement aborted!");
         return;
      }

      dbg("DEREFINE");

      ParGridFunction U = u;

      //Mass M_refine(pfes);

      //pfes.SetUpdateOperatorType(mfem::Operator::Type::MFEM_SPARSEMAT);
      dbg("pfes:%d", pfes.GetVSize());
      dbg("Update");
      pfes.Update();
      dbg("updated pfes:%d", pfes.GetVSize());

      //Mass M_coarse(pfes);

      const int vsize = pfes.GetVSize();
      MFEM_VERIFY(offset.Size() == 2, "!product_sync vs offset size error!");
      offset[0] = 0;
      offset[1] = vsize;

      dbg("S:%d", S.Size());

      S.Update(offset, Device::GetMemoryType());
      dbg("S.GetBlock(0) size:%d", S.GetBlock(0).Size());

      const mfem::Operator *R = pfes.GetUpdateOperator();
      assert(R);
      dbg("Rop: %dx%d", R->Width(), R->Height());
      assert(R->GetType() == mfem::Operator::ANY_TYPE);

      OperatorHandle Th;
      pfes.GetUpdateOperator(Th);
      SparseMatrix *Pop = Th.As<SparseMatrix>();
      assert(Pop);
      dbg("\033[31mPop: %dx%d, type:%d", Pop->Width(), Pop->Height(), Pop->GetType());
      assert(Pop->GetType() == mfem::Operator::MFEM_SPARSEMAT);

      dbg("u.MakeRef");
      u.MakeRef(&pfes, S, offset[0]);
      MFEM_VERIFY(u.Size() == vsize,"");
      u.SyncMemory(S);

#if 1
      dbg("R->Mult");
      R->Mult(U, u);
#else
      dbg("AMR_P");
      dbg("Pop: %dx%d", Pop->Width(), Pop->Height());
      AMR_P P(M_refine, M_coarse, *Pop);
      u = 0.0;
      dbg("U:%d, u:%d", U.Size(),  u.Size());
      P.Mult(U, u);

      dbg("AMR_P done");
#endif

   }
   else { assert(false); }

   u.GetTrueDofs(tmp);
   u.SetFromTrueDofs(tmp);

   if (xsub)
   {
      dbg("\033[31mXSUB!!");
      xsub->ParFESpace()->Update();
      xsub->Update();
   }
   dbg("done");
}

/// AMR_P
AMR_P::AMR_P(Mass &M_refine,
             Mass &M_coarse,
             const mfem::Operator &P)
   : M_refine(M_refine),
     M_coarse(M_coarse),
     P(P),
     rap(P, M_coarse, P),
     cg(MPI_COMM_WORLD)
{
   dbg("M_refine:%d, M_coarse: %d", M_refine.vsize, M_coarse.vsize);
   assert(M_refine.vsize > M_coarse.vsize);
   cg.SetRelTol(1e-14);
   cg.SetAbsTol(0.0);
   cg.SetMaxIter(1000);
   cg.SetPrintLevel(1);
   cg.SetOperator(rap);
   //cg.SetPreconditioner(*M_coarse.prec);
}

/// AMR_P Mult
void AMR_P::Mult(const Vector &x, Vector &y) const
{
   dbg("\033[33m[AMR_P::Mult]");
#if 0
   z1.SetSize(x.Size());
   z2.SetSize(x.Size());

   dbg("\033[33m[AMR_P::Mult] M_refine.Mult: %d => %d", x.Size(), z1.Size());
   M_refine.Mult(x, z1);

   dbg("\033[33m[AMR_P::Mult] cg.Mult: %d => %d", z1.Size(), z2.Size());
   cg.Mult(z1, z2);

   dbg("\033[33m[AMR_P::Mult] P->Mult: %d => %d", z1.Size(), z2.Size());
   P.Mult(z2, y);
#else
   z1.SetSize(x.Size());
   z2.SetSize(y.Size());

   dbg("\033[33m[AMR_P::Mult] M_refine.Mult: %d => %d", x.Size(), z1.Size());
   M_refine.Mult(x, z1);

   dbg("\033[33m[AMR_P::Mult] P->Mult: %d => %d", z1.Size(), z2.Size());
   P.Mult(z1, z2);

   dbg("\033[33m[AMR_P::Mult] M_coarse.Mult: %d => %d", z2.Size(), y.Size());
   M_coarse.Mult(z2, y);
#endif
   dbg("\033[33mAMR_P::Mult done");
}

/// Mass
Mass::Mass(ParFiniteElementSpace &fes, bool pa)
   : mfem::Operator(fes.GetVSize()),
     vsize(fes.GetVSize()),
     R_ptr(fes.GetRestrictionMatrix()),
     R((assert(R_ptr),R_ptr->ToDenseMatrix())),
     m(&fes),
     cg(MPI_COMM_WORLD)
{
   dbg("[MASS] VSize:%d",vsize);

   m.AddDomainIntegrator(new MassIntegrator);
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

/// Mass MULT
void Mass::Mult(const Vector &x, Vector &y) const
{
   dbg("[Mass] vsize:%d", vsize);
   assert(R);
   if (!R)
   {
      dbg("!R");
      M->Mult(x, y);
   }
   else
   {
      dbg("[Mass] R");
      z1.SetSize(R->Height());
      z2.SetSize(R->Height());

      dbg("[Mass] R:%dx%d", R->Height(), R->Width());
      dbg("[Mass] x:%d, z1:%d, y:%d", x.Size(), z1.Size(), y.Size());
      dbg("[Mass] R->Mult");
      R->Mult(x, z1);

      dbg("[Mass] M->Mult");
      M->Mult(z1, z2);
      dbg("[Mass] R->MultTranspose");
      R->MultTranspose(z2, y);
      dbg("[Mass] done");
   }
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
