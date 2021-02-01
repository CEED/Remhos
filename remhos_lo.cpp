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

#include "remhos_lo.hpp"
#include "remhos_tools.hpp"

using namespace std;

namespace mfem
{

DiscreteUpwind::DiscreteUpwind(ParFiniteElementSpace &space,
                               const SparseMatrix &adv,
                               const Array<int> &adv_smap, const Vector &Mlump,
                               Assembly &asmbly, bool updateD)
   : LOSolver(space),
     K(adv), D(), K_smap(adv_smap), M_lumped(Mlump),
     assembly(asmbly), update_D(updateD)
{
   D = K;
   ComputeDiscreteUpwindMatrix();
}

void DiscreteUpwind::CalcLOSolution(const Vector &u, Vector &du) const
{
   const int ndof = pfes.GetFE(0)->GetDof();
   Vector alpha(ndof); alpha = 0.0;

   // Recompute D due to mesh changes (K changes) in remap mode.
   if (update_D) { ComputeDiscreteUpwindMatrix(); }

   // Discretization and monotonicity terms.
   D.Mult(u, du);

   // Lump fluxes (for PDU).
   ParGridFunction u_gf(&pfes);
   u_gf = u;
   u_gf.ExchangeFaceNbrData();
   Vector &u_nd = u_gf.FaceNbrData();
   const int ne = pfes.GetNE();
   u.HostRead();
   du.HostReadWrite();
   M_lumped.HostRead();
   for (int k = 0; k < ne; k++)
   {
      // Face contributions.
      for (int f = 0; f < assembly.dofs.numBdrs; f++)
      {
         assembly.LinearFluxLumping(k, ndof, f, u, du, u_nd, alpha);
      }
   }

   const int s = du.Size();
   for (int i = 0; i < s; i++) { du(i) /= M_lumped(i); }
}

void DiscreteUpwind::ComputeDiscreteUpwindMatrix() const
{
   const int *Ip = K.HostReadI(), *Jp = K.HostReadJ(), n = K.Size();

   const double *Kp = K.HostReadData();

   double *Dp = D.HostReadWriteData();
   D.HostReadWriteI(); D.HostReadWriteJ();

   for (int i = 0, k = 0; i < n; i++)
   {
      double rowsum = 0.;
      for (int end = Ip[i+1]; k < end; k++)
      {
         int j = Jp[k];
         double kij = Kp[k];
         double kji = Kp[K_smap[k]];
         double dij = fmax(fmax(0.0,-kij),-kji);
         Dp[k] = kij + dij;
         Dp[K_smap[k]] = kji + dij;
         if (i != j) { rowsum += dij; }
      }
      D(i,i) = K(i,i) - rowsum;
   }
}

ResidualDistribution::ResidualDistribution(ParFiniteElementSpace &space,
                                           ParBilinearForm &Kbf,
                                           Assembly &asmbly, const Vector &Mlump,
                                           bool subcell, bool timedep)
   : LOSolver(space),
     K(Kbf), assembly(asmbly),
     M_lumped(Mlump), subcell_scheme(subcell), time_dep(timedep)
{ }

void ResidualDistribution::CalcLOSolution(const Vector &u, Vector &du) const
{
   const int ndof = pfes.GetFE(0)->GetDof();
   const int ne = pfes.GetMesh()->GetNE();
   Vector alpha(ndof); alpha = 0.0;
   Vector z(u.Size());

   const double gamma = 1.0;
   //int dof_id;
   double /*xSum,*/ sumFluctSubcellP, sumFluctSubcellN,
          /*sumWeightsP,
          sumWeightsN, weightP, weightN, rhoP, rhoN,*/ aux, fluct, eps = 1.E-15;
   Vector xMaxSubcell, xMinSubcell, sumWeightsSubcellP, sumWeightsSubcellN,
          fluctSubcellP, fluctSubcellN, nodalWeightsP, nodalWeightsN;

   double infinity = numeric_limits<double>::infinity();

   // Discretization terms
   du = 0.;
   K.Mult(u, z);

   //z = Conv * u

   ParGridFunction u_gf(&pfes);
   u_gf = u;
   u_gf.ExchangeFaceNbrData();
   Vector &u_nd = u_gf.FaceNbrData();

   z.HostReadWrite();
   u.HostRead();
   du.HostReadWrite();
   M_lumped.HostRead();

   // Boundary contributions - stored in du
   //will want this in a seperate kernel to do forall elements
   for (int k=0; k < ne; ++k)
      //int k=15;
   {
      for (int f = 0; f < assembly.dofs.numBdrs; f++)
         //int f = 0;
      {
         assembly.LinearFluxLumping(k, ndof, f, u, du, u_nd, alpha);
      }
   }

   Vector mydu(du.Size()); mydu = 0.0;
   assembly.DeviceLinearFluxLumping(u, mydu, FaceType::Interior);
   assembly.DeviceLinearFluxLumping(u, mydu, FaceType::Boundary);

   Vector diff(mydu);
   diff -= du;
   double error = diff.Norml2();
   if (error > 1e-12)
   {
      printf("error %g \n",error);
      printf("----------\n");
      du.Print(mfem::out,16);
      printf("\n------ \n");
      mydu.Print(mfem::out,16);
      printf("\n------ \n");
      diff.Print(mfem::out,16);
      exit(-1);
   }

   //Linear Flux Lumping forall elements/faces //alpha is 0 for us here
   //assembly.LinearFluxLumping_all(ndof, u, du, u_nd, alpha);

#if 1

   //initialize to infinity
   assembly.dofs.xe_min =  infinity;
   assembly.dofs.xe_max = -infinity;

   double *xe_min = assembly.dofs.xe_min.ReadWrite();
   double *xe_max = assembly.dofs.xe_max.ReadWrite();

   const double *d_u = u.Read();
   const double *d_z = z.Read();
   const double *d_M_lumped = M_lumped.Read();

   double *d_du = du.ReadWrite();

   MFEM_FORALL(k, ne,
   {

      // Boundary contributions - stored in du
      // done before this loop

      // Element contributions
      double rhoP(0.), rhoN(0.), xSum(0.);
      for (int j = 0; j < ndof; ++j)
      {
         int dof_id = k*ndof+j;
         xe_max[k] = max(xe_max[k], d_u[dof_id]);
         xe_min[k] = min(xe_min[k], d_u[dof_id]);
         xSum += d_u[dof_id];
         rhoP += max(0., d_z[dof_id]);
         rhoN += min(0., d_z[dof_id]);
      }

      //denominator of equation 47
      double sumWeightsP = ndof*xe_max[k] - xSum + eps;
      double sumWeightsN = ndof*xe_min[k] - xSum - eps;

      for (int i = 0; i < ndof; i++)
      {
         int dof_id = k*ndof+i;
         //eq 46
         double weightP = (xe_max[k] - d_u[dof_id]) / sumWeightsP;
         double weightN = (xe_min[k] - d_u[dof_id]) / sumWeightsN;

         // (lumpped trace term  + LED convection )/lumpped mass matrix
         d_du[dof_id] = (d_du[dof_id] + weightP * rhoP + weightN * rhoN) /
         d_M_lumped[dof_id];
      }

   });

#else //Reference version
   // Monotonicity terms
   for (int k = 0; k < ne; k++)
   {
      // Boundary contributions - stored in du
      // done before this loop

      // Element contributions
      rhoP = rhoN = xSum = 0.;
      assembly.dofs.xe_min(k) =   numeric_limits<double>::infinity();
      assembly.dofs.xe_max(k) = - numeric_limits<double>::infinity();
      for (int j = 0; j < ndof; j++)
      {
         dof_id = k*ndof+j;
         assembly.dofs.xe_max(k) = max(assembly.dofs.xe_max(k), u(dof_id));
         assembly.dofs.xe_min(k) = min(assembly.dofs.xe_min(k), u(dof_id));
         xSum += u(dof_id);
         rhoP += max(0., z(dof_id));
         rhoN += min(0., z(dof_id));
      }

      //denominator of equation 47
      sumWeightsP = ndof*assembly.dofs.xe_max(k) - xSum + eps;
      sumWeightsN = ndof*assembly.dofs.xe_min(k) - xSum - eps;

      for (int i = 0; i < ndof; i++)
      {
         dof_id = k*ndof+i;
         //eq 46
         weightP = (assembly.dofs.xe_max(k) - u(dof_id)) / sumWeightsP;
         weightN = (assembly.dofs.xe_min(k) - u(dof_id)) / sumWeightsN;

         // (lumpped trace term  + LED convection )/lumpped mass matrix
         du(dof_id) = (du(dof_id) + weightP * rhoP + weightN * rhoN) /
                      M_lumped(dof_id);
      }
   }
#endif

}

MFResidualDistribution::MFResidualDistribution(ParFiniteElementSpace &space,
                                               ParBilinearForm &Kbf,
                                               Assembly &asmbly, const Vector &Mlump,
                                               bool subcell, bool timedep)
   : LOSolver(space),
     K(Kbf), assembly(asmbly),
     M_lumped(Mlump), subcell_scheme(subcell), time_dep(timedep)
{ }

void MFResidualDistribution::SampleVelocity(FaceType type) const
{
   const FiniteElementSpace *fes = assembly.GetFes();
   int nf = fes->GetNFbyType(type);
   if (nf == 0) { return; }

   const IntegrationRule *ir = assembly.lom.irF;

   const Mesh *mesh = fes->GetMesh();
   const int dim = mesh->Dimension();
   const int nq = ir->GetNPoints();

   double *vel_ptr = nullptr;
   if (type == FaceType::Interior)
   {
      IntVelocity.SetSize(dim * nq * nf);
      vel_ptr = IntVelocity.Write();
   }

   if (type == FaceType::Boundary)
   {
      BdryVelocity.SetSize(dim * nq * nf);
      vel_ptr = BdryVelocity.Write();
   }

   const FiniteElement &el_trace =
      *fes->GetTraceElement(0, fes->GetMesh()->GetFaceBaseGeometry(0));

   const DofToQuad * maps = &el_trace.GetDofToQuad(*ir, DofToQuad::TENSOR);


   quad1D = maps->nqpt;
   dofs1D = maps->ndof;
   if (dim == 2) { face_dofs = quad1D; }
   if (dim == 3) { face_dofs = quad1D*quad1D; }

   auto C = mfem::Reshape(vel_ptr, dim, nq, nf);
   Vector Vq(dim);

   int f_idx = 0;
   for (int f = 0; f < fes->GetNF(); ++f)
   {
      int e1, e2;
      int inf1, inf2;
      fes->GetMesh()->GetFaceElements(f, &e1, &e2);
      fes->GetMesh()->GetFaceInfos(f, &inf1, &inf2);
      int my_face_id = inf1 / 64;

      if ((type==FaceType::Interior && (e2>=0 || (e2<0 && inf2>=0))) ||
          (type==FaceType::Boundary && e2<0 && inf2<0) )
      {

         FaceElementTransformations &T =
            *fes->GetMesh()->GetFaceElementTransformations(f);
         for (int q = 0; q < nq; ++q)
         {
            // Convert to lexicographic ordering
            int iq = ToLexOrdering(dim, my_face_id, quad1D, q);
            T.SetAllIntPoints(&ir->IntPoint(q));
            const IntegrationPoint &eip1 = T.GetElement1IntPoint();
            assembly.lom.coef->Eval(Vq, *T.Elem1, eip1);
            for (int i = 0; i < dim; ++i)
            {
               C(i,iq,f_idx) = Vq(i);
            }
         }
         f_idx++;
      }
   }

}

void MFResidualDistribution::SetupPA(FaceType type) const
{

   const FiniteElementSpace *fes = assembly.GetFes();
   int nf = fes->GetNFbyType(type);
   if (nf == 0) {return;}

   Mesh *mesh = fes->GetMesh();
   int dim = mesh->Dimension();

   //Delete old geo factors
   mesh->DeleteGeometricFactors();

   if (dim == 2) { return SetupPA2D(type); }
   if (dim == 3) { return SetupPA3D(type); }

}

void MFResidualDistribution::SetupPA2D(FaceType type) const
{
   const FiniteElementSpace *fes = assembly.GetFes();
   int nf = fes->GetNFbyType(type);
   Mesh *mesh = fes->GetMesh();
   int dim = mesh->Dimension();

   const IntegrationRule *ir = assembly.lom.irF;

   const FaceGeometricFactors *geom =
      mesh->GetFaceGeometricFactors(*ir,
                                    FaceGeometricFactors::DETERMINANTS |
                                    FaceGeometricFactors::NORMALS, type);

   const FiniteElement &el_trace =
      *fes->GetTraceElement(0, fes->GetMesh()->GetFaceBaseGeometry(0));

   const DofToQuad *maps = &el_trace.GetDofToQuad(*ir, DofToQuad::TENSOR);

   auto n = mfem::Reshape(geom->normal.HostRead(), quad1D, dim, nf);
   auto detJ = mfem::Reshape(geom->detJ.HostRead(), quad1D, nf);
   const double *w = ir->GetWeights().Read();

   const double *vel_ptr;
   if (type == FaceType::Interior) { vel_ptr = IntVelocity.Read(); }
   if (type == FaceType::Boundary) { vel_ptr = BdryVelocity.Read(); }

   auto vel = mfem::Reshape(vel_ptr, dim, ir->GetNPoints(), nf);

   int execMode = (int) assembly.GetExecMode();

   if (type == FaceType::Interior)
   {

      //two sides per face  - proof of concept for 2D.
      D_int.SetSize(quad1D*2*nf);
      auto D = mfem::Reshape(D_int.Write(), quad1D, 2, nf);

      MFEM_FORALL(f, nf,
      {
         for (int f_side=0; f_side<2; ++f_side)
         {

            for (int k1 = 0; k1 < quad1D; ++k1)
            {

               int direction = 1 - 2*f_side;

               double vvalnor =
               vel(0,k1,f)*direction*n(k1, 0, f) +
               vel(1,k1,f)*direction*n(k1, 1, f);

               if (execMode == 0)
               {
                  vvalnor = std::min(0., vvalnor); //advection
               }
               else
               {
                  vvalnor = -std::max(0., vvalnor);
               }

               double t_vn = vvalnor* w[k1] * detJ(k1, f);
               D(k1, f_side, f) = - t_vn;
               //val -= B(k1, i1) * B(k1,j1) * t_vn;
            }

         }//f_side

      });
   }///if interior

   if (type == FaceType::Boundary)
   {

      D_bdry.SetSize(quad1D*nf);

      auto D = mfem::Reshape(D_bdry.Write(), quad1D, nf);

      MFEM_FORALL(f, nf,
      {
         for (int k1 = 0; k1 < quad1D; ++k1)
         {

            double vvalnor =
            vel(0,k1,f)*n(k1, 0, f) +
            vel(1,k1,f)*n(k1, 1, f);

            if (execMode == 0)
            {
               vvalnor = std::min(0., vvalnor);
            }
            else
            {
               vvalnor = -std::max(0., vvalnor);
            }

            double t_vn = vvalnor* w[k1] * detJ(k1, f);
            D(k1, f) = - t_vn;
            //val -= B(k1, i1) * B(k1,j1) * t_vn;
         }

      });
   }//boundary

}

void MFResidualDistribution::SetupPA3D(FaceType type) const
{

   printf("need to implement 3D ... \n"); exit(-1);
}

void MFResidualDistribution::ApplyFaceTerms(const Vector &x, Vector &y,
                                            FaceType type) const
{

   const FiniteElementSpace *fes = assembly.GetFes();
   int nf = fes->GetNFbyType(type);
   if (nf == 0) {return;}

   Mesh *mesh = fes->GetMesh();
   int dim = mesh->Dimension();

   if (dim == 2) { return ApplyFaceTerms2D(x, y, type); }
   if (dim == 3) { return ApplyFaceTerms3D(x, y, type); }
}

void MFResidualDistribution::ApplyFaceTerms2D(const Vector &x, Vector &y,
                                              FaceType type) const
{
   const FiniteElementSpace *fes = assembly.GetFes();

   const int Q1D = quad1D;
   const int D1D = dofs1D;
   const int nf = fes->GetNFbyType(type);
   const Operator * face_restrict_lex = nullptr;

   const IntegrationRule *ir = assembly.lom.irF;
   const FiniteElement &el_trace =
      *fes->GetTraceElement(0, fes->GetMesh()->GetFaceBaseGeometry(0));
   const DofToQuad *maps = &el_trace.GetDofToQuad(*ir, DofToQuad::TENSOR);

   auto B = mfem::Reshape(maps->B.Read(), quad1D, dofs1D);

   if (type == FaceType::Interior)
   {
      face_restrict_lex =
         fes->GetFaceRestriction(ElementDofOrdering::LEXICOGRAPHIC,type);

      Vector x_loc(face_restrict_lex->Height());
      Vector y_loc(face_restrict_lex->Height());

      //Apply face integrator restriction
      face_restrict_lex->Mult(x, x_loc);
      y_loc = 0.0;

      auto X = mfem::Reshape(x_loc.Read(), dofs1D, 2, nf);
      auto Y = mfem::Reshape(y_loc.ReadWrite(), dofs1D, 2, nf);
      auto D = mfem::Reshape(D_int.Read(), quad1D, 2, nf);

      //MFEM_FORALL(f, nf,
      for (int f=0; f<nf; ++f)
      {
         constexpr int max_D1D = MAX_D1D;
         constexpr int max_Q1D = MAX_Q1D;

         double Bu0[max_Q1D];
         double Bu1[max_Q1D];

         for (int q=0; q<Q1D; ++q)
         {
            Bu0[q] = 0.0;
            Bu1[q] = 0.0;

            //we are lumping the terms
            for (int d=0; d < D1D; ++d)
            {
               Bu0[q] += B(q,d) * 1.0;
               Bu1[q] += B(q,d) * 1.0;
            }
         }

         //Scale with quadrature data
         double DBu0[max_Q1D];
         double DBu1[max_Q1D];
         for (int q=0; q<Q1D; ++q)
         {
            DBu0[q] = Bu0[q] * D(q, 0, f);
            DBu1[q] = Bu1[q] * D(q, 1, f);
         }

         //Apply Bt
         for (int d=0; d<D1D; ++d)
         {

            double res0(0.0), res1(0.0);
            for (int q=0; q<Q1D; ++q)
            {
               res0 += B(q,d) * DBu0[q];
               res1 += B(q,d) * DBu1[q];
            }

            Y(d, 0, f) = res0 * (X(d,1,f) - X(d,0,f));
            Y(d, 1, f) = res1 * (X(d,0,f) - X(d,1,f));
         }

      }

      face_restrict_lex->MultTranspose(y_loc,y);
   }


   if (type == FaceType::Boundary)
   {

      face_restrict_lex =
         fes->GetFaceRestriction(ElementDofOrdering::LEXICOGRAPHIC,type,
                                 L2FaceValues::SingleValued);

      Vector x_loc(face_restrict_lex->Height());
      Vector y_loc(face_restrict_lex->Height());

      //Apply face integrator restriction
      face_restrict_lex->Mult(x, x_loc);
      y_loc = 0.0;

      auto X = mfem::Reshape(x_loc.Read(), dofs1D, nf);
      auto Y = mfem::Reshape(y_loc.ReadWrite(), dofs1D, nf);
      auto D = mfem::Reshape(D_bdry.Read(), quad1D, nf);

      //MFEM_FORALL(f, nf,
      for (int f=0; f<nf; ++f)
      {
         constexpr int max_D1D = MAX_D1D;
         constexpr int max_Q1D = MAX_Q1D;

         double Bu0[max_Q1D];
         for (int q=0; q<Q1D; ++q)
         {
            Bu0[q] = 0.0;

            //we are lumping the terms
            for (int d=0; d < D1D; ++d)
            {
               Bu0[q] += B(q,d) * 1.0;
            }
         }

         //Scale with quadrature data
         double DBu0[max_Q1D];
         for (int q=0; q<Q1D; ++q)
         {
            DBu0[q] = Bu0[q] * D(q, f);
         }

         //Apply Bt
         for (int d=0; d<D1D; ++d)
         {

            double res0(0.0);
            for (int q=0; q<Q1D; ++q)
            {
               res0 += B(q,d) * DBu0[q];
            }

            Y(d, f) = -res0 * X(d,f);
         }

      }

      face_restrict_lex->MultTranspose(y_loc,y);

   }




}

void MFResidualDistribution::ApplyFaceTerms3D(const Vector &x, Vector &y,
                                              FaceType type) const
{

}


void MFResidualDistribution::CalcLOSolution(const Vector &u, Vector &du) const
{

   const int ndof = pfes.GetFE(0)->GetDof();
   const int ne = pfes.GetMesh()->GetNE();
   Vector z(u.Size());

   const double eps = 1.E-15;
   const double infinity = numeric_limits<double>::infinity();

   // Discretization terms
   du = 0.;
   K.Mult(u, z);

   //z = Conv * u

   ParGridFunction u_gf(&pfes);
   u_gf = u;
   u_gf.ExchangeFaceNbrData();

   z.HostReadWrite();
   u.HostRead();
   du.HostReadWrite();
   M_lumped.HostRead();

#if 1
   SampleVelocity(FaceType::Interior);
   SampleVelocity(FaceType::Boundary);
   SetupPA(FaceType::Interior);
   SetupPA(FaceType::Boundary);
   ApplyFaceTerms(u, du, FaceType::Interior);
   ApplyFaceTerms(u, du, FaceType::Boundary);
#else
   //Apply the face terms
   assembly.DeviceLinearFluxLumping(u, du, FaceType::Interior);
   assembly.DeviceLinearFluxLumping(u, du, FaceType::Boundary);
#endif

   //initialize to infinity
   assembly.dofs.xe_min =  infinity;
   assembly.dofs.xe_max = -infinity;

   double *xe_min = assembly.dofs.xe_min.ReadWrite();
   double *xe_max = assembly.dofs.xe_max.ReadWrite();

   const double *d_u = u.Read();
   const double *d_z = z.Read();
   const double *d_M_lumped = M_lumped.Read();

   double *d_du = du.ReadWrite();

   MFEM_FORALL(k, ne,
   {

      // Boundary contributions - stored in du
      // done before this loop

      // Element contributions
      double rhoP(0.), rhoN(0.), xSum(0.);
      for (int j = 0; j < ndof; ++j)
      {
         int dof_id = k*ndof+j;
         xe_max[k] = max(xe_max[k], d_u[dof_id]);
         xe_min[k] = min(xe_min[k], d_u[dof_id]);
         xSum += d_u[dof_id];
         rhoP += max(0., d_z[dof_id]);
         rhoN += min(0., d_z[dof_id]);
      }

      //denominator of equation 47
      double sumWeightsP = ndof*xe_max[k] - xSum + eps;
      double sumWeightsN = ndof*xe_min[k] - xSum - eps;

      for (int i = 0; i < ndof; i++)
      {
         int dof_id = k*ndof+i;
         //eq 46
         double weightP = (xe_max[k] - d_u[dof_id]) / sumWeightsP;
         double weightN = (xe_min[k] - d_u[dof_id]) / sumWeightsN;

         // (lumpped trace term  + LED convection )/lumpped mass matrix
         d_du[dof_id] = (d_du[dof_id] + weightP * rhoP + weightN * rhoN) /
         d_M_lumped[dof_id];
      }

   });

}

} // namespace mfem
