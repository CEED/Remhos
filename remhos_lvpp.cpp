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

#include "remhos_lvpp.hpp"

namespace mfem
{

void Dykstra::Project(Vector &projected_x)
{
   const int num_con = constraints.NumRows();
   Vector con_res(num_con);
   const int N = projected_x.Size();
   int total_dof;
   MPI_Allreduce(&N, &total_dof, 1, MPI_INT, MPI_SUM, comm);
   if (Mpi::Root())
   {
      out << "Dykstra::Project with " << num_con << " constraints over " << total_dof
          << std::endl;
   }

   constraints.Mult(projected_x, con_res);
   if (con_res.Normlinf() < tol)
   {
      if (Mpi::Root())
      {
         out << "Constraint violation is already small enough. Terminate without iterations"
             << std::endl;
      }
      return;
   }

   std::vector<std::unique_ptr<Vector>> q(num_con + enforce_sum_to_one);
   Vector grad(N), deriv(N);
   for (int i = 0; i < num_con + enforce_sum_to_one; ++i)
   {
      q[i] = std::make_unique<Vector>(N);
      *q[i] = 0.0;
   }

   Vector psi(N);
   psi = 0.0;
   MapPrimal(projected_x, xmin, xmax, psi);
   Vector psi_aux(psi);
   Vector psi_prev(psi);
   Vector psi_full_prev(psi);
   Vector x_full_prev(psi);
   Vector primal_dir(N);
   Vector latent_dir(N);
   Vector merit_grad(N);
   Vector con_val(1);
   Vector con_sgn(num_con);
   // return;
   for (int iter=0; iter<max_iter; iter++)
   {
      psi_full_prev = psi;
      x_full_prev = projected_x;
      // Update residual before projection
      if (shared_constraints) { shared_constraints->Update(projected_x); }
      constraints.Mult(projected_x, con_res);
      // Baseline from the previous iteration. L1 norm.
      // We also compute the sign so that we can obtain the gradient direction
      real_t baseline = 0;
      for (int i=0; i<num_con; i++)
      {
         con_sgn[i] = con_res[i] < 0 ? -1.0 : con_res[i] > 0 ? 1.0 : 0.0;
         baseline += std::abs(con_res[i]);
      }
      constraints.GetGradient(projected_x).Mult(con_sgn, merit_grad);
      // Cyclic projection
      for (int i=0; i<num_con; i++)
      {
         // Get current set information
         Vector &qi = *q[i];
         Functional &con = constraints.GetFunctional(i);

         // Store previous
         psi_prev = psi;
         MapLatent(psi, xmin, xmax, projected_x); // update x
         psi += qi;
         // apply a few iteration of tangential projection
         for (int j=0; j<3; j++)
         {
            // Projection target:
            // c(x) + <grad c(x), x_proj - x> = 0
            // -> <grad c(x), x_proj> = <grad c(x), x> - c(x)
            if (shared_constraints) { shared_constraints->Update(projected_x); }
            con.Mult(projected_x, con_val);
            con.GetGradient().Mult(projected_x, deriv);
            mass.Riesz(deriv, grad);
            real_t targ = mass.InnerProduct(grad, projected_x) - con_val(0);

            Project(con, psi, grad, targ, psi_aux, projected_x);
            if (shared_constraints) { shared_constraints->Update(projected_x); }
            con.Mult(projected_x, con_val);
            if (std::abs(con_val[0])<tol) { break; }
         }

         // Update perturbation
         qi += psi_prev;
         qi -= psi;
      } // full cycle done
      if (enforce_sum_to_one)
      {
         ProjectSumToOne(psi, *q[num_con]);
      }

      MapLatent(psi, xmin, xmax, projected_x);
      if (shared_constraints) { shared_constraints->Update(projected_x); }
      constraints.Mult(projected_x, con_res);

      if (Mpi::Root())
      {
         out << "  Dykstra iteration " << iter << ": constraint violations = (";
         for (int i=0; i<con_res.Size(); i++)
         {
            out << con_res[i] << " ";
         }
         out << "\b)\n" << std::flush;
      }
      if (con_res.Normlinf() < tol)
      {
         if (Mpi::Root())
         {
            out << "Dykstra converged in " << iter << " iterations" << std::endl;
         }
         break;
      }
   }
}
void Dykstra::ProjectSumToOne(Vector &psi, Vector &qi)
{
   const int num_materials = sum_to_one_idx_start.Size();
   Vector curr_psi(num_materials),
          curr_min(num_materials), curr_max(num_materials);
   for (int i=0; i<sum_to_one_block_size; i++) // per quadrature
   {
      real_t psimax = 0.0;
      for (int j=0; j<num_materials; j++)
      {
         const int idx = sum_to_one_idx_start[j] + i;
         // Trial psi: apply Dykstra correction.
         curr_psi[j] = psi[idx] + qi[idx];
         psimax = std::max(psimax, std::abs(curr_psi[j]));
         curr_min[j] = xmin[idx];
         curr_max[j] = xmax[idx];
      }
      // min D_R(x, x0) s.t. sum_j x_j = 1
      // The optimality condition is
      // grad R(x) - grad R(x0) + lambda 1 = 0
      // -> psi = psi0 - lambda 1
      // Here, grad R^* is the scaled sigmoid function.
      // We can solve for lambda using Illinois method
      // f(lambda) = sum_j sigmoid(curr_psi[j] - lambda, ...) - 1
      // f is monotone decreasing in lambda.
      // For the initial bracket, we use (-1,1)*||psi||_inf.
      real_t a = -psimax, b = psimax;
      auto eval = [&](real_t lam) -> real_t
      {
         real_t s = 0.0;
         for (int j=0; j<num_materials; j++)
         {
            s += sigmoid(curr_psi[j] - lam, curr_min[j], curr_max[j]);
         }
         return s - 1.0;
      };
      if (fabs(eval(0.0)) < tol)
      {
         continue;
      }

      real_t fa = eval(a), fb = eval(b);
      // f is decreasing, so fa >= fb. Expand bracket until root is enclosed.
      real_t diff = b - a;
      while (fa * fb > 0)
      {
         diff *= 2;
         if (fa > 0) // both positive: root is to the right, expand right
         {
            a = b; fa = fb;
            b += diff; fb = eval(b);
         }
         else        // both negative: root is to the left, expand left
         {
            b = a; fb = fa;
            a -= diff; fa = eval(a);
         }
      }
      real_t a0 = a, b0 = b; // for debugging
      real_t fa0 = fa, fb0 = fb;

      // Illinois method (same structure as Dykstra::Project).
      int side = 0; real_t lambda;
      for (int i=0; i<100; i++)
      {
         lambda = (fa*b - fb*a)/(fa - fb);
         real_t f = eval(lambda);
         if (f * fb > 0)
         {
            b = lambda; fb = f;
            if (side == -1) { fa *= 0.5; }
            side = -1;
         }
         else
         {
            a = lambda; fa = f;
            if (side == 1) { fb *= 0.5; }
            side = 1;
         }
         if (std::abs(f) < tol)
         {
            break;
         }
      }
      // out << "      ProjectSumToOne: lambda = " << lambda << ", f(lambda) = " << eval(
      //        lambda)
      //     << ", initial bracket = (" << a0 << ", " << b0 << ") with "
      //     << "f(a0) = " << fa0 << ", f(b0) = " << fb0 << std::endl;

      // Apply shift and update psi and qi (Dykstra pattern).
      // qi_new = old_qi + psi_prev - psi_new = lambda (simplifies algebraically)
      for (int j=0; j<num_materials; j++)
      {
         const int idx = sum_to_one_idx_start[j] + i;
         const real_t psi_prev = psi[idx];
         psi[idx] = curr_psi[j] - lambda;
         qi[idx] += psi_prev - psi[idx];
      }
   }
}

void Dykstra::Project(const Functional &con, Vector &psi, const Vector &grad,
                      const real_t targ, Vector &psi_aux, Vector &projected_x)
{
   MapLatent(psi, xmin, xmax, projected_x);
   real_t b = 1e03;
   real_t a = -1e03;
   real_t diff = b - a;

   // update projected_x,
   // psi_aux = psi + shift*grad
   // and compute the linear residual
   // int g dot (sigmoid(psi + shift*grad)) - targ
   auto eval = [&](real_t shift) -> real_t
   {
      add(psi, shift, grad, psi_aux);
      MapLatent(psi_aux, xmin, xmax, projected_x);
      return mass.InnerProduct(grad, projected_x) - targ;
   };
   if (fabs(eval(0.0)) < tol)
   {
      return;
   }

   real_t fa = eval(a);
   real_t fb = eval(b);
   MFEM_VERIFY(fa <= fb,
               "Dykstra::Project: Initial values must satisfy f(a) < f(b). "
               "This is not the case for a = " << a << ", b = " << b
               << ", f(a) = " << fa << ", f(b) = " << fb);

   // if there is no root between a and b
   // shift the interval
   while (fa*fb > 0)
   {
      diff *= 2;
      // Note that int g dot sigmoid(psi + mu*grad) is stritly increasing.
      // Therefore, either (0<fa<fb) or (fa<fb<0)
      if (fa > 0)
      {
         b = a; fb = fa;
         a -= diff; fa = eval(a);
      }
      else
      {
         a = b; fa = fb;
         b += diff; fb = eval(b);
      }
      if (Mpi::Root())
      {
         out << "      Initial values: a = " << a << ", b = " << b
             << ", f(a) = " << fa << ", f(b) = " << fb << std::endl;
      }
   }
   // Now we have a root in (a,b).
   // Search using Illinois method
   int side = 0;
   for (int i=0; i<100; i++)
   {
      real_t c = (fa*b - fb*a)/(fa - fb);
      real_t fc = eval(c);
      if (fc * fb > 0)
      {
         b = c; fb = fc;
         if (side == -1) { fa *= 0.5; }
         side = -1;
      }
      else
      {
         a = c; fa = fc;
         if (side == 1) { fb *= 0.5; }
         side = 1;
      }
      if (std::abs(fc) < tol)
      {
         psi.Add(c, grad);
         break;
      }
   }
}

void Dykstra::MapLatent(const Vector &psi_,
                        const Vector &xmin_,
                        const Vector &xmax_,
                        Vector &x_)
{
   Vector curr_psi;
   Vector curr_x;
   for (int j=0; j<offsets.Size()-1; j++)
   {
      curr_psi.MakeRef(const_cast<Vector&>(psi_), offsets[j],
                       offsets[j+1]-offsets[j]);
      curr_x.MakeRef(x_, offsets[j], offsets[j+1]-offsets[j]);
      legendre_funcs[j]->gradinv(curr_psi, curr_x);
   }
}

void Dykstra::MapPrimal(const Vector &x_,
                        const Vector &xmin_,
                        const Vector &xmax_,
                        Vector &psi_)
{
   Vector curr_psi;
   Vector curr_x;
   for (int j=0; j<offsets.Size()-1; j++)
   {
      curr_x.MakeRef(const_cast<Vector&>(x_), offsets[j],
                     offsets[j+1]-offsets[j]);
      curr_psi.MakeRef(psi_, offsets[j], offsets[j+1]-offsets[j]);
      legendre_funcs[j]->grad(curr_x, curr_psi);
   }
}


}
