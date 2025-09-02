#ifndef MFEM_DYKSTRA_HPP
#define MFEM_DYKSTRA_HPP

#include "mfem.hpp"
#include "general/forall.hpp"
#include "miniapps/autodiff/admfem.hpp"
#include "linalg/functional.hpp"
#include "remap.hpp"

namespace mfem
{
inline real_t allreduce(MPI_Comm comm, real_t val, MPI_Op op)
{
   real_t recv;
   MPI_Allreduce(&val, &recv, 1, MPITypeMap<real_t>::mpi_type, op, comm);
   return recv;
}
class Dykstra
{
   MPI_Comm comm = MPI_COMM_NULL;
   StackedFunctional &constraints;
   StackedSharedFunctional *shared_constraints = nullptr;
   MassOperator &mass;
   const Vector &xmin;
   const Vector &xmax;
   real_t tol;
   int max_iter;
   int max_linesearch = 30;
   real_t c1 = 1e-03; // Armijo condition constant
public:
   Dykstra(MPI_Comm comm, StackedFunctional &constraints, MassOperator &mass,
           const Vector &xmin, const Vector &xmax, real_t tol=1e-10, int max_iter=1000)
      : comm(comm), constraints(constraints), mass(mass)
      , xmin(xmin), xmax(xmax)
      , tol(tol), max_iter(max_iter)
   {
      shared_constraints = dynamic_cast<StackedSharedFunctional*>(&constraints);
   }
   void SetAbsTol(real_t tol) { this->tol = tol; }
   void SetMaxIter(int max_iter) { this->max_iter = max_iter; }

   // Dykstra projection with Bregman divergence
   // At each iteration, we project onto the tangent plane of each constraint
   // psi_k = inv_sigmoid(Project_{k mod N}(sigmoid(psi_{k-1} + q_{k mod N})))
   // q_{k mod N} = psi_{k-1} + q_{k - N mod N} - psi_k
   // where Project_k is the projection onto the k-th constraint (tangent plane)
   void Project(Vector &projected_x)
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

      std::vector<std::unique_ptr<Vector>> q(num_con);
      Vector grad(N), deriv(N);
      for (int i = 0; i < num_con; ++i)
      {
         q[i] = std::make_unique<Vector>(N);
         *q[i] = 0.0;
      }

      Vector psi(N);
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
            for (int j=0; j<2; j++)
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
            // if (Mpi::Root())
            // {
            //    out << "  " << i << "'th tangent: " << con_val[0] << std::endl;
            // }

            // Update perturbation
            qi += psi_prev;
            qi -= psi;
         } // full cycle done

         MapLatent(psi, xmin, xmax, projected_x);
         if (shared_constraints) { shared_constraints->Update(projected_x); }
         constraints.Mult(projected_x, con_res);
         // Start line search
         // subtract(psi, psi_full_prev, latent_dir);
         // real_t step_size = 1.0;
         // for (int i=0; i<max_linesearch; i++)
         // {
         //    add(psi_full_prev, step_size, latent_dir, psi);
         //    MapLatent(psi, xmin, xmax, projected_x);
         //    subtract(projected_x, x_full_prev, primal_dir);
         //    if (shared_constraints) { shared_constraints->Update(projected_x); }
         //    constraints.Mult(projected_x, con_res);
         //    real_t gd = mass.InnerProduct(primal_dir, merit_grad);
         //    if (con_res.Norml1() <= baseline + c1*gd && gd <= 0)
         //    {
         //       break;
         //    }
         //    step_size *= 0.5;
         //    if (i +1 == max_linesearch)
         //    {
         //       if (Mpi::Root())
         //       {
         //          MFEM_WARNING("Could not find search direction. Return");
         //       }
         //       return;
         //    }
         // }

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
private:

   void Project(const Functional &con, Vector &psi, const Vector &grad,
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
      // if (Mpi::Root())
      // {
      //    out << "      Initial values: a = " << a << ", b = " << b
      //        << ", f(a) = " << fa << ", f(b) = " << fb << std::endl;
      // }
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
         // if (Mpi::Root())
         // {
         //    out << "      Illinois iter " << i
         //        << ": a = " << a << ", b = " << b
         //        << ", c = " << c
         //        << ", f(c) = " << fc << std::endl;
         // }
         if (std::abs(fc) < tol)
         {
            psi.Add(c, grad);
            break;
         }
      }
   }

   static void MapLatent(const Vector &psi_,
                         const Vector &xmin_,
                         const Vector &xmax_,
                         Vector &x_)
   {
      const real_t* psi = psi_.Read();
      const real_t* xmin = xmin_.Read();
      const real_t* xmax = xmax_.Read();
      real_t* x = x_.Write();
      MFEM_FORALL(i, x_.Size(), x[i] = sigmoid(psi[i], xmin[i], xmax[i]););
   }

   static void MapPrimal(const Vector &x_,
                         const Vector &xmin_,
                         const Vector &xmax_,
                         Vector &psi_)
   {
      const real_t* x = x_.Read();
      const real_t* xmin = xmin_.Read();
      const real_t* xmax = xmax_.Read();
      real_t* psi = psi_.Write();
      MFEM_FORALL(i, x_.Size(), psi[i] = std::clamp(logit(x[i], xmin[i], xmax[i]),
                                         -20., 20.););
   }

};
} // namespace mfem

#endif // MFEM_REMAP_HPP
