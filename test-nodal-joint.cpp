// Copyright (c) 2026, Lawrence Livermore National Security, LLC.
// See files LICENSE and NOTICE for details.

#include "remhos_nodal.hpp"
#include <cmath>
#include <iostream>
#include <limits>

using namespace mfem;

// Manufacture the target from a known primal/dual KKT solution, rather than
// comparing two executions of the same solver. Run on 1, 2, and 4 MPI ranks.
int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   const MPI_Comm comm = MPI_COMM_WORLD;
   const int rank = Mpi::WorldRank(), ranks = Mpi::WorldSize();
   int failures = 0;
   for (int test = 0; test < 6; test++)
   {
      const int blocks = test == 3 ? 2 : 8;
      const int copies = test == 1 ? 16 : 1;
      DenseMatrix A(6*copies, 2);
      Vector b(6*copies);
      A = 0.0;
      b = 0.0;
      for (int k = 0; k < copies; k++)
      {
         const int i = 6*k;
         // Duplicate/scaled inequalities exercise redundant control-point rows.
         const real_t scale = 1 + k;
         A(i,0) = A(i+1,1) = scale;
         A(i+2,0) = A(i+3,1) = -scale;
         b(i+2) = b(i+3) = -scale;
         A(i+4,0) = scale; A(i+4,1) = -0.5*scale;
         A(i+5,0) = 0.25*scale; A(i+5,1) = -scale;
         b(i+5) = -0.8*scale;
      }
      std::vector<DenseMatrix> mass;
      std::vector<Vector> target, expected, result;
      real_t desired_mass = 0;
      for (int e = rank; e < blocks; e += ranks)
      {
         DenseMatrix M(2);
         const real_t scale = 0.1*(e+1);
         M(0,0) = 2*scale; M(1,1) = scale;
         M(0,1) = M(1,0) = 0.3*scale;
         Vector x(2), z(A.Height()), one(2), c(2), atz(2), shift(2);
         z = 0.0;
         one = 1.0;
         switch (e%4)
         {
            case 0: x(0)=0.2; x(1)=0.4; z(4)=0.03*scale; break;
            case 1: x(0)=0.6; x(1)=0.95; z(5)=0.02*scale; break;
            case 2: x=0.0; z(0)=0.01*scale; z(1)=0.015*scale; break;
            default: x(0)=0.7; x(1)=0.3;
         }
         if (test == 2) { x = 0.0; z = 0.0; }
         if (test == 4) { x = 1.0; z = 0.0; }
         if (test == 5) { z = 0.0; }
         A.MultTranspose(z, atz);
         DenseMatrixInverse inverse(M, true);
         inverse.Mult(atz, shift);
         Vector t(x);
         t -= shift;
         if (test != 5) { t.Add(0.07, one); }
         M.Mult(one, c);
         desired_mass += c*x;
         mass.push_back(M);
         target.push_back(t);
         expected.push_back(x);
      }
      MPI_Allreduce(MPI_IN_PLACE, &desired_mass, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
      NodalRemapOptions options;
      options.upper_constraint = true;
      options.upper = 1;
      options.mass_solver = 4;
      const JointFitInfo info = FitConservative(comm, mass, target, A, b,
                                                desired_mass, options, result);
      real_t error = 0, measured_mass = 0, violation = 0;
      for (int e = 0; e < int(result.size()); e++)
      {
         Vector difference(result[e]), slack(A.Height()), one(2), c(2);
         difference -= expected[e];
         error = std::max(error, difference.Normlinf());
         A.Mult(result[e], slack);
         slack -= b;
         violation = std::max(violation, -slack.Min());
         one = 1.0;
         mass[e].Mult(one, c);
         measured_mass += c*result[e];
      }
      MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX, comm);
      MPI_Allreduce(MPI_IN_PLACE, &violation, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX, comm);
      MPI_Allreduce(MPI_IN_PLACE, &measured_mass, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
      const bool passed = info.converged && std::isfinite(error) && error < 2e-6 &&
                          violation < 1e-9 && std::abs(measured_mass-desired_mass) < 1e-9;
      failures += !passed;
      if (test == 5) { failures += info.iterations != 0 || error != 0; }
      if (!rank)
      {
         std::cout << "KKT test " << test << ": " << (passed ? "PASS" : "FAIL")
                   << ", error=" << error << ", iterations=" << info.iterations
                   << ", stationarity=" << info.stationarity << ", gap=" << info.gap << '\n';
      }
      if (test == 0)
      {
         options.mass_max_iterations = 1;
         failures += FitConservative(comm, mass, target, A, b, desired_mass,
                                     options, result).converged;
         options.mass_max_iterations = 100;
         failures += FitConservative(comm, mass, target, A, b, -1,
                                     options, result).converged;
      }
   }
   if (!rank) { std::cout << "Joint QP regression failures: " << failures << '\n'; }
   return failures ? 1 : 0;
}
