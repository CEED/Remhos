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
//
//                    ____                 __
//                   / __ \___  ____ ___  / /_  ____  _____
//                  / /_/ / _ \/ __ `__ \/ __ \/ __ \/ ___/
//                 / _, _/  __/ / / / / / / / / /_/ (__  )
//                /_/ |_|\___/_/ /_/ /_/_/ /_/\____/____/
//
//                       High-order Remap Miniapp
//
// Remhos (REMap High-Order Solver) is a miniapp that solves the pure advection
// equations that are used to perform discontinuous field interpolation (remap)
// as part of the Eulerian phase in Arbitrary-Lagrangian Eulerian (ALE)
// simulations.
//
// Sample runs: see README.md, section 'Verification of Results'.

#pragma once
#include "mfem.hpp"
#include <string>
#include <fem/pgridfunc.hpp>


/**
 * @brief the different variable sets available to remap
 */
enum class VARIABLE_SET {
   MATERIAL_INDICATORS,
   CONSERVATIVE_VARIABLES_N, /// Material indicators (eta_i), mass fraction (Y_i*rho_i), internal energy (Y_i*e_i)
   PRIMITIVE_VARIABLES /// Material indicators (eta_i), densities (rho_i), specific internal energies (e_i)
};
namespace mfem {
   
// Forward Declaration
class SmoothnessIndicator;

/**
 * @brief multivariable system
 * Organizes a set of gridfunctions to represent
 * a system of indicator functions and optionally a set of
 * physical quantities
 */
class VariableSystem {
   private:
   // vectors for printing to datacollection
   std::vector<ParGridFunction> densities;
   std::vector<ParGridFunction> pressures;
   std::vector<SumCoefficient> sum_coeffs;
   std::vector<GridFunctionCoefficient> mat_coeffs;
   ParGridFunction indicator_sum_gf;
  
   public:
   int nmat;                     /// the number of materials
   VARIABLE_SET varset;          /// the variable set being used
   ParFiniteElementSpace &pfes;  /// finite element space reference for creating gridfunctions
   int neq;                      /// total number of gridfunctions
   /// offsets for each gridfunction into udata
   Array<int> offset;
   /// Block vector for all the grid functions
   BlockVector udata;
   /// gridfunctions with references into udata for each variable
   /// starts with material indicators
   std::vector<ParGridFunction> u_vec;

   /// to record the initial masses

   VariableSystem(int nmat, VARIABLE_SET varset, ParFiniteElementSpace &pfes);

   // Allow an IC condition function to access the class
   friend void initial_conditions(VariableSystem &u);

   // ===================
   // = Field Accessors =
   // ===================

   /**
    * @brief get the gridfunction for the indicator for imat
    * @param imat the index of the material
    * @return the gridfunction reference corresponding to the given material indicator
    */
   ParGridFunction &getIndicator(int imat) { return u_vec[imat]; }

   ParGridFunction &operator[](int imat) { return u_vec[imat]; }
   
   // =====================
   // = Field Calculation =
   // =====================
   ParGridFunction computeDensity(int imat);

   ParGridFunction computePressure(int imat);

   /**
    * @brief calculate the initial Mass
    * and individual field "mass" to record mass loss per field
    * @param lumpedM the mass matrix to use
    * @param MPI_Comm the mpi communicator
    * @param [out] field_masses the masses for each field
    */
   double calculateMasses(Vector lumpedM, MPI_Comm comm, std::vector<double> &field_masses);

   /**
    * @brief calculate final mass and check for conservation
    * and individual field "masses"
    * @param lumpedM the mass matrix to apply to the dofs
    * @param comm the mpi communicator
    */
   void checkMassConservation(Vector lumpedM, MPI_Comm comm);

   void fieldMaxes(MPI_Comm comm, std::vector<double> &field_maxes);

   // ===========
   // = Utility =
   // ===========
    
   /**
    * @brief renormalize the degrees of freedom over the materials
    * such that the sum of the materials at each dof = 1
    */
   void renormalize();

   void SyncUvec(){ for(int imat = 0; imat < nmat; ++imat) { u_vec[imat].SyncMemory(udata.GetBlock(imat)); }}

   // ==========
   // = Output =
   // ==========
   void printGridFunctions(std::string suffix, int precision);

   void initDataCollection(DataCollection *dc);

   void updateDataCollectionFields();

   void printSmoothnessIndicator(std::string suffix, int precision, SmoothnessIndicator *smth_indicator);
   
}; // end VariableSystem class

   /**
    * @brief apply initial conditions to the variable system
    * allows for user defined initial conditions (friend of VariableSystem)
    * @param u the variable system
    */
   void initial_conditions(VariableSystem &u);
} // end mfem namespace


