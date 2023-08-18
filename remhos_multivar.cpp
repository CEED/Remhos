#include "remhos_multivar.hpp"
#include <fem/coefficient.hpp>
#include <fstream>
#include <iostream>
#include <numeric>

namespace mfem {
   VariableSystem::VariableSystem(int nmat, VARIABLE_SET varset, ParFiniteElementSpace &pfes)
      : indicator_sum_gf(&pfes), nmat(nmat), varset(varset), pfes(pfes),
      neq( [varset, nmat]()-> int { // initialize neq based on variable set
         switch(varset){
            case VARIABLE_SET::MATERIAL_INDICATORS:
               return nmat;
            case VARIABLE_SET::CONSERVATIVE_VARIABLES_N:
               return 3 * nmat;
            case VARIABLE_SET::PRIMITIVE_VARIABLES:
               return 3 * nmat;
            default:
               return 0;
         }
      }()), offset(neq + 1), udata(), u_vec() 
   {
      // setup the block vector 
      int vsize = pfes.GetVSize(); 
      for(int i = 0; i < offset.Size(); ++i){ offset[i] = i * vsize; }
      udata.Update(offset, Device::GetMemoryType());
      // setup the pargridfunction references
      u_vec.reserve(neq);
      for(int i = 0; i < neq; ++i){ u_vec.emplace_back(&pfes, udata, i * vsize); }

      // Coefficients that get reused for output
      // get the sum of indicators
      for(int imat = 0; imat < nmat; ++imat) { mat_coeffs.push_back(GridFunctionCoefficient(&u_vec[imat])); }
      sum_coeffs.reserve(nmat - 1); // prevent pointer invalidation
      if(nmat > 1)
         sum_coeffs.push_back(SumCoefficient(mat_coeffs[0], mat_coeffs[1]));
      else
         sum_coeffs.push_back(SumCoefficient(0.0, mat_coeffs[0])); // if single material add to a zero field
      for(int imat = 2; imat < nmat; ++imat){
         sum_coeffs.push_back(SumCoefficient(mat_coeffs[imat], sum_coeffs[imat - 2]));
      }
   }

   ParGridFunction VariableSystem::computeDensity(int imat) {
      if(varset == VARIABLE_SET::CONSERVATIVE_VARIABLES_N){
         GridFunctionCoefficient Yi(&getIndicator(imat));
         GridFunctionCoefficient Yirhoi(&u_vec[nmat + imat]);
         TransformedCoefficient rhoi_coeff(&Yi, &Yirhoi, 
            [](double Y, double Yrho)->double { 
               // protect from material void
               return (std::abs(Y) < 1e-8) ? 0.0 : Yrho / Y; 
            });
         ParGridFunction rhoi(&pfes);
         rhoi.ProjectCoefficient(rhoi_coeff);
         return rhoi;
      } else if (varset == VARIABLE_SET::PRIMITIVE_VARIABLES) {
         return u_vec[nmat + imat];
      } else {
         ParGridFunction rhoi(&pfes);
         rhoi = 1.0;
         return rhoi;
      }
   }

   ParGridFunction VariableSystem::computePressure(int imat){
      if(varset == VARIABLE_SET::CONSERVATIVE_VARIABLES_N){
         return u_vec[2 * nmat + imat];
      } else if (varset == VARIABLE_SET::PRIMITIVE_VARIABLES) {
         GridFunctionCoefficient Y(&getIndicator(imat));
         GridFunctionCoefficient rho(&u_vec[nmat + imat]);
         GridFunctionCoefficient e(&u_vec[2 * nmat + imat]);
         ProductCoefficient prodA(Y, rho);
         ProductCoefficient pres_coeff(e, prodA);
         ParGridFunction pres(&pfes);
         pres.ProjectCoefficient(pres_coeff);
         return pres;
      } else {
         ParGridFunction pres(&pfes);
         pres = 1.0;
         return pres;
      }
   }

   double VariableSystem::calculateMasses(Vector lumpedM, MPI_Comm comm, std::vector<double> &field_masses){
      field_masses.resize(neq);
      for(int ieq = 0; ieq < neq; ++ieq){
         double mass_loc = 0;
         mass_loc = lumpedM * u_vec[ieq];
         MPI_Allreduce(&mass_loc, &field_masses[ieq], 1, MPI_DOUBLE, MPI_SUM, comm);
      }

      double total_mass = 0.0;
      for(int imat = 0; imat < nmat; ++imat){
         double mass_loc = 0;
// TODO: check total mass calculations
         if(varset == VARIABLE_SET::MATERIAL_INDICATORS){
            mass_loc = lumpedM * u_vec[imat];
         } else if (varset == VARIABLE_SET::CONSERVATIVE_VARIABLES_N) {
            mass_loc = lumpedM * u_vec[nmat + imat];
         } else if (varset == VARIABLE_SET::PRIMITIVE_VARIABLES) {

            GridFunctionCoefficient Y(&getIndicator(imat));
            GridFunctionCoefficient rho(&u_vec[nmat + imat]);
            ProductCoefficient partial_dens_coeff(Y, rho);
            ParGridFunction partial_dens(&pfes);
            partial_dens.ProjectCoefficient(partial_dens_coeff);
            mass_loc = lumpedM * partial_dens;
         }
         MPI_Allreduce(&mass_loc, &total_mass, 1, MPI_DOUBLE, MPI_SUM, comm);
      }
//      TODO: product_sync
//      if (product_sync)
//      {
//         const double mass0_us_loc = lumpedM * us;
//         MPI_Allreduce(&mass0_us_loc, &mass0_us, 1, MPI_DOUBLE, MPI_SUM, comm);
//      }
      return total_mass;
   }

   void VariableSystem::fieldMaxes(MPI_Comm comm, std::vector<double> &field_maxes){
      field_maxes.resize(neq);
      for(int ieq = 0; ieq < neq; ++ieq){
         const double max_loc = u_vec[ieq].Max();
         MPI_Allreduce(&max_loc, &field_maxes[ieq], 1, MPI_DOUBLE, MPI_MAX, comm);
      }
   }

   void VariableSystem::renormalize(){
      int ndof = u_vec[0].Size();
      for(int idof = 0; idof < ndof; ++idof){
         double sum = 0.0;
         for(int imat = 0; imat < nmat; ++imat){
            sum += udata[imat * ndof + idof];
         }
         for(int imat = 0; imat < nmat; ++imat){
            udata[imat * ndof + idof] /= sum;
         }
      }
   }
      
   void VariableSystem::printGridFunctions(std::string suffix, int precision){
      using namespace std;
      for(int imat = 0; imat < nmat; ++imat){
         // print out material indicator
         ofstream Y_out("Y" + std::to_string(imat) + suffix + ".gf");
         Y_out.precision(precision);
         getIndicator(imat).SaveAsOne(Y_out);

         // print out indicator sum
         ofstream sum_out("indicator_sum" + suffix + ".gf");
         ParGridFunction sum_indicator(&pfes);
         sum_indicator.ProjectCoefficient(sum_coeffs.back());
         sum_indicator.SaveAsOne(sum_out);

         //print out density
         ofstream dens_out("rho" + std::to_string(imat) + suffix + ".gf");
         dens_out.precision(precision);
         computeDensity(imat).SaveAsOne(dens_out);

         // print out pressure
         ofstream pres_out("pres" + std::to_string(imat) + suffix + ".gf");
         pres_out.precision(precision);
         computePressure(imat).SaveAsOne(pres_out);
      }
   }

   void VariableSystem::initDataCollection(DataCollection *dc){
      for(int imat = 0; imat < nmat; ++ imat) 
         { dc->RegisterField("material_indicator_" + std::to_string(imat), &getIndicator(imat)); }
      // get the sum of indicators
      std::vector<GridFunctionCoefficient> mat_coeffs;
      for(int imat = 0; imat < nmat; ++imat) { mat_coeffs.push_back(GridFunctionCoefficient(&u_vec[imat])); }
      std::vector<SumCoefficient> sum_coeffs;
      sum_coeffs.reserve(nmat - 1); // prevent pointer invalidation
      if(nmat > 1)
         sum_coeffs.push_back(SumCoefficient(mat_coeffs[0], mat_coeffs[1]));
      else
         sum_coeffs.push_back(SumCoefficient(0.0, mat_coeffs[0])); // if single material add to a zero field
      for(int imat = 2; imat < nmat; ++imat){
         sum_coeffs.push_back(SumCoefficient(mat_coeffs[imat], sum_coeffs[imat - 2]));
      }
      indicator_sum_gf.ProjectCoefficient(sum_coeffs.back());
      dc->RegisterField("indicator_sum", &indicator_sum_gf);

// TODO: rvalue move semantics not working
//      for(int imat = 0; imat < nmat; ++imat){
//         densities.push_back(std::move(computeDensity(imat)));
//         dc->RegisterField("density_" + std::to_string(imat), &densities[imat]);
//      }
//      for(int imat = 0; imat < nmat; ++imat){
//         pressures.push_back(computePressure(imat));
//         dc->RegisterField("pressure_" + std::to_string(imat), &pressures[imat]);
//      }
   }

   void VariableSystem::updateDataCollectionFields(){
      for(int imat = 0; imat < nmat; ++imat){
 //        densities[imat] = computeDensity(imat);
 //        pressures[imat] = computePressure(imat);
         indicator_sum_gf.ProjectCoefficient(sum_coeffs.back());
      }
   }
}

