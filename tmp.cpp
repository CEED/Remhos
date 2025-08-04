   else if (opt_type == 4)
   {
      Vector target_volume(1); target_volume[0] = mass_0;
      ScalarLatentVolumeProjector projector(target_volume, pos_final,
                                            *u_final.ParFESpace(), u_final);
      ParGridFunction psi(u_interpolated);
      Vector search_l({infinity()}), search_r({-infinity()}), lambda(1);
      for (int i=0; i<u_interpolated.Size(); i++)
      {
         psi[i] = inv_sigmoid(psi[i], u_final_min[i], u_final_max[i]);
         search_l[0] = std::min(search_l[0], psi[i]);
         search_r[0] = std::max(search_r[0], psi[i]);
      }
      MPI_Allreduce(MPI_IN_PLACE, &search_l[0], 1, MFEM_MPI_REAL_T, MPI_MIN,
                    pmesh_init.GetComm());
      MPI_Allreduce(MPI_IN_PLACE, &search_r[0], 1, MFEM_MPI_REAL_T, MPI_MAX,
                    pmesh_init.GetComm());
      projector.SetVerbose(2);
      projector.Apply(psi, u_final_min, u_final_max, 1.0, search_l, search_r,
                      lambda, max_iter);
   }
   else if (opt_type == 4)
   {
      Vector target_volume(1); target_volume[0] = mass_0;
      ScalarLatentVolumeProjector projector(target_volume, pos_final,
                                            *u.GetSpace(), u);
      QuadratureFunction psi(u);
      Vector search_l({infinity()}), search_r({-infinity()}), lambda(1);
      for (int i=0; i<psi.Size(); i++)
      {
         psi[i] = inv_sigmoid(psi[i], u_min[i], u_max[i]);
         search_l[0] = std::min(search_l[0], psi[i]);
         search_r[0] = std::max(search_r[0], psi[i]);
      }
      MPI_Allreduce(MPI_IN_PLACE, &search_l[0], 1, MFEM_MPI_REAL_T, MPI_MIN,
                    pmesh_init.GetComm());
      MPI_Allreduce(MPI_IN_PLACE, &search_r[0], 1, MFEM_MPI_REAL_T, MPI_MAX,
                    pmesh_init.GetComm());
      projector.SetVerbose(2);
      projector.Apply(psi, u_min, u_max, 1.0, search_l, search_r, lambda, max_iter);
   }
   else if (opt_type == 4)
   {
      Vector target_volume(3);
      target_volume[0] = volume_0;
      target_volume[1] = mass_0;
      target_volume[2] = energy_0;
      IndRhoEVolumeProjectorCorrect projector(target_volume, pos_final,
                                              *qspace, *pfes_e, ind_rho_e);
      Vector psi(ind_rho_e);
      int offset = 0;
      for (int i=0; i<ind.Size(); i++)
      {
         psi[i] = logit(psi[i], ind_min[i], ind_max[i]);
      }
      offset += ind.Size();
      for (int i=0; i<rho.Size(); i++)
      {
         psi[offset + i] = inv_sigmoid(psi[offset + i], rho_min[i], rho_max[i]);
      }
      offset += rho.Size();
      L2_FECollection nodal_fec(pfes_e->GetOrder(0), pfes_e->GetParMesh()->Dimension());
      ParFiniteElementSpace pfes_nodal(pfes_e->GetParMesh(), &nodal_fec);
      ParGridFunction E_gf(pfes_e, ind_rho_e.GetData() + offset);
      ParGridFunction lower_gf(&pfes_nodal, e_min);
      ParGridFunction upper_gf(&pfes_nodal, e_max);
      LogitCoefficient logit_coeff(E_gf, lower_gf, upper_gf);
      ParGridFunction psi_gf(&pfes_nodal, psi.GetData() + offset);
      psi_gf.ProjectCoefficient(logit_coeff);
      projector.SetVerbose(1);
      Vector search_l, search_r, lambda; // not used anymore..
      projector.Apply(psi, x_min, x_max, 1e-01,
                      search_l, search_r, lambda, 1e03);
   }
