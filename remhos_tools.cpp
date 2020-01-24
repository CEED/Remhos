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

#include "remhos_tools.hpp"

using namespace std;

namespace mfem
{

int GetLocalFaceDofIndex3D(int loc_face_id, int face_orient,
                           int face_dof_id, int face_dof1D_cnt)
{
   int k1, k2;
   const int kf1 = face_dof_id % face_dof1D_cnt;
   const int kf2 = face_dof_id / face_dof1D_cnt;
   switch (loc_face_id)
   {
      case 0://BOTTOM
      switch (face_orient)
      {
      case 0://{0, 1, 2, 3}
         k1 = kf1;
         k2 = face_dof1D_cnt-1-kf2;
         break;
      case 1://{0, 3, 2, 1}
         k1 = face_dof1D_cnt-1-kf2;
         k2 = kf1;
         break;
      case 2://{1, 2, 3, 0}
         k1 = face_dof1D_cnt-1-kf2;
         k2 = face_dof1D_cnt-1-kf1;
         break;
      case 3://{1, 0, 3, 2}
         k1 = face_dof1D_cnt-1-kf1;
         k2 = face_dof1D_cnt-1-kf2;
         break;
      case 4://{2, 3, 0, 1}
         k1 = face_dof1D_cnt-1-kf1;
         k2 = kf2;
         break;
      case 5://{2, 1, 0, 3}
         k1 = kf2;
         k2 = face_dof1D_cnt-1-kf1;
         break;
      case 6://{3, 0, 1, 2}
         k1 = kf2;
         k2 = kf1;
         break;
      case 7://{3, 2, 1, 0}
         k1 = kf1;
         k2 = kf2;
         break;
      default:
         mfem_error("This orientation does not exist in 3D");
         break;
      }
      break;
   case 1://SOUTH
      switch (face_orient)
      {
      case 0://{0, 1, 2, 3}
         k1 = kf1;
         k2 = kf2;
         break;
      case 1://{0, 3, 2, 1}
         k1 = kf2;
         k2 = kf1;
         break;
      case 2://{1, 2, 3, 0}
         k1 = kf2;
         k2 = face_dof1D_cnt-1-kf1;
         break;
      case 3://{1, 0, 3, 2}
         k1 = face_dof1D_cnt-1-kf1;
         k2 = kf2;
         break;
      case 4://{2, 3, 0, 1}
         k1 = face_dof1D_cnt-1-kf1;
         k2 = face_dof1D_cnt-1-kf2;
         break;
      case 5://{2, 1, 0, 3}
         k1 = face_dof1D_cnt-1-kf2;
         k2 = face_dof1D_cnt-1-kf1;
         break;
      case 6://{3, 0, 1, 2}
         k1 = face_dof1D_cnt-1-kf2;
         k2 = kf1;
         break;
      case 7://{3, 2, 1, 0}
         k1 = kf1;
         k2 = face_dof1D_cnt-1-kf2;
         break;
      default:
         mfem_error("This orientation does not exist in 3D");
         break;
      }
      break;
   case 2://EAST
      switch (face_orient)
      {
      case 0://{0, 1, 2, 3}
         k1 = kf1;
         k2 = kf2;
         break;
      case 1://{0, 3, 2, 1}
         k1 = kf2;
         k2 = kf1;
         break;
      case 2://{1, 2, 3, 0}
         k1 = kf2;
         k2 = face_dof1D_cnt-1-kf1;
         break;
      case 3://{1, 0, 3, 2}
         k1 = face_dof1D_cnt-1-kf1;
         k2 = kf2;
         break;
      case 4://{2, 3, 0, 1}
         k1 = face_dof1D_cnt-1-kf1;
         k2 = face_dof1D_cnt-1-kf2;
         break;
      case 5://{2, 1, 0, 3}
         k1 = face_dof1D_cnt-1-kf2;
         k2 = face_dof1D_cnt-1-kf1;
         break;
      case 6://{3, 0, 1, 2}
         k1 = face_dof1D_cnt-1-kf2;
         k2 = kf1;
         break;
      case 7://{3, 2, 1, 0}
         k1 = kf1;
         k2 = face_dof1D_cnt-1-kf2;
         break;
      default:
         mfem_error("This orientation does not exist in 3D");
         break;
      }
      break;
   case 3://NORTH
      switch (face_orient)
      {
      case 0://{0, 1, 2, 3}
         k1 = face_dof1D_cnt-1-kf1;
         k2 = kf2;
         break;
      case 1://{0, 3, 2, 1}
         k1 = kf2;
         k2 = face_dof1D_cnt-1-kf1;
         break;
      case 2://{1, 2, 3, 0}
         k1 = kf2;
         k2 = kf1;
         break;
      case 3://{1, 0, 3, 2}
         k1 = kf1;
         k2 = kf2;
         break;
      case 4://{2, 3, 0, 1}
         k1 = kf1;
         k2 = face_dof1D_cnt-1-kf2;
         break;
      case 5://{2, 1, 0, 3}
         k1 = face_dof1D_cnt-1-kf2;
         k2 = kf1;
         break;
      case 6://{3, 0, 1, 2}
         k1 = face_dof1D_cnt-1-kf2;
         k2 = face_dof1D_cnt-1-kf1;
         break;
      case 7://{3, 2, 1, 0}
         k1 = face_dof1D_cnt-1-kf1;
         k2 = face_dof1D_cnt-1-kf2;
         break;
      default:
         mfem_error("This orientation does not exist in 3D");
         break;
      }
      break;
   case 4://WEST
      switch (face_orient)
      {
      case 0://{0, 1, 2, 3}
         k1 = face_dof1D_cnt-1-kf1;
         k2 = kf2;
         break;
      case 1://{0, 3, 2, 1}
         k1 = kf2;
         k2 = face_dof1D_cnt-1-kf1;
         break;
      case 2://{1, 2, 3, 0}
         k1 = kf2;
         k2 = kf1;
         break;
      case 3://{1, 0, 3, 2}
         k1 = kf1;
         k2 = kf2;
         break;
      case 4://{2, 3, 0, 1}
         k1 = kf1;
         k2 = face_dof1D_cnt-1-kf2;
         break;
      case 5://{2, 1, 0, 3}
         k1 = face_dof1D_cnt-1-kf2;
         k2 = kf1;
         break;
      case 6://{3, 0, 1, 2}
         k1 = face_dof1D_cnt-1-kf2;
         k2 = face_dof1D_cnt-1-kf1;
         break;
      case 7://{3, 2, 1, 0}
         k1 = face_dof1D_cnt-1-kf1;
         k2 = face_dof1D_cnt-1-kf2;
         break;
      default:
         mfem_error("This orientation does not exist in 3D");
         break;
      }
      break;
   case 5://TOP
      switch (face_orient)
      {
      case 0://{0, 1, 2, 3}
         k1 = kf1;
         k2 = kf2;
         break;
      case 1://{0, 3, 2, 1}
         k1 = kf2;
         k2 = kf1;
         break;
      case 2://{1, 2, 3, 0}
         k1 = kf2;
         k2 = face_dof1D_cnt-1-kf1;
         break;
      case 3://{1, 0, 3, 2}
         k1 = face_dof1D_cnt-1-kf1;
         k2 = kf2;
         break;
      case 4://{2, 3, 0, 1}
         k1 = face_dof1D_cnt-1-kf1;
         k2 = face_dof1D_cnt-1-kf2;
         break;
      case 5://{2, 1, 0, 3}
         k1 = face_dof1D_cnt-1-kf2;
         k2 = face_dof1D_cnt-1-kf1;
         break;
      case 6://{3, 0, 1, 2}
         k1 = face_dof1D_cnt-1-kf2;
         k2 = kf1;
         break;
      case 7://{3, 2, 1, 0}
         k1 = kf1;
         k2 = face_dof1D_cnt-1-kf2;
         break;
      default:
         mfem_error("This orientation does not exist in 3D");
         break;
      }
      break;
   default: MFEM_ABORT("This face_id does not exist in 3D");
   }
   return k1 + face_dof1D_cnt * k2;
}

int GetLocalFaceDofIndex(int dim, int loc_face_id, int face_orient,
                         int face_dof_id, int face_dof1D_cnt)
{
   switch (dim)
   {
      case 1: return face_dof_id;
      case 2:
         if (loc_face_id <= 1)
         {
            // SOUTH or EAST (canonical ordering)
            return face_dof_id;
         }
         else
         {
            // NORTH or WEST (counter-canonical ordering)
            return face_dof1D_cnt - 1 - face_dof_id;
         }
      case 3: return GetLocalFaceDofIndex3D(loc_face_id, face_orient,
                                            face_dof_id, face_dof1D_cnt);
      default: MFEM_ABORT("Dimension too high!"); return 0;
   }
}


// Assuming L2 elements.
void ExtractBdrDofs(int p, Geometry::Type gtype, DenseMatrix &dofs)
{
   switch (gtype)
   {  
      case Geometry::SEGMENT:
      {
         dofs.SetSize(1, 2);
         dofs(0, 0) = 0;
         dofs(0, 1) = 1;
         break;
      }
      case Geometry::SQUARE:
      {
         dofs.SetSize(p+1, 4);
         for (int i = 0; i <= p; i++)
         {
            dofs(i,0) = i;
            dofs(i,1) = i*(p+1) + p;
            dofs(i,2) = (p+1)*(p+1) - 1 - i;
            dofs(i,3) = (p-i)*(p+1);
         }
         break;
      }
      case Geometry::CUBE:
      {
         dofs.SetSize((p+1)*(p+1), 6);
         for (int bdrID = 0; bdrID < 6; bdrID++)
         {
            int o(0);
            switch (bdrID)
            {
            case 0:
               for (int i = 0; i < (p+1)*(p+1); i++)
               {
                  dofs(o++,bdrID) = i;
               }
               break;
            case 1:
               for (int i = 0; i <= p*(p+1)*(p+1); i+=(p+1)*(p+1))
                  for (int j = 0; j < p+1; j++)
                  {
                     dofs(o++,bdrID) = i+j;
                  }
               break;
            case 2:
               for (int i = p; i < (p+1)*(p+1)*(p+1); i+=p+1)
               {
                  dofs(o++,bdrID) = i;
               }
               break;
            case 3:
               for (int i = 0; i <= p*(p+1)*(p+1); i+=(p+1)*(p+1))
                  for (int j = p*(p+1); j < (p+1)*(p+1); j++)
                  {
                     dofs(o++,bdrID) = i+j;
                  }
               break;
            case 4:
               for (int i = 0; i <= (p+1)*((p+1)*(p+1)-1); i+=p+1)
               {
                  dofs(o++,bdrID) = i;
               }
               break;
            case 5:
               for (int i = p*(p+1)*(p+1); i < (p+1)*(p+1)*(p+1); i++)
               {
                  dofs(o++,bdrID) = i;
               }
               break;
            }
         }
         break;
      }
      default: MFEM_ABORT("Geometry not implemented.");
   }
}

} // namespace mfem
