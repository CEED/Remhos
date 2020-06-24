cd ../ &&
git clone --recursive git@github.com:hypre-space/hypre.git &&
cd hypre/src/ &&
./configure --disable-fortran &&
make -j 16 &&
cd ../.. &&
wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/OLD/metis-4.0.3.tar.gz &&
tar -zxvf metis-4.0.3.tar.gz && cd metis-4.0.3 &&
make -j 3 && cd .. &&
ln -s metis-4.0.3 metis-4.0 &&
git clone --recursive https://github.com/mfem/mfem.git &&
cd mfem && make pcudebug MFEM_USE_SIMD=NO CUDA_ARCH=sm_70 -j 10
cd ../Remhos && make -j 10
