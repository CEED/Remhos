# Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
# the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
# reserved. See files LICENSE and NOTICE for details.
#
# This file is part of CEED, a collection of benchmarks, miniapps, software
# libraries and APIs for efficient high-order finite element and spectral
# element discretizations for exascale applications. For more information and
# source code availability see http://github.com/ceed.
#
# The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
# a collaborative effort of two U.S. Department of Energy organizations (Office
# of Science and the National Nuclear Security Administration) responsible for
# the planning and preparation of a capable exascale ecosystem, including
# software, applications, hardware, advanced system engineering and early
# testbed platforms, in support of the nation's exascale computing imperative.

cmake_minimum_required(VERSION 3.8)

# Require C++11 and disable compiler-specific extensions
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Project name
project(REMHOS VERSION 1.0)


#  =========================
#  = Find the mfem package =
#  =========================
# Makefile uses ../mfem
# this would be the cmake equivalent for a build
set(MFEM_BUILD_DIR ${CMAKE_SOURCE_DIR}/../mfem/build)
message("${MFEM_BUILD_DIR}")
if(MFEM_BUILD_DIR)
    # searches for MFEMConfig.cmake in the given path
    find_package(mfem REQUIRED NAMES MFEM HINTS ${MFEM_BUILD_DIR} NO_DEFAULT_PATH)
else()
    find_package(mfem REQUIRED NAMES MFEM)
endif()


set(SOURCE_FILES
    remhos.cpp
    remhos_tools.cpp
    remhos_lo.cpp
    remhos_ho.cpp
    remhos_fct.cpp
    remhos_mono.cpp
    remhos_sync.cpp
)

add_executable(remhos ${SOURCE_FILES})
target_link_libraries(remhos PRIVATE ${MFEM_LIBRARIES})
target_include_directories(remhos PRIVATE ${MFEM_INCUDE_DIRS})


