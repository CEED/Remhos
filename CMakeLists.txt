cmake_minimum_required(VERSION 3.21)

set(project remhos)
project(${project} LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CXX flags *******************************************************************
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-fsanitize=address -O0)
    add_link_options(-fsanitize=address)
endif()

# remove -DNDEBUG from default RelWithDebInfo flags
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2" CACHE STRING "" FORCE)

# Verbosity options ***********************************************************
set(CMAKE_VERBOSE_MAKEFILE OFF CACHE BOOL "" FORCE)

# Dependencies ****************************************************************
find_package(HYPRE REQUIRED)
find_package(MFEM REQUIRED)
find_package(MPI REQUIRED)

option(REMHOS_USE_CALIPER "Use Caliper" OFF)
if (REMHOS_USE_CALIPER)
    find_package(caliper REQUIRED)
    find_package(adiak REQUIRED)
endif()

# HIP support *****************************************************************
# Read MFEM_USE_HIP from MFEMConfig or from the cache (passed via -DMFEM_USE_HIP=ON)
if (NOT DEFINED MFEM_USE_HIP)
    set(MFEM_USE_HIP OFF)
endif()

if (MFEM_USE_HIP)
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES gfx942)
    endif()
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD ${CMAKE_CXX_STANDARD} CACHE STRING
        "HIP standard to use.")
    set(CMAKE_HIP_STANDARD_REQUIRED ON CACHE BOOL
        "Force the use of the chosen HIP standard.")
    set(CMAKE_HIP_EXTENSIONS OFF CACHE BOOL "Enable HIP standard extensions.")
    message(STATUS "Using HIP architecture: ${CMAKE_HIP_ARCHITECTURES}")
endif()

# Copy mesh files *************************************************************
file(GLOB SRC_MESH_FILES LIST_DIRECTORIES false data/*.mesh)
set(BUILD_DATA_DIR ${CMAKE_CURRENT_BINARY_DIR}/data)
add_custom_command(OUTPUT data_is_copied
    COMMAND ${CMAKE_COMMAND} -E make_directory ${BUILD_DATA_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC_MESH_FILES} ${BUILD_DATA_DIR}
    COMMAND ${CMAKE_COMMAND} -E touch data_is_copied
    COMMENT "Copying mesh files ...")
add_custom_target(copy_mesh_files DEPENDS data_is_copied)

# Library source files ********************************************************
set(LIB_SOURCES
    remhos.cpp
    remhos_fct.cpp
    remhos_ho.cpp
    remhos_lo.cpp
    remhos_mono.cpp
    remhos_solvers.cpp
    remhos_sync.cpp
    remhos_tools.cpp
)

if (MFEM_USE_HIP)
    set_source_files_properties(${LIB_SOURCES} PROPERTIES LANGUAGE HIP)
endif()

add_library(Remhos STATIC ${LIB_SOURCES})

target_compile_definitions(Remhos PUBLIC MFEM_USE_CMAKE_TESTS)

target_include_directories(Remhos PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${MFEM_INCLUDE_DIRS}
    ${MFEM_INCLUDE_DIRS}/mfem
)

target_link_libraries(Remhos
    PUBLIC mfem
    PUBLIC MPI::MPI_CXX
    PUBLIC HYPRE::HYPRE
)

if (REMHOS_USE_CALIPER)
    target_compile_definitions(Remhos PUBLIC REMHOS_USE_CALIPER)
    target_link_libraries(Remhos PUBLIC caliper adiak::adiak)
endif()

# Executables *****************************************************************
if (MFEM_USE_HIP)
    set_source_files_properties(remhos_main.cpp PROPERTIES LANGUAGE HIP)
endif()
add_executable(remhos_main remhos_main.cpp)
add_dependencies(remhos_main copy_mesh_files)
target_link_libraries(remhos_main PRIVATE Remhos)

# Install *********************************************************************
install(TARGETS remhos_main DESTINATION bin)
install(DIRECTORY data/ DESTINATION data)

# Testing *********************************************************************
enable_testing()

function(add_remhos_test name)
    add_executable(${name} ${name}.cpp)
    add_dependencies(${name} copy_mesh_files)
    target_link_libraries(${name} PRIVATE Remhos)
    if (MFEM_USE_HIP)
        set_source_files_properties(${name}.cpp PROPERTIES LANGUAGE HIP)
    endif()
    add_test(NAME ${name}_n1
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 $<TARGET_FILE:${name}>)
    add_test(NAME ${name}_n3
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3 $<TARGET_FILE:${name}>)
endfunction()

add_test(NAME remhos_main_n1
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 $<TARGET_FILE:remhos_main>)
add_test(NAME remhos_main_n3
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3 $<TARGET_FILE:remhos_main>)

add_remhos_test(remhos_tests)
