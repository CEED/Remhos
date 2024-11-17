cmake_minimum_required(VERSION 3.18)

set(project remhos)
project(${project} LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Sanitizer options ***********************************************************
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-fsanitize=address -O0)
    add_link_options(-fsanitize=address)
endif()

# Verbosity options ***********************************************************
set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "" FORCE)
set(CUDA_VERBOSE_BUILD OFF CACHE BOOL "" FORCE)

# Include paths ***************************************************************
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../mfem)

include_directories(/opt/homebrew/opt/openmpi/include
                    /opt/homebrew/opt/metis/include
                    /opt/homebrew/opt/hypre/include)

# Library options *************************************************************
# add_link_options(-L/opt/homebrew/lib -lfmt)

# Library source files ********************************************************
add_library(libRemhos STATIC  remhos.cpp remhos_fct.cpp 
                              remhos_ho.cpp remhos_lo.cpp 
                              remhos_mono.cpp remhos_sync.cpp 
                              remhos_tools.cpp)

# Testing options *************************************************************
enable_testing()
function(add_remhos_test name)
    add_executable(test_${name} test_${name}.cpp)
    target_link_libraries(test_${name} libRemhos -lmpi -lmfem -lhypre -lmetis -lfmt)
    target_link_directories(test_${name} PUBLIC /opt/homebrew/opt/openmpi/lib
                                          /opt/homebrew/opt/metis/lib
                                          /opt/homebrew/opt/hypre/lib
                                          /opt/homebrew/opt/fmt/lib
                                          ${CMAKE_CURRENT_SOURCE_DIR}/../mfem)  
    add_test(NAME test_${name}_1 COMMAND mpirun -n 1 test_${name})
    add_test(NAME test_${name}_3 COMMAND mpirun -n 3 test_${name})
endfunction()


add_remhos_test(101)