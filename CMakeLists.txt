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
    set(file test_${name}.cpp)
    set(target test_${name})
    list(LENGTH ARGN ARGN_COUNT)
    if(ARGN_COUNT GREATER 0) 
        list(GET ARGN 0 extra)
        string(APPEND name "_" ${extra})
        string(APPEND target "_" ${extra})
    endif()
    message(STATUS "Adding target: ${target}")
    add_executable(${target} ${file})
    target_link_libraries(${target} libRemhos -lmpi -lmfem -lhypre -lmetis -lfmt)
    target_link_directories(${target} PUBLIC /opt/homebrew/opt/openmpi/lib
                                          /opt/homebrew/opt/metis/lib
                                          /opt/homebrew/opt/hypre/lib
                                          /opt/homebrew/opt/fmt/lib
                                          ${CMAKE_CURRENT_SOURCE_DIR}/../mfem)  
    message(STATUS "Adding test: ${target}_n1")
    add_test(NAME ${target}_n1  COMMAND mpirun -n 1 ${target} ${extra})
    # add_test(NAME dperf_${name}_1 COMMAND dperf test_${name})
    message(STATUS "Adding test: ${target}_n3")
    add_test(NAME ${target}_n3 COMMAND mpirun -n 3 ${target})
endfunction()


add_remhos_test(101)
# add_remhos_test(101 0)