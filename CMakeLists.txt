cmake_minimum_required(VERSION 3.18)

set(project remhos)
project(${project} LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
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
# set(CUDA_VERBOSE_BUILD OFF CACHE BOOL "" FORCE)

# Include paths ***************************************************************
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../mfem)
if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
include_directories(/usr/include/hypre /usr/include)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
include_directories(/opt/homebrew/opt/fmt/include
                    /opt/homebrew/opt/openmpi/include
                    /opt/homebrew/opt/metis/include
                    /opt/homebrew/opt/hypre/include)
add_compile_definitions(MFEM_USE_CMAKE_TESTS)
else()
    message(FATAL_ERROR "Unsupported system")
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
add_library(Remhos STATIC remhos.cpp remhos_fct.cpp 
                          remhos_ho.cpp remhos_lo.cpp 
                          remhos_mono.cpp remhos_sync.cpp 
                          remhos_tools.cpp)

# Testing options *************************************************************
enable_testing()
function(add_remhos_test name)
    set(file ${name}.cpp)
    set(target ${name})
    list(LENGTH ARGN ARGN_COUNT)
    if(ARGN_COUNT GREATER 0) 
        list(GET ARGN 0 extra)
        string(APPEND name "_" ${extra})
        string(APPEND target "_" ${extra})
    endif()
    message(STATUS "Adding target: ${target}")
    add_executable(${target} ${file})
    add_dependencies(${target} copy_mesh_files)
    target_link_libraries(${target} Remhos -lmpi -lmfem -lhypre -lmetis -lfmt)
if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    target_link_directories(${target} PUBLIC /usr/lib/x86_64-linux-gnu) 
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    target_link_directories(${target} PUBLIC /opt/homebrew/opt/openmpi/lib
                                          /opt/homebrew/opt/metis/lib
                                          /opt/homebrew/opt/hypre/lib
                                          /opt/homebrew/opt/fmt/lib) 
endif()
    target_link_directories(${target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../mfem)
    message(STATUS "Adding test: ${target}_n1")
    add_test(NAME ${target}_n1  COMMAND mpirun -n 1 ${target} ${extra})
    message(STATUS "Adding test: ${target}_n3")
    add_test(NAME ${target}_n3 COMMAND mpirun -n 3 ${target})
endfunction()

add_remhos_test(remhos_main)
add_remhos_test(remhos_tests)