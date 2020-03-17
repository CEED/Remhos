              ____                 __
             / __ \___  ____ ___  / /_  ____  _____
            / /_/ / _ \/ __ `__ \/ __ \/ __ \/ ___/
           / _, _/  __/ / / / / / / / / /_/ (__  )
          /_/ |_|\___/_/ /_/ /_/_/ /_/\____/____/

                  High-order Remap Miniapp


## Purpose

**Remhos** (REMap High-Order Solver) is a miniapp that solves the pure
advection equations that are used to perform discontinuous field interpolation
(remap) as part of the Eulerian phase in Arbitrary-Lagrangian Eulerian (ALE)
simulations.

Remhos is based on the discretization method described in the following article:

...

The Remhos miniapp is part of the [CEED software suite](http://ceed.exascaleproject.org/software),
a collection of software benchmarks, miniapps, libraries and APIs for
efficient exascale discretizations based on high-order finite element
and spectral element methods. See http://github.com/ceed for more
information and source code availability.

The CEED research is supported by the [Exascale Computing Project](https://exascaleproject.org/exascale-computing-project)
(17-SC-20-SC), a collaborative effort of two U.S. Department of Energy
organizations (Office of Science and the National Nuclear Security
Administration) responsible for the planning and preparation of a
[capable exascale ecosystem](https://exascaleproject.org/what-is-exascale),
including software, applications, hardware, advanced system engineering and early
testbed platforms, in support of the nation’s exascale computing imperative.

## Characteristics

The problem that Remhos is solving is formulated as a big (block) system of
ordinary differential equations (ODEs) for the ...

## Code Structure

- The file `remhos.cpp` contains the main driver with the time integration loop
  starting around line ...

## Building

Remhos has the following external dependencies:

- MFEM - parallel build of the matrix-free-FCT branch
  <br> https://github.com/mfem/mfem.

## Running

#### Test problem 1
...

#### Test problem 2
...

## Verification of Results

To make sure the results are correct, we verify the final mass (`mass`) and
maximum value (`max`) for the runs listed below:

1.  `mpirun -np 8 remhos -m ./data/periodic-hexagon.mesh -p 0 -rs 2 -dt 0.005 -tf 10 -ho 1 -lo 1 -fct 2`
2.  `mpirun -np 8 remhos -m ./data/periodic-hexagon.mesh -p 0 -rs 2 -dt 0.005 -tf 10 -ho 1 -lo 2 -fct 2`
3.  `mpirun -np 8 remhos -m ./data/disc-nurbs.mesh -p 1 -rs 3 -dt 0.005 -tf 3 -ho 1 -lo 1 -fct 2`
4.  `mpirun -np 8 remhos -m ./data/disc-nurbs.mesh -p 1 -rs 3 -dt 0.005 -tf 3 -ho 1 -lo 2 -fct 2`
5.  `mpirun -np 8 remhos -m ./data/periodic-square.mesh -p 5 -rs 3 -dt 0.005 -tf 0.8 -ho 1 -lo 1 -fct 2`
6.  `mpirun -np 8 remhos -m ./data/periodic-square.mesh -p 5 -rs 3 -dt 0.002 -tf 0.8 -ho 1 -lo 2 -fct 2`
7.  `mpirun -np 8 remhos -m ./data/periodic-cube.mesh -p 0 -rs 1 -o 2 -dt 0.014 -tf 8 -ho 1 -lo 2 -fct 2`
8.  `mpirun -np 8 remhos -m ../mfem/data/ball-nurbs.mesh -p 1 -rs 1 -dt 0.02 -tf 3 -ho 1 -lo 2 -fct 2`
9.  `mpirun -np 8 remhos -m ./data/square01_quad.mesh -p 14 -rs 2 -dt 0.001 -tf 0.75 -ho 1 -lo 2 -fct 2`
10. `mpirun -np 8 remhos -m ./data/cube01_hex.mesh -p 10 -rs 1 -o 2 -dt 0.02 -tf 0.8 -ho 1 -lo 2 -fct 2`
11. `mpirun -np 8 remhos -m ./data/inline-quad.mesh -p 7 -rs 3 -o 1 -dt 0.01 -tf 20 -mono 1 -si 2`
12. `mpirun -np 8 remhos -m ./data/inline-quad.mesh -p 6 -rs 2 -o 1 -dt 0.01 -tf 20 -mono 1 -si 1`

| `run` | `mass` | `max` |
| ----- | ------ | ----- |
|  1. | 0.3888354875 | 0.9333315791 |
|  2. | 0.3888354875 | 0.9449786245 |
|  3. | 3.5982222    | 0.9995717563 |
|  4. | 3.5982222    | 0.9995717563 |
|  5. | 0.1623263888 | 0.7676354393 |
|  6. | 0.1623263888 | 0.7469836332 |
|  7. | 0.9607429525 | 0.767823337  |
|  8. | 0.8087330861 | 0.9999889315 |
|  9. | 0.08479546727| 0.8378749205 |
| 10. | 0.1197297047 | 0.9985405673 |
| 11. | 0.1570667907 | 0.9987771164 |
| 12. | 0.3182739921 | 1            |

An implementation is considered valid if the computed values are all within
round-off distance from the above reference values.

## Performance Timing and FOM
...

## Versions
...


## Contact

You can reach the Remhos team by emailing remhos@llnl.gov or by leaving a
comment in the [issue tracker](https://github.com/CEED/Remhos/issues).

## Copyright

The following copyright applies to each file in the CEED software suite,
unless otherwise stated in the file:

> Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the
> Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights reserved.

See files LICENSE and NOTICE for details.
