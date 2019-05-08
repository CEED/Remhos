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

- MFEM - serial build of the matrix-free-FCT branch
  <br> https://github.com/mfem/mfem.

## Running

#### Test problem 1
...

#### Test problem 2
...

## Verification of Results

To make sure the results are correct, we verify the final mass (`mass`) and
maximum value (`max`) for the runs listed below:

1.  `./remhos -m ./data/periodic-hexagon.mesh -p 0 -rs 2 -dt 0.01 -tf 10 -mt 2`
2.  `./remhos -m ./data/periodic-hexagon.mesh -p 0 -rs 2 -dt 0.01 -tf 10 -mt 4`
3.  `./remhos -m ./data/disc-nurbs.mesh -p 1 -rs 3 -dt 0.005 -tf 3 -mt 2`
4.  `./remhos -m ./data/disc-nurbs.mesh -p 1 -rs 3 -dt 0.005 -tf 3 -mt 4`
5.  `./remhos -m ./data/periodic-square.mesh -p 5 -rs 4 -dt 0.002 -o 2 -tf 0.8 -mt 2`
6.  `./remhos -m ./data/periodic-square.mesh -p 5 -rs 4 -dt 0.002 -o 2 -tf 0.8 -mt 4`
7.  `./remhos -m ./data/periodic-cube.mesh -p 0 -rs 1 -o 2 -dt 0.02 -tf 8 -mt 4`
8.  `./remhos -m ../mfem/data/ball-nurbs.mesh -p 1 -rs 1 -dt 0.01 -tf 3 -mt 4`
9.  `./remhos -m ./data/periodic-square.mesh -p 14 -rs 3 -dt 0.005 -tf 0.5 -mt 4`
10. `./remhos -m ./data/periodic-cube.mesh -p 10 -rs 1 -o 2 -dt 0.02 -tf 0.5 -mt 4`

| `run` | `mass` | `max` |
| ----- | ------ | ----- |
|  1. | 0.3888354875 | 0.9304836587 |
|  2. | 0.3888354875 | 0.941987734  |
|  3. | 3.5982222    | 0.9995717563 |
|  4. | 3.5982222    | 0.9995717563 |
|  5. | 0.1631944444 | 0.9916942643 |
|  6. | 0.1631944444 | 0.9501635831 |
|  7. | 0.9607429525 | 0.763596485  |
|  8. | 0.8087489307 | 0.9999889315 |
|  9. | 0.3725812939 | 0.9999996397 |
| 10. | 0.9607434514 | 0.9999600963 |

An implementation is considered valid if the computed values are all within
round-off distance from the above reference values.

## Performance Timing and FOM
...

## Versions
...


## Contact

You can reach the Laghos team by emailing remhos@llnl.gov or by leaving a
comment in the [issue tracker](https://github.com/CEED/Remhos/issues).

## Copyright

The following copyright applies to each file in the CEED software suite,
unless otherwise stated in the file:

> Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the
> Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights reserved.

See files LICENSE and NOTICE for details.
