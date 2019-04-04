## Building

- MFEM - serial build of the matrix-free-FCT branch
  <br> https://github.com/mfem/mfem.

## Verification of Results

To make sure the results are correct, we verify the final mass (`mass`) and
maximum value (`max`) for the runs listed below:

1. `./remhos -m ./data/periodic-hexagon.mesh -p 0 -r 2 -dt 0.01 -tf 10 -mt 2`
2. `./remhos -m ./data/periodic-hexagon.mesh -p 0 -r 2 -dt 0.01 -tf 10 -mt 4`
3. `./remhos -m ./data/disc-nurbs.mesh -p 1 -r 3 -dt 0.005 -tf 3 -mt 2`
4. `./remhos -m ./data/disc-nurbs.mesh -p 1 -r 3 -dt 0.005 -tf 3 -mt 4`
5. `./remhos -m ./data/periodic-square.mesh -p 5 -r 4 -dt 0.002 -o 2 -tf 0.8 -mt 2`
6. `./remhos -m ./data/periodic-square.mesh -p 5 -r 4 -dt 0.002 -o 2 -tf 0.8 -mt 4`
7. `./remhos -m ./data/periodic-square.mesh -p 14 -r 3 -dt 0.005 -tf 0.5 -mt 4`

| `run` | `mass` | `max` |
| ----- | ------ | ----- |
|  1. | 0.3888354875 | 0.9304836587 |
|  2. | 0.3888354875 | 0.9435930561 |
|  3. | 3.598223766  | 0.9995717563 |
|  4. | 3.598223766  | 0.9995717563 |
|  5. | 0.1631944444 | 0.9916942643 |
|  6. | 0.1631944444 | 0.9501635831 |
|  7. | 0.3725812745 | 0.9999992137 |

An implementation is considered valid if the computed values are all within
round-off distance from the above reference values.
