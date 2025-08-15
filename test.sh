mpirun -np 4 remhos -m ./data/inline-quad.mesh -p 14 -rs 4 -dt -1.0 -tf 0.75 -mono 3 -opt 2 -mi 1000
mpirun -np 4 remhos -m ./data/inline-quad.mesh -p 13 -rs 2 -dt -1.0 -tf 0.75 -mono 3 -proj -opt 2 -mi 1000
mpirun -np 8 remhos -m ./data/cube01_hex.mesh -p 10 -rs 3 -o 2 -dt -1.0 -tf 0.5 -mono 3 -opt 2 -mi 1000
mpirun -np 4 remhos -m ./data/inline-quad.mesh -p 18 -rs 4 -dt -1.0 -tf 0.75 -mono 4 -opt 2 -mi 1000
mpirun -np 4 remhos -m ./data/inline-quad.mesh -p 14 -rs 4 -dt -1.0 -tf 0.75 -mono 4 -opt 2 -mi 1000
mpirun -np 8 remhos -m ./data/cube01_hex.mesh -p 10 -rs 3 -o 2 -dt -1.0 -tf 0.5 -mono 4 -opt 2 -mi 1000
mpirun -np 4 remhos -m ./data/inline-quad.mesh -p 18 -rs 4 -dt -1.0 -tf 0.75 -mono 5 -opt 2 -mi 1000
mpirun -np 4 remhos -m ./data/inline-quad.mesh -p 38 -rs 4 -dt -1.0 -tf 0.75 -mono 5 -opt 2 -mi 1000
mpirun -np 4 remhos -m ./data/inline-quad.mesh -p 14 -rs 4 -dt -1.0 -tf 0.75 -mono 5 -opt 2 -mi 1000
mpirun -np 4 remhos -m ./data/inline-quad.mesh -p 34 -rs 4 -dt -1.0 -tf 0.75 -mono 5 -opt 2 -mi 1000
mpirun -np 4 remhos -m ./data/inline-quad.mesh -p 18 -rs 4 -dt -1.0 -tf 0.75 -mono 6 -opt 2 -mi 1000
mpirun -np 4 remhos -m ./data/inline-quad.mesh -p 38 -rs 4 -dt -1.0 -tf 0.75 -mono 6 -opt 2 -mi 1000
mpirun -np 4 remhos -m ./data/inline-quad.mesh -p 14 -rs 4 -dt -1.0 -tf 0.75 -mono 6 -opt 2 -mi 1000
mpirun -np 4 remhos -m ./data/inline-quad.mesh -p 34 -rs 4 -dt -1.0 -tf 0.75 -mono 6 -opt 2 -mi 1000
