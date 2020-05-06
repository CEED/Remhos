#!/bin/bash

file="out_test.dat"

ntask=$1

command="mpirun -np "$((ntask))" ../remhos -no-vis --verify-bounds"
vis_command="remhos --verify-bounds"

methods=( "-ho 1 -lo 2 -fct 2"   # Hennes 1
          "-ho 3 -lo 4 -fct 2"   # Hennes 2
        # "-ho 2 -lo 3 -fct 3"   # Manuel
          "-ho 3 -lo 1 -fct 1" ) # Blast

rm -f $file

for method in "${methods[@]}"; do

  echo -e '\n'"--- Method "$method >> $file

  echo -e '\n'"- Remap pacman nonper-struct-2D" >> $file
  echo -e $vis_command" -m ./data/inline-quad.mesh -p 14 -rs 1 -dt 0.0015 -tf 0.75 "$method >> $file
  $command -m ../data/inline-quad.mesh -p 14 -rs 1 -dt 0.0015 -tf 0.75 $method | grep -e 'Final' -e 'value'>> $file

  echo -e '\n'"- Remap bump nonper-struct-3D" >> $file
  echo -e $vis_command" -m ./data/cube01_hex.mesh -p 10 -rs 1 -o 2 -dt 0.02 -tf 0.7 "$method >> $file
  $command -m ../data/cube01_hex.mesh -p 10 -rs 1 -o 2 -dt 0.02 -tf 0.7 $method | grep -e 'Final' -e 'value'>> $file

  echo -e '\n'"- Transport per-1D" >> $file
  echo -e $vis_command" -m ./data/periodic-segment.mesh -p 0 -rs 5 -dt 0.0005 -tf 1 "$method >> $file
  $command -m ../data/periodic-segment.mesh -p 0 -rs 5 -dt 0.0005 -tf 1 $method | grep -e 'Final' -e 'value'>> $file

  echo -e '\n'"- Transport bump per-unstruct-2D" >> $file
  echo -e $vis_command" -m ./data/periodic-hexagon.mesh -p 0 -rs 2 -dt 0.005 -tf 2.5 "$method >> $file
  $command -m ../data/periodic-hexagon.mesh -p 0 -rs 2 -dt 0.005 -tf 2.5 $method | grep -e 'Final' -e 'value'>> $file

  echo -e '\n'"- Transport balls-jacks per-struct-2D" >> $file
  echo -e $vis_command" -m ./data/periodic-square.mesh -p 5 -rs 3 -dt 0.004 -tf 0.8 "$method >> $file
  $command -m ../data/periodic-square.mesh -p 5 -rs 3 -dt 0.004 -tf 0.8 $method | grep -e 'Final' -e 'value'>> $file

  echo -e '\n'"- Transport bump per-struct-3D" >> $file
  echo -e $vis_command" -m ./data/periodic-cube.mesh -p 0 -rs 1 -o 2 -dt 0.015 -tf 2 "$method >> $file
  $command -m ../data/periodic-cube.mesh -p 0 -rs 1 -o 2 -dt 0.015 -tf 2 $method | grep -e 'Final' -e 'value'>> $file

  echo -e '\n'"- Transport bump nonper-unstruct-3D" >> $file
  echo -e $vis_command" -m ../mfem/data/ball-nurbs.mesh -p 1 -rs 1 -dt 0.04 -tf 3 "$method >> $file
  $command -m ../data/ball-nurbs.mesh -p 1 -rs 1 -dt 0.04 -tf 3 $method | grep -e 'Final' -e 'value'>> $file

done

echo -e '\n'"--- Steady monolithic 2 2D" >> $file
echo -e $vis_command" -m ./data/inline-quad.mesh -p 7 -rs 3 -o 1 -dt 0.01 -tf 20 -mono 1 -si 2" >> $file
$command -m ../data/inline-quad.mesh -p 7 -rs 3 -o 1 -dt 0.01 -tf 20 -mono 1 -si 2 | grep -e 'Final' -e 'value'>> $file
echo -e '\n'"--- Steady monolithic 1 2D" >> $file
echo -e $vis_command" -m ./data/inline-quad.mesh -p 7 -rs 3 -o 1 -dt 0.01 -tf 20 -mono 1 -si 1" >> $file
$command -m ../data/inline-quad.mesh -p 6 -rs 2 -o 1 -dt 0.01 -tf 20 -mono 1 -si 1 | grep -e 'Final' -e 'value'>> $file

exit 0
