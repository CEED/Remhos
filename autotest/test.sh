#!/bin/bash

# execute with
# ./test.sh 2      -- cpu
# ./test.sh 2 cuda -- cuda

file="autotest/out_test.dat"

ntask=$1

if [ "$2" = "cuda" ]; then
  command="lrun -n "$((ntask))" ./remhos -no-vis --verify-bounds -d cuda"
else
  command="mpirun -np "$((ntask))" ./remhos -no-vis --verify-bounds"
fi

methods=( "-ho 1 -lo 2 -fct 2"   # Hennes 1
          "-ho 3 -lo 4 -fct 2"   # Hennes 2
        # "-ho 2 -lo 3 -fct 3"   # Manuel
          "-ho 3 -lo 1 -fct 1" ) # Blast

cd ..
rm -f $file

for method in "${methods[@]}"; do

  echo -e '\n'"--- Method "$method >> $file

  echo -e '\n'"- Remap pacman nonper-struct-2D" >> $file
  run_line=$command" -m ./data/inline-quad.mesh -p 14 -rs 1 -dt 0.0015 -tf 0.75 "$method
  echo -e $run_line >> $file
  $run_line | grep -e 'mass u' -e 'value u'>> $file

  echo -e '\n'"- Remap bump nonper-struct-3D" >> $file
  run_line=$command" -m ./data/cube01_hex.mesh -p 10 -rs 1 -o 2 -dt 0.02 -tf 0.7 "$method
  echo -e $run_line >> $file
  $run_line | grep -e 'mass u' -e 'value u'>> $file

  echo -e '\n'"- Transport per-1D" >> $file
  run_line=$command" -m ./data/periodic-segment.mesh -p 0 -rs 5 -dt 0.0005 -tf 1 "$method
  echo -e $run_line >> $file
  $run_line | grep -e 'mass u' -e 'value u'>> $file

  echo -e '\n'"- Transport bump per-unstruct-2D" >> $file
  run_line=$command" -m ./data/periodic-hexagon.mesh -p 0 -rs 2 -dt 0.005 -tf 2.5 "$method
  echo -e $run_line >> $file
  $run_line | grep -e 'mass u' -e 'value u'>> $file

  echo -e '\n'"- Transport balls-jacks per-struct-2D" >> $file
  run_line=$command" -m ./data/periodic-square.mesh -p 5 -rs 3 -dt 0.004 -tf 0.8 "$method
  echo -e $run_line >> $file
  $run_line | grep -e 'mass u' -e 'value u'>> $file

  echo -e '\n'"- Transport bump per-struct-3D" >> $file
  run_line=$command" -m ./data/periodic-cube.mesh -p 0 -rs 1 -o 2 -dt 0.015 -tf 2 "$method
  echo -e $run_line >> $file
  $run_line | grep -e 'mass u' -e 'value u'>> $file

  echo -e '\n'"- Transport bump nonper-unstruct-3D" >> $file
  run_line=$command" -m ../mfem/data/ball-nurbs.mesh -p 1 -rs 1 -dt 0.04 -tf 3 "$method
  echo -e $run_line >> $file
  $run_line | grep -e 'mass u' -e 'value u'>> $file

done

echo -e '\n'"--- Product remap 2D" >> $file
run_line=$command" -m ./data/inline-quad.mesh -p 14 -rs 2 -dt 0.005 -tf 0.75 -ho 3 -lo 1 -fct 1 -ps -s 1"
echo -e $run_line >> $file
$run_line | grep -e 'mass us' -e 'loss us'>> $file

echo -e '\n'"--- Steady monolithic 2 2D" >> $file
run_line=$command" -m ./data/inline-quad.mesh -p 7 -rs 3 -o 1 -dt 0.01 -tf 20 -mono 1 -si 2"
echo -e $run_line >> $file
$run_line | grep -e 'mass u' -e 'value u'>> $file
  
echo -e '\n'"--- Steady monolithic 1 2D" >> $file
run_line=$command" -m ./data/inline-quad.mesh -p 6 -rs 2 -o 1 -dt 0.01 -tf 20 -mono 1 -si 1"
echo -e $run_line >> $file
$run_line | grep -e 'mass u' -e 'value u'>> $file

cd autotest

exit 0
