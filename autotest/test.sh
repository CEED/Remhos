#!/bin/bash

# execute with
# ./test.sh 2      -- cpu
# ./test.sh 2 cuda -- cuda

file="autotest/out_test.dat"

ntask=$1

if [ "$2" = "cuda" ]; then
  command="lrun -n "$((ntask))" ./remhos -no-vis -d cuda"
else
  command="mpirun -np "$((ntask))" ./remhos -no-vis"
fi

methods=( "-ho 1 -lo 2 -fct 2"     # Hennes 1
          "-ho 3 -lo 4 -fct 2"     # Hennes 2
          "-ho 2 -lo 3 -fct 2 -pa" # Arturo 1 (PA for HO and LO RD)
          "-ho 2 -lo 4 -fct 2 -pa" # Arturo 2 (PA for HO and LO RDsubcell)
        # "-ho 2 -lo 3 -fct 3"     # Manuel (penalty-based FCT)
          "-ho 3 -lo 1 -fct 1")     # Blast default remap

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
  run_line=$command" -m ./data/periodic-segment.mesh -p 0 -rs 3 -dt 0.001 -tf 1 "$method
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
  run_line=$command" -m ../mfem/data/ball-nurbs.mesh -p 1 -rs 1 -dt 0.035 -tf 3 "$method
  echo -e $run_line >> $file
  $run_line | grep -e 'mass u' -e 'value u'>> $file

done

echo -e '\n'"--- Product remap 2D (FCT)" >> $file
run_line=$command" -m ./data/inline-quad.mesh -p 14 -rs 2 -dt 0.005 -tf 0.75 -ho 3 -lo 1 -fct 1 -ps -s 1"
echo -e $run_line >> $file
$run_line | grep -e 'mass us' -e 'loss us'>> $file

echo -e '\n'"--- Product remap 2D IDP2 (ClipScale)" >> $file
run_line=$command" -m ./data/inline-quad.mesh -p 14 -rs 2 -dt 0.005 -tf 0.75 -ho 1 -lo 5 -fct 2 -ps -s 12"
echo -e $run_line >> $file
$run_line | grep -e 'mass us' -e 'loss us'>> $file

echo -e '\n'"--- Product remap 2D IDP3 (FCTProject)" >> $file
run_line=$command" -m ./data/inline-quad.mesh -p 14 -rs 2 -dt 0.005 -tf 0.75 -ho 3 -lo 5 -fct 4 -ps -s 13"
echo -e $run_line >> $file
$run_line | grep -e 'mass u' -e 'mass us'>> $file

echo -e '\n'"--- BLAST sharpening test - Pacman remap auto-dt" >> $file
run_line=$command" -m ./data/inline-quad.mesh -p 14 -rs 1 -dt -1 -tf 0.75 -ho 3 -lo 5 -fct 4 -bt 1 -dtc 1"
echo -e $run_line >> $file
$run_line | grep -e 'mass u' -e 'loss u'>> $file

echo -e '\n'"--- BLAST sharpening test - Transport balls-jacks auto-dt" >> $file
run_line=$command" -m ./data/periodic-square.mesh -p 5 -rs 3 -dt 0.01 -tf 0.8 -ho 3 -lo 5 -fct 4 -bt 1 -dtc 1"
echo -e $run_line >> $file
$run_line | grep -e 'mass u' -e 'value u'>> $file

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
