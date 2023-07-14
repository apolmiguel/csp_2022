#!/bin/bash

#printf "ecutwfc\tenergy\n" >> energies_rhocut.dat
# list of distances. distance must have been generated
config=($(seq 20 1 21))

for r in ${config[@]}
do

energy=$( awk '/!/ {print $5}' ./config${r}/CuAu.scf.out)

printf "$((r-1))\t$energy\n" >> config_energies_dft.dat

done
