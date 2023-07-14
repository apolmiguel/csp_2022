#!/bin/bash

#do for each config
cfg_num=($(seq 1 1 20))
for i in ${cfg_num[@]}; do

file="printout_config_$i"

while read line; do 
echo -e "$line\t" >> en_ce_test.dat
done <$file

#converts single column to four columns
xargs -n4 < en_ce_test.dat >> config_energies_ce.dat

rm en_ce_test.dat

done
