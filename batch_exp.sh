#!/bin/bash
read -r -a array <<<"$1"

for index in "${!array[@]}"
do
    echo "Seed=${array[index]}"
    ./reproduce_proposed_elmo_train.sh ${array[index]} true
    ./reproduce_proposed_elmo_train.sh ${array[index]} false 
done
