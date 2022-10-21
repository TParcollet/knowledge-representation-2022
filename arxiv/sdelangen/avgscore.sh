#!/bin/bash

for f in $@; do
    awk '{s+=$1}END{print s/NR}' RS=" " $f
done