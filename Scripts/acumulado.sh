#!/bin/bash

for mes in {1..12}
do
    python MAPA_REG.py ENACTS2_prcp.nc \
        --date 2025,$mes,1 \
        --region 3 \
        --frequency daily 
done
