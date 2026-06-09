#!/bin/bash

for mes in {1..12}
do
    python MAPA_REG.py prediccionesNextGen.nc \
        --date 2025,$mes,1 \
        --region 3 \
        --frequency monthly \
        --pronostico \
        --tipo-pronostico NextGen
done
