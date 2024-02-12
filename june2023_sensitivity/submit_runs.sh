#!/bin/bash

python -u FNONet_dynVal_halfdata.py |& tee -a out.txt
python -u FNONet_dynVal_liftdrag_halfdata.py |& tee -a out.txt
python -u FNONet_dynVal_medfilt.py |& tee -a out.txt
python -u FNONet_dynVal_liftdrag_medfilt.py |& tee -a out.txt
python -u FNONet_dynVal_lpf.py |& tee -a out.txt
python -u FNONet_dynVal_liftdrag_lpf.py |& tee -a out.txt