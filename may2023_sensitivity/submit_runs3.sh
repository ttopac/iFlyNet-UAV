#!/bin/bash

python -u FNONet_dynVal_SGLETEincl.py |& tee -a out3.txt
python -u FNONet_dynVal_liftdrag_SGLETEincl.py |& tee -a out3.txt