#!/bin/sh

Test=("$1")
Filename=("$2")
python finaltest.py $Test
python bleu_eval.py $Filename