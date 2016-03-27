#!/bin/bash
#$ -S /bin/bash
#$ -q low.q
#$ -N R_test
#$ -cwd
#$ -j Y
#$ -V
#$ -m be
#$ -M haolyu@berkeley.edu

python rf.py

