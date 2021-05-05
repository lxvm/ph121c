#!/bin/sh
# Run this script to build the whole project
# You are warned that the calculations may take
# in excess of 1 hour
# This script contains the commands I executed
# when building the project, though it may need
# some adjustments depending on the operating
# system or the location of certain directories
# or the latex installation or gnuplot.
# It should be executed from the current directory
# The only output you should expect are the runtimes
# in Fortran, which will also be saved to a file

# load intel
INTEL_START_SCRIPT=/opt/intel/oneapi/setvars.sh
. $INTEL_START_SCRIPT intel64 mod

# Move to project dir
cd $(dirname `realpath $0`)

# Fortran
cd fortran
# compile
. bin/compile_lines.txt
# run programs
ulimit -s unlimited
for i in `ls bin | grep -v compile_lines.txt`
do
  ./bin/$i | tee -a data/runtimes.txt
done
cd -

cd gnuplot
# Make plots
. processing.sh

# Make pdf
cd -
cd tex
latexmk main.tex
