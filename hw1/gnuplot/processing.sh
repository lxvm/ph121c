#!/bin/sh

tmp=`mktemp`

# dis-interleave open/closed data from problem 1
data="../fortran/data/dense_ed.dat"
out="data/dense_ed.dat"
cat $data | sed -n '/ o /p' > $out
cat $data | sed -n '/ c /p' | awk '{print $4}'| paste $out - | awk '{$3=""; print $0}' > $tmp
awk '{$5=$4-$3; print $0}' $tmp > $out
sed -i '1 i\L h open closed difference' $out

for i in `ls | grep .gp`
do
    gnuplot $i
done
