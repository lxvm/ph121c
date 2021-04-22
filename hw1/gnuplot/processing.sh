#!/bin/sh

tmp=`mktemp`

# dis-interleave open/closed data from problem 1
data="../fortran/data/dense_ed.dat"
out="data/dense_ed.dat"
cat $data | sed -n '/ o /p' > $out
cat $data | sed -n '/ c /p' | awk '{print $4}'| paste $out - | awk '{$3=""; print $0}' > $tmp
awk '{$5=$4-$3; print $0}' $tmp > $out
sed -i '1 i\L h open closed difference' $out

data="../fortran/data/transition.dat"
out="data/transition.dat"
grep -e " 1 " $data | cut -d" " -f 14 | paste <(grep -e " 2 " $data) - | awk '{print $2, $5- $6}' > $out
sed -i '1 i\h Egap' $out

data="../fortran/data/ordering.dat"
out="data/magnetic.dat"
grep -e " m " $data > $out
sed -i "1 i\L j bc m expectation" $out

data="../fortran/data/symmetry.dat"
out="data/gap.dat"
grep -e "3  c 1" $data | awk -v CONVFMT=%.15g '{gsub($6, $6/$1); print $0}' > $out
sed -i "1 i\L h bc k s eigenvalue" $out

for i in `ls | grep .gp`
do
    gnuplot $i
done
