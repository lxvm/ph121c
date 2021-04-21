set terminal cairolatex pdf
set output "../tex/include/plots/3_trick.tex"
data="../fortran/data/trick.dat"
array hh[2]
hh[1] = 0.3
hh[2] = 1.7

set multiplot layout 1, 2 \
              title "C2: Additional ground state energy per center site, open b.c."

set xrange [10:24]
set xlabel "$L$"
set ylabel "$\\lambda_0 / L$"
set linetype 1
do for [h=0:1] {
    set title sprintf("h=%.1f", hh[h+1])
    plot data \
        every 2::h \
        using 2:5 \
        title "" \
        with l
}

unset multiplot
unset output
unset terminal
reset
