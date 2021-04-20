set terminal cairolatex pdf
set output "../tex/include/plots/2_sparse_ed_cmp.tex"
data="../fortran/data/sparse_ed.dat"
Nsim=9

set title "B1: Comparison of numerical spectra"
set xlabel "$n$"
set ylabel "$\\lambda_n$"
set linetype 1 pt 3
set key left top
plot \
    data \
        index 0 \
        every 1::0::255 \
        using "k":5 \
        title "Lanczos"  \
        with l, \
    data \
        index 0 \
        every 1::0::255 \
        using "k":6 \
        title "Dense"  \
        with l
        
unset output
reset
