set terminal cairolatex pdf
set output "../tex/include/plots/2_sparse_ed_cmp.tex"
data="../fortran/data/lanczos_test.dat"

set title "B1: Comparison of numerical spectra for open b.c., $L=8$, $h=0.3$"
set xlabel "$n$"
set ylabel "$\\lambda_n$"
set xrange [1:(2**8)]
set linetype 1 pt 3
set key left top
plot \
    data \
        index 0 \
        every 1::0::255 \
        using 4:8 \
        title "Dense"  \
        with l, \
    data \
        index 0 \
        every 1::509::764 \
        using 4:8 \
        title "Lanczos"  \
        with l
        
unset output
unset terminal
reset
