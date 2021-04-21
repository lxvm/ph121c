set terminal cairolatex pdf \
    size 5, 4
set output "../tex/include/plots/3_convergence.tex"
data="../fortran/data/convergence.dat"
array hh[2]
hh[1] = 0.3
hh[2] = 1.7
array bc[2]
bc[1] = "closed"
bc[2] = "open"

set multiplot layout 2, 2 \
              title "C1: Convergence of ground state energy per site"

set xrange [8:24]
set xlabel "$L$"
set ylabel "$\\lambda_0 / L$"
set linetype 1
do for [h=0:1] {
    do for [k=0:1] {
        set title sprintf("%s b.c., h=%.1f", bc[k+1], hh[h+1])
        plot data \
            every 4::(1+k+2*h) \
            using 1:4 \
            title "" \
            with l
    }
}

unset multiplot
unset output
unset terminal
reset
