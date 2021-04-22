set terminal cairolatex pdf
set output "../tex/include/plots/4_transition.tex"
data="../fortran/data/transition.dat"
array hh[2]
hh[1] = 0.3
hh[2] = 1.7

set multiplot layout 1, 2 \
    title "D1: Ground state excitation gap, $L=24$ , closed b.c."
set xlabel "$h$"
set ylabel "$\\lambda$"
set linetype 1 pt 3
plot for [k=1:4] data \
    every 4::(k-1) \
    using 2:5 \
    title sprintf("$\\lambda_%i$", k-1) \
    with lp

set ylabel "$\\lambda_0 - \\lambda_1$"
plot "data/transition.dat" \
    using 1:2 \
    title "" \
    with lp
unset multiplot
unset output
unset terminal
reset
