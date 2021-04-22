set terminal cairolatex pdf
set output "../tex/include/plots/6_gap.tex"
data="data/gap.dat"
array sector[2]
sector[1] = '+'
sector[2] = '-'

set xlabel "$L$"
set ylabel "$\\lambda_0/L$"
set linetype 1 pt 3
set title "F1: Ground state sector energy gap, closed b.c., $h=0.3$"

plot for [k=1:2] data \
    every 2::k-1 \
    using "L":"eigenvalue" \
    title sprintf("$%s$", sector[k]) \
    with lp

unset output
unset terminal
reset
