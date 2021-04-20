set terminal cairolatex pdf
set output "../tex/include/plots/1_dense_ed.tex"
data="data/dense_ed.dat"
Nsim=9

set multiplot layout 1, 2 rowsfirst

set title "A1: Lowest eigenvalue of open system"
set xlabel "$h$"
set ylabel "$E_{0}$"
set logscale x 10
set linetype 1 pt 3
set key left bottom
plot for [n=0:3] data \
    every 1::(n*Nsim)::((n+1)*Nsim-1) \
    using "h":"open" \
    title sprintf("$\\; L=%d$", 2*n + 8 )  \
    with lp

set title "A2: Lowest eigenvalue of closed system"
set xlabel "$h$"
set ylabel "$E_{0}$"
set logscale x 10
set linetype 1 pt 3
set key left bottom
plot for [n=0:3] data \
    every 1::(n*Nsim)::((n+1)*Nsim-1) \
    using "h":"closed" \
    title sprintf("$\\; L=%d$", 2*n + 8 )  \
    with lp

unset multiplot
unset output
reset
