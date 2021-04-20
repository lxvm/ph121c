set terminal cairolatex pdf
set output "../tex/include/plots/1_dense_ed_gap.tex"
data="data/dense_ed.dat"
Nsim=9

set title "A3: Difference in lowest eigenvalues"
set xlabel "$h$"
set ylabel "$\\Delta E_{0}$"
set logscale x 10
set linetype 1 pt 3
set key right bottom
plot for [n=0:3] data \
    every 1::(n*Nsim)::((n+1)*Nsim-1) \
    using "h":"difference" \
    title sprintf("$\\; L=%d$", 2*n + 8 )  \
    with lp

unset output
reset
