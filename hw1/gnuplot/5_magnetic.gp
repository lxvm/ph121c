set terminal cairolatex pdf
set output "../tex/include/plots/5_magnetic.tex"
data="data/magnetic.dat"
array hh[3]
hh[1] = 0.3
hh[2] = 1.0
hh[3] = 1.7


set xlabel "$L$"
set ylabel "$\\ev{(M/L)^2}$"
set linetype 1 pt 3
set title "E1: Long-range magnetic order, $L=24$ , closed b.c."

plot for [k=1:3] data \
    every 3::k-1 \
    using "L":"expectation" \
    title sprintf("$h=%.1f$", hh[k])\
    with lp

unset output
unset terminal
reset
