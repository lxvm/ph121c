set terminal cairolatex pdf
set output "../tex/include/plots/4_fidelity.tex"
data="../fortran/data/fidelity.dat"
array hh[2]
hh[1] = 0.3
hh[2] = 1.7


set xlabel "$h$"
set ylabel "$\\abs{\\ip{\\psi_0(h)}{\\psi_0(h+dh)}}$"
set linetype 1 pt 3
set title "D2: Fidelity, dh = 0.01, $L=24$ , closed b.c."
plot data \
    using "h":"fidelity" \
    with lp

unset output
unset terminal
reset
