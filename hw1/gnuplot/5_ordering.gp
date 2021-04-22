set terminal cairolatex pdf \
    size 5, 8
set output "../tex/include/plots/5_ordering.tex"
data="../fortran/data/ordering.dat"
array hh[3]
hh[1] = 0.3
hh[2] = 1.0
hh[3] = 1.7
emin  = 12
emax  = 24
estp  = 2
array Lstp[7]
Lstp[1] = 0
Lstp[2] = (12+1)*3
Lstp[3] = Lstp[2] + (14+1)*3 
Lstp[4] = Lstp[3] + (16+1)*3 
Lstp[5] = Lstp[4] + (18+1)*3 
Lstp[6] = Lstp[5] + (20+1)*3 
Lstp[7] = Lstp[6] + (22+1)*3 

set multiplot layout 4, 2 \
    title "E2: Pairwise correlation, closed b.c."

set xlabel "$r$"
set ylabel "$C^{zz}(r)$"
set linetype 1 pt 3

do for [L=emin:emax:estp] {
    i = (L -10)/2
    set title sprintf("$L=%i$", L)
    plot for [k=1:3] data \
        every ::(k-1)*(L+1)+Lstp[i]::(k-1)*(L+1)+Lstp[i]+L \
        using 4:5 \
        title sprintf("$h=%.1f$", hh[k])\
        with lp
}

unset multiplot
unset output
unset terminal
reset
