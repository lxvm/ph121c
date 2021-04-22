set terminal cairolatex pdf \
    size 5, 8
set output "../tex/include/plots/6_symmetry.tex"
data="../fortran/data/symmetry.dat"
array hh[3]
hh[1] = 0.3
hh[2] = 1.0
hh[3] = 1.7
array LL[4]
LL[1] = 20
LL[2] = 22
LL[3] = 24
LL[4] = 25
array sector[2]
sector[1] = '+'
sector[2] = '-'

set multiplot layout 4, 2 \
    title "F2: Lowest eigenvalues by symmetry sector, closed b.c."

set xlabel "$h$"
set ylabel "$\\lambda$"
set linetype 1 pt 3

# Do values of L >= 20
do for [L=0:3] {
    do for [s=0:1] {
        set title sprintf("$%s$ sector, $L=%i$", sector[s+1], LL[L+1])
        plot for [k=0:3] data \
            skip 217 \
            every 8::k+s*4+24*L::k+s*4+24*L+23 \
            using 2:6 \
            title sprintf("$\\lambda_%i$", k+1)\
            with lp
    }
}

unset multiplot
unset output
unset terminal
reset
