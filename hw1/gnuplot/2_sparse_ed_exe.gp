set terminal cairolatex pdf \
    size 5, 6
set output "../tex/include/plots/2_sparse_ed_exe.tex"
data="../fortran/data/sparse_ed.dat"

array LL[5]
LL[1]=12
LL[2]=14
LL[3]=16
LL[4]=18
LL[5]=20
array bc[2]
bc[1] = "open"
bc[2] = "closed"
array hh[2]
hh[1] = 0.3
hh[2] = 1.7

set multiplot layout 2, 2\
    title "B4: Lanczos spectra with 32 eigenvalues"

set xlabel "$n$"
set ylabel "$\\lambda_n$"
set xrange [1:32]
set linetype 1
set key right bottom \
    samplen 0.5

do for [h=0:1] {
    do for [k=0:1] {
        set title sprintf("%s b.c., h=%.1f", bc[k+1], hh[h+1])
        plot for [n=1:5] data \
            index 1 \
            every 1::(32*4*(n-1) + 32*2*h + 32*k)::(31 + 32*4*(n-1) + 32*2*h + 32*k) \
            using "k":5 \
            title sprintf("$L=%i$", LL[n]) \
            with l
    }
}

unset multiplot
unset output
unset terminal
reset

