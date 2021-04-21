set terminal cairolatex pdf \
    size 5, 8
set output "../tex/include/plots/2_sparse_ed_cmp_all.tex"
data="../fortran/data/sparse_ed.dat"
array bc[2]
bc[1] = "open"
bc[2] = "closed"
array hh[2]
hh[1] = 0.3
hh[2] = 1.7
array LL[2]
LL[1] = 8
LL[2] = 10
array methods[2]
methods[1] = "Lanczos"
methods[2] = "Dense"


set multiplot layout 4, 2 \
              title "B3: Comparison of numerical spectra"

set xlabel "$n$"
set ylabel "$\\lambda_n$"
set linetype 1
set key right bottom

do for [L=0:1] {
    do for [h=0:1] {
        do for [k=0:1] {
            set title sprintf("%s b.c., L=%i, h=%.1f", bc[k+1], LL[L+1], hh[h+1])
            set xrange [1:(2**LL[L+1])]
            plot for [n=1:2] data \
                index 0 \
                every 1::((2*h+k)*(2**LL[L+1])+L*1024)::((2*h+k+1)*(2**LL[L+1])+L*1024 - 1) \
                using "k":4+n \
                title methods[n] \
                with l
        }
    }
}

unset multiplot
unset output
unset terminal
reset
