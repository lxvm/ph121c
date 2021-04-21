set terminal cairolatex pdf
set output "../tex/include/plots/2_lanczos_test.tex"
data="../fortran/data/lanczos_test.dat"
array hh[2]
hh[1] = 0.3
hh[2] = 1.7
emin=2
emax=8

hset=((2**(emax+1))-(2**emin)+(2**emax))
geom="((2**n)-(2**emin))"

set multiplot layout 1, 2 \
              title sprintf("B2: Lanczos convergence: closed b.c., $L=%i$", emax)

set xrange [1:(2**emax)]
set xlabel "$n$"
set ylabel "$\\lambda_n$"
set linetype 1
set key right bottom \
    samplen 0.5 \
    spacing 0.8
do for [h=0:1] {
    set title sprintf("h=%.1f", hh[h+1])
    plot for [n=emin:emax] data \
        every 1::(@geom + 1 + (2**emax) + h*hset )::( (2**n) + @geom + (2**emax) + h*hset ) \
        using 5:8 \
        title sprintf("$n=2^{%i}$", n) \
        with l, \
        data \
        every 1::(h*(hset))::((2**emax) - 1 + h*hset) \
        using "keff":"eigenvalue" \
        title "Dense" \
        with l
}

unset multiplot
unset output
unset terminal
reset
