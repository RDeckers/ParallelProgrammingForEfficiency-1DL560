set term term_type
set output output_file
set logscale x 2
set logscale y 10
set key outside;
set key center top;
set format x "2^{%L}"
set format y "10^{%L}"
set ylabel "runtime/element (ns)"
set xlabel "Elements"
set grid;
plot\
 'data/memory_test.dat' i 0 using 1:(($4+$2)/$1) w lp title 'Transfer blocking',\
 'data/memory_test.dat' i 0 using 1:($3/$1) w lp title 'Kernel blocking',\
 'data/memory_test.dat' i 0 using 1:($5/$1) w lp title 'total blocking',\
 'data/memory_test.dat' i 1 using 1:(($4+$2)/$1) w lp title 'Transfer clFinish',\
 'data/memory_test.dat' i 1 using 1:($3/$1) w lp title 'Kernel clFinish',\
 'data/memory_test.dat' i 1 using 1:($5/$1) w lp title 'total clFinish'
set output
