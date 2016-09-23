set term term_type
set output output_file
set logscale xy 2
set key outside;
set key center top;
set format x "2^{%L}"
set format y "2^{%L}"
set ylabel "runtime (ns)"
set xlabel "Image dimensions (width & height)"
set grid;
plot\
 'data/scaling.dat' using 1:4 w lp title 'Min. Upload',\
 'data/scaling.dat' using 1:5 w lp title 'Min. Kernel',\
 'data/scaling.dat' using 1:6 w lp title 'Min. Readback',\
 'data/scaling.dat' using 1:7 w lp title 'Min. Upload+Kernel+Readback'
set output
