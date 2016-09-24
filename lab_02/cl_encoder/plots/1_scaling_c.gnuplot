set term term_type
set output output_file
set logscale x 2;
set key outside;
set key center top;
set format x "2^{%L}"
#set format y "2^{%L}"
set ylabel "speedup"
set xlabel "work dimnsions"
set grid;
plot\
  '< paste data/0_scaling.dat data/1_scaling_c.dat' u 1:($14/$7) w lp
 #'data/c_scaling.dat' using 1:7 w lp title 'CPU',\
 #'data/scaling.dat' using 1:7 w lp title 'OpenCL'
set output
