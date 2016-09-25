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
 '< awk "FNR==NR{a[$1]=$2 FS $7;next}{ print $0, a[$1]}" data/1_scaling_c.dat data/2_scaling_nonblock.dat' u 1:($9/$7) w lp title "Non-Blocking"
 #'data/1_scaling_c.dat' using 1:7 w lp title 'CPU',\
 #'data/scaling.dat' using 1:7 w lp title 'OpenCL'
set output
