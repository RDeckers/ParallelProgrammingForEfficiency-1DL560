set term term_type
set output output_file
#set logscale xy 2
#set key outside;
#set key right top;
#set format x "2^{%L}"
set ylabel "runtime (ns)"
set yrange [0:3.0*10**8]
#set xlabel "Image dimensions (width & height)"
set grid xtics ytics;
#set logscale y;
#set format y "10^{%L}"
set xtics nomirror rotate by -45 scale 0
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set boxwidth 1.0
set xrange [0.5:12.5];
#plot '< paste data/opencl.dat data/refference.dat' using 0:2:xtic(1) with boxes
plot\
  'data/0_refference.dat' using 2:xtic(1) ti "Refference",\
  'data/7_encoder_dumb_loc_vec3.dat' u 2 ti "Optimized";

#titles = system('ls data/*.dat')
#plot '< paste data/*.dat' using 2:xtic(1) ti word(titles, 1),\
# '' u 4 ti  word(titles, 2),\
# '' u 6 ti  word(titles,3),\
# '' u 8 ti  word(titles,4),\
# '' u 10 ti word(titles,5),\
# '' u 12 ti word(titles,6),\
# '' u 14 ti word(titles,7),\
# '' u 16 ti word(titles,8),\
# '' u 18 ti word(titles,9);
#FILES = system("ls data/*.dat")
#plot for [data in FILES] data using 2:xtic(1) with boxes fc rgb "#dd2222" ti data;


set output
