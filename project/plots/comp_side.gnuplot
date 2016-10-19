set term term_type
set output output_file
#set logscale xy 2
set key outside;
set key center top;
#set format x "2^{%L}"
set ylabel "runtime (ns)"
set yrange [0:1*10**9]
#set xlabel "Image dimensions (width & height)"
set grid xtics ytics;
#set logscale y;
#set format y "10^{%L}"
set xtics nomirror rotate by -45 scale 0
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set boxwidth 1.0
#plot '< paste data/opencl.dat data/refference.dat' using 0:2:xtic(1) with boxes
plot '< paste data/*.dat' using 2:xtic(1) fc rgb "#dd2222" ti "refference",\
 '' u 4 fc rgb "#2222dd" ti "Optimized",\
 '' u 6 fc rgb "#2222dd" ti "Optimized2",\
 '' u 8 fc rgb "#2222dd" ti "Optimized2",\
 '' u 10 fc rgb "#2222dd" ti "Optimized2",\
 '' u 12 fc rgb "#2222dd" ti "Optimized2";
#FILES = system("ls data/*.dat")
#plot for [data in FILES] data using 2:xtic(1) with boxes fc rgb "#dd2222" ti data;


set output
