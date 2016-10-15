set term term_type
set output output_file
#set logscale xy 2
set key outside;
set key center top;
#set format x "2^{%L}"
set ylabel "runtime (ns)"
set yrange [10**5:10**11]
#set xlabel "Image dimensions (width & height)"
set grid mytics xtics ytics;
set logscale y;
set format y "10^{%L}"
set xtics nomirror rotate by -45
set boxwidth 0.75
set style fill solid
plot "data/opencl.dat" using 0:2:xtic(1) with boxes
set output
