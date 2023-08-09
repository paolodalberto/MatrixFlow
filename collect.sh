
for core in 1 2 4 8 16
do
    export OPENBLAS_NUM_THREADS=$core

    for k in 10 50 75 100 125 150 200 300 500
    do
	echo $k
	python3 Examples/play_4.py -e "middle" -m 4 -n 9  -k $k -v "t" | grep "#" >> out.$core 
    done
done 
    
