CC=gcc
CFLAGS=-c

FF  = gfortran
AR = ar rcs
CPLUS=g++

DIR=./
Executable=Executable



## Machine Specific optimizations
OPT =   -pthread  -fwrapv  -Wall -Wstrict-prototypes -fno-strict-aliasing -Wdate-time -D_FORTIFY_SOURCE=2  -fstack-protector-strong -Wformat -Werror=format-security -fPIC -O3  -ffast-math -msse4.2
#OPT += -DMat="float" 
INC =   -I ./$(DIR)  #-I/usr/include/python2.7/ -I/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ -I/usr/include/python2.7

.c.o:
	$(CC) -c  $(OPT) $(INC) $< -o $@



OBJ= $(DIR)/main.o  $(DIR)/block.o 

main: $(OBJ) 
	$(CC) $^ -o $@   -lm


clean:

	rm  $(DIR)/*.o 
