CC=gcc
CFLAGS=-c

FF  = gfortran
AR = ar rcs
CPLUS=g++

DIR=./
Executable=Executable



## Machine Specific optimizations
OPT = -pthread  -fwrapv  -Wall  -fno-strict-aliasing -Wdate-time \
	-D_FORTIFY_SOURCE=2  -fstack-protector-strong -Wformat \
	-Werror=format-security -fPIC -O3 -msse4.1 -ffast-math \
	#-mbmi2 -mavx2 #-mavx512f  -mavx512dq   -mavx512vl

#OPT += -DMat="float" 
INC =   -I ./$(DIR)  #-I/usr/include/python2.7/ -I/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ -I/usr/include/python2.7

OBJ= $(DIR)/benchmark.o #$(DIR)/fast-accurate.o  $(DIR)/accurate.o  $(DIR)/naive.o	


.cc.o: 	$(OBJ)
	$(CPLUS) -c  $(OPT) $(INC) $< -o $@





main: $(OBJ)
	$(CPLUS) $(OPT) $^ -o $@   -lm


clean:

	rm  $(DIR)/*.o 
