CC=g++
CPPFLAGS=-O2 -larmadillo -std=c++11 -fopenmp /home/couturie/tools/openblas/lib/libopenblas_haswellp-r0.2.15.a -lpthread -lgfortran 
OBJ = joined_reservoir.o
OBJ2 = Reservoir_mnist_new2_im_nils2.o
BIN = bin

all: joined_reservoir Reservoir_mnist_new2_im_nils2

%.o: %.c
	$(CC) -c -o $@ $< $(CPPFLAGS) -I/home/couturie/Downloads/OpenBLAS-0.2.15/

joined_reservoir: $(OBJ)
	mkdir -p $(BIN)
	$(CC) -o $(BIN)/$@ $< $(CPPFLAGS)

Reservoir_mnist_new2_im_nils2: $(OBJ2)
	mkdir -p $(BIN)
	$(CC) -o $(BIN)/$@ $< $(CPPFLAGS)

install:
	cp $(BIN)/* /bin

clean:
	rm *.o
