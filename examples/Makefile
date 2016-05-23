CC=g++
CPPFLAGS=-O2 -larmadillo -std=c++11 -fopenmp
OBJ = joined_reservoir.o
BIN = bin

all: joined_reservoir

%.o: %.c
	$(CC) -c -o $@ $< $(CPPFLAGS)

joined_reservoir: $(OBJ)
	mkdir $(BIN)
	$(CC) -o $(BIN)/$@ $< $(CPPFLAGS)

clean:
	rm *.o