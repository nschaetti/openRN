CC=g++
CPPFLAGS=-O2 -larmadillo -std=c++11 -fopenmp
OBJ = joined_reservoir.o
BIN = bin

all: joined_reservoir

%.o: %.c
	$(CC) -c -o $@ $< $(CPPFLAGS)

joined_reservoir: $(OBJ)
	mkdir -p $(BIN)
	$(CC) -o $(BIN)/$@ $< $(CPPFLAGS)

install:
	cp $(BIN)/* /bin

clean:
	rm *.o