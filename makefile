OUTPUT_FOLDER = bin

all: serial parallel

parallel:
	mpic++ src/open-mpi/parallel.cpp -o $(OUTPUT_FOLDER)/parallel -lm -std=c++17

serial:
	g++ src/serial/c++/serial.cpp -o $(OUTPUT_FOLDER)/serial -lm -std=c++17