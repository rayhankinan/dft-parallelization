OUTPUT_FOLDER = bin

all: serial parallel

parallel:
	mpicc src/open-mpi/parallel.c -o $(OUTPUT_FOLDER)/parallel -lm

serial:
	gcc src/serial/c/serial.c -o $(OUTPUT_FOLDER)/serial -lm