OUTPUT_FOLDER = bin
TEST_FOLDER = result

all: serial mpi mp

mpi:
	mpicc src/open-mpi/parallel.c -o $(OUTPUT_FOLDER)/parallel -lm

serial:
	gcc src/serial/c/serial.c -o $(OUTPUT_FOLDER)/serial -lm
	time ./${OUTPUT_FOLDER}/serail < ./test_case/32.txt > ${TEST_FOLDER}/serial.txt

mp:
	gcc src/open-mp/mp.c --openmp -o $(OUTPUT_FOLDER)/open-mp -lm
	time ./${OUTPUT_FOLDER}/open-mp < ./test_case/32.txt > ${TEST_FOLDER}/open-mp.txt