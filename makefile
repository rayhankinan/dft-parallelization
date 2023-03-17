OUTPUT_FOLDER = bin
TEST_FOLDER = result

all: serial mpi mp cuda cudaFFT

mpi:
	mpicc src/open-mpi/open-mpi.c -o $(OUTPUT_FOLDER)/open-mpi -lm
	time ./${OUTPUT_FOLDER}/open-mpi < ./test_case/32.txt > ${TEST_FOLDER}/open-mpi.txt

serial:
	gcc src/serial/c/serial.c -o $(OUTPUT_FOLDER)/serial -lm
	time ./${OUTPUT_FOLDER}/serial < ./test_case/32.txt > ${TEST_FOLDER}/serial.txt

mp:
	gcc src/open-mp/mp.c --openmp -o $(OUTPUT_FOLDER)/open-mp -lm
	time ./${OUTPUT_FOLDER}/open-mp < ./test_case/32.txt > ${TEST_FOLDER}/open-mp.txt

cuda:
	nvcc src/cuda/cuda.cu -o $(OUTPUT_FOLDER)/cuda -lm
	time ./${OUTPUT_FOLDER}/cuda < ./test_case/512.txt > ${TEST_FOLDER}/cuda.txt

cudaFFT:
	nvcc src/cuda/cudaFFT.cu -o $(OUTPUT_FOLDER)/cudaFFT -lm
	time ./${OUTPUT_FOLDER}/cudaFFT < ./test_case/512.txt > ${TEST_FOLDER}/cudaFFT.txt