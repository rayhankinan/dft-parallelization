// create dft program with cuda parallelization
#include <cuComplex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MAX_N 512

typedef struct {
    int size;
    double mat[MAX_N][MAX_N];
} Matrix;

typedef struct {
    int size;
    cuDoubleComplex mat[MAX_N][MAX_N];
} FreqMatrix;

void readMatrix(Matrix *m) {
    scanf("%d", &(m->size));
    for (int i = 0; i < m->size; i++)
        for (int j = 0; j < m->size; j++)
            scanf("%lf", &(m->mat[i][j]));
}

void printMatrix(FreqMatrix *m) {
    cuDoubleComplex sum = make_cuDoubleComplex(0, 0);
    for (int i = 0; i < m->size; i++) {
        for (int j = 0; j < m->size; j++) {
            sum = cuCadd(sum, m->mat[i][j]);
            printf("(%lf, %lf) ", cuCreal(m->mat[i][j]), cuCimag(m->mat[i][j]));
        }
        printf("\n");
    }
    sum = cuCdiv(sum, make_cuDoubleComplex(m->size, 0));
    printf("sum = (%lf, %lf)", cuCreal(sum), cuCimag(sum));
}

__device__ cuDoubleComplex handleElement(Matrix *mat, int k, int l, int i, int j) {
    double angle = 2 * M_PI * (i * k + j * l) / mat->size;
    cuDoubleComplex exp = make_cuDoubleComplex(cos(angle), -sin(angle));
    return cuCmul(make_cuDoubleComplex(mat->mat[i][j], 0), exp);
}

__device__ cuDoubleComplex handleRow(Matrix *mat, int k, int l, int i) {
    cuDoubleComplex sum = make_cuDoubleComplex(0, 0);
    for (int j = 0; j < mat->size; j++) {
        sum = cuCadd(sum, handleElement(mat, k, l, i, j));
    }
    return sum;
}

__device__ cuDoubleComplex handleColumn(Matrix *mat, int k, int l) {
    cuDoubleComplex sum = make_cuDoubleComplex(0, 0);
    for (int i = 0; i < mat->size; i++) {
        sum = cuCadd(sum, handleRow(mat, k, l, i));
    }
    return sum;
}

__device__ cuDoubleComplex dft(Matrix *mat, int k, int l) {
    cuDoubleComplex sum = handleColumn(mat, k, l);
    sum = cuCdiv(sum, make_cuDoubleComplex(mat->size * mat->size, 0));
    return sum;
}

__global__ void dft_kernel(Matrix *source, FreqMatrix *freq_domain) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y * blockDim.y + threadIdx.y;
    if (k < source->size && l < source->size) {
        freq_domain->mat[k][l] = dft(source, k, l);
    }
}

void dft(Matrix *source, FreqMatrix *freq_domain) {
    freq_domain->size = source->size;
    Matrix *dev_source;
    FreqMatrix *dev_freq_domain;
    cudaMalloc((void**)&dev_source, sizeof(Matrix));
    cudaMalloc((void**)&dev_freq_domain, sizeof(FreqMatrix));
    cudaMemcpy(dev_source, source, sizeof(Matrix), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_freq_domain, freq_domain, sizeof(FreqMatrix), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((source->size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (source->size + threadsPerBlock.y - 1) / threadsPerBlock.y);
    dft_kernel<<<blocksPerGrid, threadsPerBlock>>>(dev_source, dev_freq_domain);
    cudaDeviceSynchronize();
    cudaMemcpy(freq_domain, dev_freq_domain, sizeof(FreqMatrix), cudaMemcpyDeviceToHost);
    cudaFree(dev_source);
    cudaFree(dev_freq_domain);
}

int main() {
    Matrix source;
    FreqMatrix freq_domain;
    readMatrix(&source);
    dft(&source, &freq_domain);
    printMatrix(&freq_domain);
    return 0;
}
