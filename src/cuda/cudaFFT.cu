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

void transpose(FreqMatrix *m) {
    for (int i = 0; i < m->size; i++) {
        for (int j = i + 1; j < m->size; j++) {
            cuDoubleComplex tmp = m->mat[i][j];
            m->mat[i][j] = m->mat[j][i];
            m->mat[j][i] = tmp;
        }
    }
}

__global__ void fft_kernel(FreqMatrix *m, FreqMatrix *fm, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < size && j < size) {
        fm->mat[i][j] = make_cuDoubleComplex(0, 0);
        for (int k = 0; k < size; k++) {
            double theta = -2 * M_PI * k * j / size;
            cuDoubleComplex w = make_cuDoubleComplex(cos(theta), sin(theta));
            fm->mat[i][j] = cuCadd(fm->mat[i][j], cuCmul(m->mat[i][k], w));
        }
        fm->mat[i][j] = cuCdiv(fm->mat[i][j], make_cuDoubleComplex(size, 0));
    }
}

__global__ void fft_kernel_transpose(FreqMatrix *m, FreqMatrix *fm, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < size && j < size) {
        fm->mat[i][j] = make_cuDoubleComplex(0, 0);
        for (int k = 0; k < size; k++) {
            double theta = -2 * M_PI * k * i / size;
            cuDoubleComplex w = make_cuDoubleComplex(cos(theta), sin(theta));
            fm->mat[i][j] = cuCadd(fm->mat[i][j], cuCmul(m->mat[j][k], w));
        }
    }
    fm->mat[i][j] = cuCdiv(fm->mat[i][j], make_cuDoubleComplex(size, 0));
}

void fft(Matrix *mat, FreqMatrix *freq_domain) {
    int size = mat->size;
    freq_domain->size = size;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            freq_domain->mat[i][j] = make_cuDoubleComplex(mat->mat[i][j], 0);
        }
    }
    FreqMatrix *d_mat, *d_freq_domain;

    cudaMalloc((void **)&d_mat, sizeof(FreqMatrix));
    cudaMalloc((void **)&d_freq_domain, sizeof(FreqMatrix));
    cudaMemcpy(d_mat, freq_domain, sizeof(FreqMatrix), cudaMemcpyHostToDevice);
    cudaMemcpy(d_freq_domain, freq_domain, sizeof(FreqMatrix), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((size + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                      (size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    fft_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_mat, d_freq_domain, size);
    cudaDeviceSynchronize();
    cudaMemcpy(freq_domain, d_freq_domain, sizeof(FreqMatrix), cudaMemcpyDeviceToHost);

    transpose(freq_domain);

    cudaMemcpy(d_mat, freq_domain, sizeof(FreqMatrix), cudaMemcpyHostToDevice);

    fft_kernel_transpose<<<blocksPerGrid, threadsPerBlock>>>(d_mat, d_freq_domain, size);
    cudaDeviceSynchronize();
    cudaMemcpy(freq_domain, d_freq_domain, sizeof(FreqMatrix), cudaMemcpyDeviceToHost);

    cudaFree(d_mat);
    cudaFree(d_freq_domain);
}

int main() {
    Matrix mat;
    FreqMatrix freq_domain;
    readMatrix(&mat);
    fft(&mat, &freq_domain);
    printMatrix(&freq_domain);
    return 0;
}