#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define MAX_N 512

struct Matrix {
    int size;
    double mat[MAX_N][MAX_N];
};

struct FreqMatrix {
    int size;
    double complex mat[MAX_N][MAX_N];
};

void readMatrix(struct Matrix *m) {
    scanf("%d", &(m->size));
    for (int i = 0; i < m->size; i++)
        for (int j = 0; j < m->size; j++)
            scanf("%lf", &(m->mat[i][j]));
}

double complex handleElement(struct Matrix *mat, int k, int l, int i, int j) {
    double complex arg = (k * i / (double) mat->size) + (l * j / (double) mat->size);
    double complex exponent = cexp(-2.0 * I * M_PI * arg);
    double complex element = mat->mat[i][j] * exponent;

    return element;
}

double complex handleRow(struct Matrix *mat, int k, int l, int i) {
    double complex row = 0.0;
    for (int j = 0; j < mat->size; j++) {
        row += handleElement(mat, k, l, i, j);
    }

    return row;
}

double complex handleColumn(struct Matrix *mat, int k, int l) {
    double complex element = 0.0;
    for (int i = 0; i < mat->size; i++) {
        element += handleRow(mat, k, l, i);
    }

    return element;
}

double complex dft(struct Matrix *mat, int k, int l) {
    double complex element = handleColumn(mat, k, l);

    return element / (double) (mat->size*mat->size);
}

int main(void) {
    struct Matrix source;
    struct FreqMatrix freq_domain;

    readMatrix(&source);
    freq_domain.size = source.size;

    MPI_Init(NULL, NULL);
    for (int m = 0; m < source.size; m++)
        for (int n = 0; n < source.size; n++)
            freq_domain.mat[m][n] = dft(&source, m, n);
    MPI_Finalize();

    double complex sum = 0.0;
    for (int m = 0; m < source.size; m++) {
        for (int n = 0; n < source.size; n++) {
            double complex el = freq_domain.mat[m][n];
            sum += el;

            printf("(%lf, %lf) ", creal(el), cimag(el));
        }

        printf("\n");
    }

    printf("Sum : (%lf, %lf)\n", creal(sum), cimag(sum));

    return 0;
}