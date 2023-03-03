#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

    return element / (double) (mat->size * mat->size);
}

int main(void) {
    struct Matrix source;
    struct FreqMatrix freq_domain;
    readMatrix(&source);
    freq_domain.size = source.size;
    
    // make the code to executed in parallel and the data is shared in all threads
    #pragma omp parallel shared(freq_domain, source)
    {
        // execute the loop in parallel and collapse the two nested loops into single loop
        #pragma omp for collapse(2)
        for (int k = 0; k < source.size; k++) {
            for (int l = 0; l < source.size; l++) {
                freq_domain.mat[k][l] = dft(&source, k, l);
            }
        }
    }

    double complex sum = 0.0;
    for (int k = 0; k < source.size; k++) {
        for (int l = 0; l < source.size; l++) {
            double complex el = freq_domain.mat[k][l];
            printf("(%lf, %lf) ", creal(el), cimag(el));
            sum += el;
        }
        printf("\n");
    }
    sum /= source.size;

    printf("Sum : (%lf, %lf)\n", creal(sum), cimag(sum));

    return 0;
}