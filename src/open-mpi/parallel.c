#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define MAX_N 512

struct Element {
    int k;
    int l;
    double value;
};

struct Matrix {
    int size;
    double mat[MAX_N][MAX_N];
};

struct FreqElement {
    int k;
    int l;
    double complex value;
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

struct FreqElement* handleElement(struct Matrix *mat, int k, int l, int i, int j) {
    struct FreqElement *element;

    double complex arg = (k * i / (double) mat->size) + (l * j / (double) mat->size);
    double complex exponent = cexp(-2.0 * I * M_PI * arg);

    element = malloc(sizeof(struct FreqElement));
    element->k = i;
    element->l = j;
    element->value = mat->mat[i][j] * exponent;

    return element;
}

struct FreqElement* handleRow(struct Matrix *mat, int k, int l, int i) {
    struct FreqElement *row;

    row = malloc(sizeof(struct FreqElement));
    row->k = i;
    row->l = l;
    row->value = 0.0;

    for (int j = 0; j < mat->size; j++) {
        struct FreqElement *tempElement;

        tempElement = handleElement(mat, k, l, i, j);
        row->value += tempElement->value;

        free(tempElement);
    }

    return row;
}

struct FreqElement* handleColumn(struct Matrix *mat, int k, int l) {
    struct FreqElement *element;

    element = malloc(sizeof(struct FreqElement));
    element->k = k;
    element->l = l;
    element->value = 0.0;

    for (int i = 0; i < mat->size; i++) {
        struct FreqElement *tempRow;

        tempRow = handleRow(mat, k, l, i);
        element->value += tempRow->value;

        free(tempRow);
    }

    element->value /= (double) (mat->size * mat->size);

    return element;
}

struct FreqElement* dft(struct Matrix *mat, int k, int l) {
    return handleColumn(mat, k, l);
}

int main(void) {
    struct Matrix source;
    struct FreqMatrix freq_domain;

    readMatrix(&source);
    freq_domain.size = source.size;

    MPI_Init(NULL, NULL);
    for (int m = 0; m < source.size; m++) {
        for (int n = 0; n < source.size; n++) {
            struct FreqElement *element = dft(&source, m, n);
            freq_domain.mat[m][n] = element->value;
        }
    }
    MPI_Finalize();

    /* Print Result */
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