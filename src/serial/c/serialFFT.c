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

void fft(double complex *x, double complex *y, int n) {
    if (n == 1) {
        y[0] = x[0];
        return;
    }

    double complex xe[n/2], xo[n/2], ye[n/2], yo[n/2];

    for (int i = 0; i < n/2; i++) {
        xe[i] = x[2*i];
        xo[i] = x[2*i+1];
    }

    fft(xe, ye, n/2);
    fft(xo, yo, n/2);

    for (int i = 0; i < n/2; i++) {
        double complex t = cexp(-2 * M_PI * I * i / n) * yo[i];
        y[i] = ye[i] + t;
        y[i + n/2] = ye[i] - t;
    }
}

void fft2D(struct Matrix *m, double complex *y) {
    double complex x[m->size];
    for (int i = 0; i < m->size; i++) {
        for (int j = 0; j < m->size; j++) {
            x[j] = m->mat[i][j];
        }
        fft(x, &y[i*m->size], m->size);
    }

    for (int j = 0; j < m->size; j++) {
        double complex x[m->size];
        for (int i = 0; i < m->size; i++) {
            x[i] = y[i*m->size + j];
        }
        fft(x, x, m->size);
        for (int i = 0; i < m->size; i++) {
            y[i*m->size + j] = x[i];
        }
    }
}

int main(void) {
    struct Matrix source;
    struct FreqMatrix freq_domain;
    readMatrix(&source);
    freq_domain.size = source.size;
    double complex y[source.size*source.size];
    fft2D(&source, y);
    
    for (int k = 0; k < source.size; k++)
        for (int l = 0; l < source.size; l++)
            freq_domain.mat[k][l] = y[k*source.size + l] / (source.size * source.size);

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