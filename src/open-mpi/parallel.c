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

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_rank == 0) {
        /* Master Process */
        int element_per_process = mat->size / world_size;
        int extra_elements = mat->size % world_size;

        /* Send Divisible Process */
        for (int i = 1; i < world_size; i++) {
            int processed_size = i * element_per_process;
            int current_size = processed_size < mat->size ? element_per_process : extra_elements;

            MPI_Send(&current_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&processed_size, current_size, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        /* Work on Master Process */
        for (int i = 0; i < element_per_process; i++) {
            element += handleRow(mat, k, l, i);
        }

        /* Receive Row */
        double complex row;
        for (int i = 1; i < world_size; i++) {
            MPI_Recv(&row, 1, MPI_DOUBLE_COMPLEX, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            element += row;
        }
    } else {
        /* Slave Process */

        /* Receive Process */
        int elements_received;
        int index_received;
        double complex temp_element = 0.0;

        MPI_Recv(&elements_received, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&index_received, elements_received, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Work on Slave Process */
        for (int i = index_received; i < index_received + elements_received; i++) {
            temp_element += handleRow(mat, k, l, i);
        }

        /* Send Row */
        MPI_Send(&temp_element, 1, MPI_DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD);
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
    for (int m = 0; m < source.size; m++) {
        for (int n = 0; n < source.size; n++) {
            freq_domain.mat[m][n] = dft(&source, m, n);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
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