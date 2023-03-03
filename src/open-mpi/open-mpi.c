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

/* TO DO: Coba algoritma FFT */
double complex handleRow(struct Matrix *mat, int k, int l, int i) {
    double complex row = 0.0;

    /* TO DO: Coba Reduce di sini */
    for (int j = 0; j < mat->size; j++) {
        row += handleElement(mat, k, l, i, j);
    }

    return row;
}

/* TO DO: Coba algoritma FFT */
double complex handleColumn(struct Matrix *mat, int k, int l) {
    double complex element = 0.0;

    /* TO DO: Coba Reduce di sini */
    for (int i = 0; i < mat->size; i++) {
        element += handleRow(mat, k, l, i);
    }

    return element;
}

double complex dft(struct Matrix *mat, int k, int l) {
    double complex element = handleColumn(mat, k, l);

    return element / (double) (mat->size * mat->size);
}

void fillMatrix(struct Matrix *mat, struct FreqMatrix *freq_domain) {
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_rank == 0) {
        /* Master Process */
        readMatrix(mat);
        freq_domain->size = mat->size;

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(&(mat->size), 1, MPI_INT, 0, MPI_COMM_WORLD);

        /* TO DO: For loop dapat dihilangkan dengan membuat matriks sebagai contiguous array */
        for (int i = 0; i < mat->size; i++) {
            MPI_Bcast(&(mat->mat[i]), mat->size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        /* Send Divisible Process */
        int element_per_process = mat->size / world_size;
        int extra_elements = mat->size % world_size;

        for (int i = 1; i < world_size; i++) {
            int index_sent = i * element_per_process;
            int elements_sent = index_sent < mat->size ? element_per_process : extra_elements;

            MPI_Send(&index_sent, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&elements_sent, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        /* Work on Master Process */
        for (int i = 0; i < element_per_process; i++) {
            for (int j = 0; j < mat->size; j++) {
                freq_domain->mat[i][j] = dft(mat, i, j);
            }
        }

        /* Receive Row */
        for (int i = 1; i < world_size; i++) {
            int elements_received;
            int index_received;
            MPI_Status status;

            MPI_Recv(&index_received, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&elements_received, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            /* TO DO: For loop dapat dihilangkan dengan membuat matriks sebagai contiguous array */
            for (int j = index_received; j < index_received + elements_received; j++) {
                MPI_Recv(&(freq_domain->mat[j]), mat->size, MPI_DOUBLE_COMPLEX, status.MPI_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        /* Print Result */
        double complex sum = 0.0;
        for (int m = 0; m < mat->size; m++) {
            for (int n = 0; n < mat->size; n++) {
                double complex el = freq_domain->mat[m][n];
                sum += el;
                printf("(%lf, %lf) ", creal(el), cimag(el));
            }
            printf("\n");
        }
        sum /= (double) (mat->size);
        printf("Sum : (%lf, %lf)\n", creal(sum), cimag(sum));

    } else {
        /* Slave Process */
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(&(mat->size), 1, MPI_INT, 0, MPI_COMM_WORLD);

        /* TO DO: For loop dapat dihilangkan dengan membuat matriks sebagai contiguous array */
        for (int i = 0; i < mat->size; i++) {
            MPI_Bcast(&(mat->mat[i]), mat->size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        /* Receive Process */
        int elements_received;
        int index_received;

        MPI_Recv(&index_received, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&elements_received, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Work on Slave Process */
        for (int i = index_received; i < index_received + elements_received; i++) {
            for (int j = 0; j < mat->size; j++) {
                freq_domain->mat[i][j] = dft(mat, i, j);
            }
        }

        /* Send Row */
        MPI_Send(&index_received, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&elements_received, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

        /* TO DO: For loop dapat dihilangkan dengan membuat matriks sebagai contiguous array */
        for (int i = index_received; i < index_received + elements_received; i++) {
            MPI_Send(&(freq_domain->mat[i]), mat->size, MPI_DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD);
        }
    }
}

int main(int argc, char *argv[]) {
    struct Matrix source;
    struct FreqMatrix freq_domain;

    MPI_Init(&argc, &argv);
    fillMatrix(&source, &freq_domain);
    MPI_Finalize();

    return 0;
}