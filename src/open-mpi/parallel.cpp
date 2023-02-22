// mpic++ parallel.cpp -o parallel

#include <complex>
#include <iostream>
#include <vector>
#include "mpi.h"

using namespace std;

const complex<double> pi() {
    return atan(1) * 4;
}

class Matrix {
    private:
        int n;
        vector<vector<double>> *data_ptr;
        vector<vector<double>> &data;

        complex<double> handleElement(int k, int l, int i, int j) {
            complex<double> sample = (k * i / (double) this->n) + (l * j / (double) this->n);
            complex<double> exponent = exp(-2.0i * pi() * sample);
            complex<double> element = this->data[i][j] * exponent;

            return element;
        }

        complex<double> handlerRow(int k, int l, int i) {
            MPI_Init(NULL, NULL);

            int world_rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

            complex<double> row = 0;
            if (world_rank == 0) {
                /* Master Process */

                /* Send Index */
                for (int j = 0; j < this->n; j++) {
                    MPI_Send(&j, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
                }

                /* Receive Element */
                complex<double> element;
                for (int j = 0; j < this->n; j++) {
                    MPI_Recv(&element, 1, MPI_DOUBLE_COMPLEX, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    row += element;
                }
            } else {
                /* Slave Process */

                /* Receive Index */
                int j;
                MPI_Recv(&j, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                /* Send Element */
                complex<double> element = this->handleElement(k, l, i, j);
                MPI_Send(&element, 1, MPI_DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD);
            }

            MPI_Finalize();

            return row;
        }

        complex<double> handlerColumn(int k, int l) {
            MPI_Init(NULL, NULL);

            int world_rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

            complex<double> total = 0;
            if (world_rank == 0) {
                /* Master Process */

                /* Send Index */
                for (int i = 0; i < this->n; i++) {
                    MPI_Send(&i, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
                }

                /* Receive Row */
                complex<double> row;
                for (int i = 0; i < this->n; i++) {
                    MPI_Recv(&row, 1, MPI_DOUBLE_COMPLEX, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    total += row;
                }
            } else {
                /* Slave Process */

                /* Receive Index */
                int i;
                MPI_Recv(&i, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                /* Send Row */
                complex<double> row = this->handlerRow(k, l, i);
                MPI_Send(&row, 1, MPI_DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD);
            }

            MPI_Finalize();

            return total;
        }

    public:
        Matrix() : data_ptr(new vector<vector<double>>()), data(*data_ptr) {
            this->data.resize(10, vector<double>(10, 0));
        }

        ~Matrix() {
            delete data_ptr;
        }

        void readMatrix() {
            cin >> this->n;
            this->data.resize(this->n, vector<double>(this->n, 0.0));
            for (int i = 0; i < this->n; i++)
                for (int j = 0; j < this->n; j++)
                    cin >> this->data[i][j];
        }

        int size() {
            return this->n;
        }

        complex<double> dftElement(int k, int l) {
            complex<double> total = this->handlerColumn(k, l);

            return total / (complex<double>)(this->n * this->n);
        }
};

int main(void) {
    Matrix *source = new Matrix();
    source->readMatrix();

    complex<double> sum = 0.0;
    for (int i = 0; i < source->size(); i++) {
        for (int j = 0; j < source->size(); j++) {
            complex<double> el = source->dftElement(i, j);
            sum += el;
        }
    }
    cout << sum / (double) source->size();
    return 0;
}