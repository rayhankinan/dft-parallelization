# Tugas Kecil - Paralel DFT

## Open MP

Program ini dibuat untuk memenuhi tugas kecil mata kuliah IF3230 Sistem Paralel dan Terdistribusi 2023. Program ini dibuat menggunakan bahasa pemrograman C dan library OpenMP. Paralelisasi program memanfaatkan OpenMP dengan menggunakan directive pragma untuk melakukan parallelisasi pada perhitungan DFT.

Inti dari penggunaan OpenMP adalah melakukan parallelisasi pada DFT. Penggunaan OpenMP diimplementasikan dengan menggunakan directive pragma pada loop for untuk melakukan parallelisasi pada looping.

Alasan penggunaan OpenMP adalah untuk mempercepat waktu perhitungan DFT pada matriks yang besar dengan memanfaatkan beberapa thread secara bersamaan. Dengan menggunakan OpenMP, perhitungan DFT dapat dilakukan secara paralel pada beberapa thread yang tersedia pada komputer yang digunakan.

Untuk implementasi OpenMP pada program ini, dapat digunakan directive pragma #pragma omp parallel pada loop for untuk melakukan parallelisasi pada looping. Selain itu, juga dapat digunakan directive pragma #pragma omp for collapse(2) pada loop for yang memiliki nested loop, dimana pada kasus ini berjumlah dua nested loop, untuk melakukan parallelisasi pada nested loop tersebut.
