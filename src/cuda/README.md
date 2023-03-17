# Tugas Kecil - Paralel DFT

## CUDA

Program ini dibuat untuk memenuhi tugas kecil mata kuliah IF3230 Sistem Paralel dan Terdistribusi 2023. Program ini dibuat menggunakan bahasa pemrograman C++ dan library CUDA. 

Implementasi paralelisasi ini digunakan pada algoritmaFFT. Perhitungan dilakukan dengan formula FFT dan dilakukan secara 2 tahap. Tahap pertama melakukan perhitungan FFT pada tiap baris matriks, dan tahap kedua melakukan perhitungan FFT pada tiap kolom matriks.

Paralelisasi dilakukan dengan membuat tiap thread pada block melakukan perhitungan pada tiap elemen pada matriks output. Pada tahap kedua , matriks di transpose terlsebih dahulu agar akses memori bersifat coallesced. Algoritma ini jauh lebih cepat dibandingkan dengan algoritma DFT biasa.