# Tugas Kecil - Paralel DFT

## Open MPI

Program ini dibuat untuk memenuhi tugas kecil mata kuliah IF3230 Sistem Paralel dan Terdistribusi 2023. Program ini dibuat menggunakan bahasa pemrograman C dan library Open MPI. Paralelisasi program memanfaatkan MPI (Message Passing Interface) dengan fungsi broadcast dan send-receive untuk mengirimkan data dari proses master ke proses slave.

Inti dari penggunaan proses paralelisasi adalah melakukan pengiriman matriks input ke seluruh slave menggunakan fungsi broadcast, lalu master juga mengirimkan data berupa index dan banyak elemen yang akan diproses oleh slave.

Setelah proses pengiriman dilakukan, master dan slave akan melakukan perhitungan DFT secara paralel. Setelah proses perhitungan selesai, slave akan mengirimkan hasil perhitungan ke master menggunakan fungsi send-receive.

Alasan kedua skema tersebut digunakan adalah :

- Untuk broadcast, seluruh slave akan menerima data (matriks awal) yang sama sehingga tidak perlu melakukan send-receive. Seluruh slave juga perlu mengetahui matriks awal karena perhitungan DFT tidak dapat dilakukan tanpa matriks awal.
- Untuk send-receive, data yang akan diterima oleh slave berbeda-beda sehingga tidak dapat menggunakan broadcast. Setiap slave akan menerima data yang relatif adil. Namun apabila jumlah data tidak dapat dibagi secara merata antar slave, maka slave terakhir akan menerima data yang lebih sedikit (ditandakan dengan extra_elements pada kode)
