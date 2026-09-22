buatkan contoh microservice untuk 2 services. 
1. Agregate Service Pendaftaran Maba
2. Agregate Service Admin UTBK

hubungkan 2 service ini dengan Restful API.

kerjakan tombol fitur generate akun UTBK di aplikasi Admin UTBK berdasarkan database pendaftaran maba. 
akun didapatkan jika maba sudah membayar biaya pendaftaran. 
jadi pada database pendaftaran maba ada kolom flag untuk maba bayar(camaba-eligble) dan tidak bayar (camaba tidak eligble).
Service Pendaftaran Maba hanya api GET Camaba-Eligble saja.
buat tombol yang memanggil service GET Camaba-Eligble di aplikasi admin.

tampilan HTML hanya untuk Admin UTBK saja. 

gunakan contoh simple saja, jangan kompleks!
gunakan sqlite. 
