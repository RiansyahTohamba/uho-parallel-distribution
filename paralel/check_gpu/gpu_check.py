import tensorflow as tf
import torch

def cek_gpu():
    print("=== CEK GPU VIA TENSORFLOW ===")
    gpus_tf = tf.config.list_physical_devices('GPU')
    if gpus_tf:
        print(f"Status: GPU Terdeteksi ({len(gpus_tf)} unit)")
        for gpu in gpus_tf:
            print(f" - {gpu.name}")
    else:
        print("Status: Tidak ada GPU terdeteksi (Hanya CPU)")

    print("\n=== CEK GPU VIA PYTORCH ===")
    if torch.cuda.is_available():
        print(f"Status: GPU Terdeteksi")
        print(f" - Jumlah GPU : {torch.cuda.device_count()}")
        print(f" - Nama GPU   : {torch.cuda.get_device_name(0)}")
    else:
        print("Status: Tidak ada GPU/CUDA terdeteksi (Hanya CPU)")

if __name__ == "__main__":
    cek_gpu()