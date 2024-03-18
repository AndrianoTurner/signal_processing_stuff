import numpy as np
import matplotlib.pyplot as plt
import wave
import tkinter as tk
from tkinter import messagebox

def generate_signal(bits):
    bit_rate = 500  # бит в секунду
    carrier_frequency = 2000  # Гц (частота несущей)
    sampling_rate = 8000  # отсчетов в секунду

    # Вычисление времени на один бит
    time_per_bit = 1 / bit_rate
    # Создание временной оси
    time = np.linspace(0, len(bits) * time_per_bit, int(len(bits) * time_per_bit * sampling_rate), endpoint=False)

    # Амплитуда сигнала
    amplitude = 100

    # Пустой массив для сигнала
    signal = np.zeros(len(time))

    for i, bit in enumerate(bits):
        if bit == 1:
            # Добавление синусоиды для бита "1"
            signal[i * int(time_per_bit * sampling_rate): (i + 1) * int(time_per_bit * sampling_rate)] = \
                amplitude * np.sin(2 * np.pi * carrier_frequency * time[i * int(time_per_bit * sampling_rate): (i + 1) * int(time_per_bit * sampling_rate)])
        else:
            # Установка значения сигнала равным половине амплитуды для бита "0"
            signal[i * int(time_per_bit * sampling_rate): (i + 1) * int(time_per_bit * sampling_rate)] = \
                0.5 * amplitude * np.sin(2 * np.pi * carrier_frequency * time[i * int(time_per_bit * sampling_rate): (i + 1) * int(time_per_bit * sampling_rate)])

    return signal


# Функция для сохранения сигнала в файл WAV
def save_wav(filename, signal, sampling_rate):
    normalized_signal = signal / np.max(np.abs(signal))  # Нормализация сигнала
    signal = (normalized_signal * 32767).astype(np.int16)  # Приведение к 16-битному целому типу
    with wave.open(filename, 'wb') as wave_file:
        wave_file.setnchannels(1)  # Один канал (моно)
        wave_file.setsampwidth(2)  # 2 байта на сэмпл (16 бит)
        wave_file.setframerate(sampling_rate)  # Частота дискретизации
        wave_file.writeframes(signal.tobytes())  # Запись сигнала в файл


# Функция для построения визуализации сигнала
def plot_waveform(signal, time, interpolated_signal):
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, signal, 'b-', label='Дискретные отсчеты')
    plt.title('Сигнал')
    plt.xlabel('Время (мс)')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    time_interp = np.linspace(0, time[-1], len(interpolated_signal))  # Временная ось для интерполированного сигнала
    plt.step(time_interp, interpolated_signal, 'r-', label='Интерполированный сигнал')
    plt.title('Интерполированный сигнал')
    plt.xlabel('Время (мс)')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


# Функция, которая вызывается при нажатии кнопки "Сгенерировать и сохранить"
def generate_and_save():
    bits_str = entry.get()
    try:
        # Проверяем длину введенной битовой последовательности
        if len(bits_str) != 12:
            raise ValueError("Длина битовой последовательности должна быть равна 12")

        # Преобразуем строку с битами в список целых чисел
        bits = [int(bit) for bit in bits_str]

        sampling_rate = 8000
        signal = generate_signal(bits)
        time = np.arange(0, len(signal)) * 1000 / sampling_rate

        # Интерполяция сигнала
        interpolated_signal = np.interp(time, time, signal)

        save_wav('output.wav', signal, sampling_rate)

        messagebox.showinfo("Успех", "Файл успешно сохранен как output.wav")
        plot_waveform(signal, time, interpolated_signal)

    except ValueError as e:
        messagebox.showerror("Ошибка", str(e))


# Создание графического интерфейса
app = tk.Tk()
app.title("Генератор WAV файла")

label = tk.Label(app, text="Введите битовую последовательность (12 бит):")
label.pack()

entry = tk.Entry(app)
entry.pack()

button = tk.Button(app, text="Сгенерировать и сохранить", command=generate_and_save)
button.pack()

app.mainloop()
