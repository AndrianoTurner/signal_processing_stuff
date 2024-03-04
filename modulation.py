import numpy as np
import wave
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import math
# Функция для генерации сигнала на основе битовой последовательности
def generate_signal(bits, bit_rate, carrier_frequency, sampling_rate, quantization_depth):
    # Вычисление времени на один бит
    time_per_bit = 1 / bit_rate
    # Общее время сигнала
    total_time = len(bits) * time_per_bit
    # Создание временной оси
    time = np.linspace(0, total_time, int(total_time * sampling_rate))
    print(f"Total time: {total_time}")
    # Амплитуда сигнала
    amplitude = 127
    signal = []
    mod = 0.5
    # Формирование сигнала для каждого бита
    for bit in bits:
        if bit == 1:
            # Добавление синусоиды для бита "1"
            signal += list(amplitude * np.sin(2 * np.pi * carrier_frequency * time[:int(time_per_bit * sampling_rate)]))
        else:
            # Добавление нулевого сигнала для бита "0"
            signal += list(mod * amplitude * np.sin(2 * np.pi * carrier_frequency * time[:int(time_per_bit * sampling_rate)]))
            #signal += [0] * int(time_per_bit * sampling_rate)

    # Преобразование списка в массив NumPy и установка типа данных int8
    signal = np.array(signal, dtype=np.int8)

    # Квантование сигнала
    quantization_levels = 2**quantization_depth
    #signal = np.round(signal * (quantization_levels - 1) / amplitude)
    signal = np.clip(signal, -128, 127)

    return signal

# Функция для сохранения сигнала в файл WAV
def save_wav(filename, signal, sampling_rate):
    with wave.open(filename, 'w') as wave_file:
        wave_file.setnchannels(1)
        wave_file.setsampwidth(1)
        wave_file.setframerate(sampling_rate)
        wave_file.writeframes(signal.tobytes())

# Функция для построения визуализации сигнала
def plot_waveform(q_values, sampling_rate, bit_rate):

    print(len(q_values))
    q_steps = np.arange(0, len(q_values)) / sampling_rate
    
    interpolated_f = interp1d(q_steps, q_values)
    
    #plt.plot(new_time, interpolated_f(new_time))    
    fig = plt.figure(figsize=(12,10))

    ax = fig.add_subplot(3,1,(1,2))
    ax.step(q_steps,q_values,label="step")
    ax.plot(q_steps,q_values,label="exact")
    new_steps = np.arange(0, len(q_values) * 10)
    ax.plot(new_steps,interpolated_f(new_steps))
    fig.legend()
    plt.autoscale(True,"y",None)
    #plt.xlim((0,10))
    plt.show()

# Пример использования
bits = [1,1,0,0,1,1,1,0,1,0]  # Пример битовой последовательности
bit_rate = 0.5  # бит в секунду
carrier_frequency = 2000  # Гц
sampling_rate = 8  # отсчетов в секунду
quantization_depth = 8  # бит

# Генерация сигнала, сохранение в файл и построение визуализации
signal = generate_signal(bits, bit_rate, carrier_frequency, sampling_rate, quantization_depth)
save_wav('output.wav', signal, sampling_rate)
plot_waveform(signal, sampling_rate, bit_rate)
