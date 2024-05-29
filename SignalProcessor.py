import numpy as np
from itertools import product
import wave
class SignalProcessor():
    def __init__(self,bits,bit_rate,carrier_frequency,sampling_rate) -> None:
        self.bits = bits
        self.carrier_frequency = carrier_frequency
        self.bit_rate = bit_rate
        self.sampling_rate = sampling_rate
        self.amplitude = 100
        

    def generate_am_signal(self):

        time_per_bit = 1 / self.bit_rate
        time = np.linspace(0, len(self.bits) * time_per_bit, int(len(self.bits) * time_per_bit * self.sampling_rate), endpoint=False)
        signal = np.zeros(len(time))
        for i, bit in enumerate(self.bits):
            time_start = i * int(time_per_bit * self.sampling_rate)
            time_end = (i + 1) * int(time_per_bit * self.sampling_rate)
            if bit == 1:
                # Добавление синусоиды для бита "1"
                signal[time_start:time_end] = \
                    self.amplitude * np.sin(2 * np.pi * self.carrier_frequency * time[time_start:time_end])
            else:
                # Установка значения сигнала равным половине амплитуды для бита "0"
                signal[time_start:time_end] = \
                    0.5 * self.amplitude * np.sin(2 * np.pi * self.carrier_frequency * time[time_start:time_end])
        return signal
    
    def generate_am_signal_pam(self,mod_depth,bit_count):
        total_parts = len(self.bits) // bit_count
        print(f"Частей: {total_parts}")
        total_time = 1 / self.bit_rate * total_parts
        packet_time = 1 / self.bit_rate
        print(f"Всего передача идет: {total_time}")
        time_pam = np.linspace(0, total_time, endpoint=True)
        print(f"Общее время: {time_pam[-1]}")
        signal = np.zeros(len(time_pam))
        mod_depth = 1 - mod_depth
        mod_levels = 2 ** bit_count
        levels = np.linspace(mod_depth,1,num=mod_levels)

        binary_sequence = list(product([0,1],repeat=bit_count))
        encoding_table = {binary_sequence[i]: round(levels[i],3) for i in range(0,len(levels))}

        for i in range(0,len(self.bits),bit_count):
            time_start = i * int(packet_time  * self.sampling_rate)
            time_end = (i + 1) * int(packet_time  * self.sampling_rate)
            bit_slice = self.bits[i:i+bit_count]
            m = encoding_table[tuple(bit_slice)]
            signal[time_start:time_end] = (
                m * self.amplitude * np.sin(2 * np.pi * self.carrier_frequency * time_pam[time_start:time_end]))
        return np.array(signal)
    
    def generate_psk_signal(self,phase0,phase1):
        # Пустой массив для сигнала
        time_per_bit = 1 / self.bit_rate
        time = np.linspace(0, len(self.bits) * time_per_bit, int(len(self.bits) * time_per_bit * self.sampling_rate), endpoint=False)
        signal = np.zeros(len(time))
 
        for i, bit in enumerate(self.bits):
            time_start = i * int(time_per_bit * self.sampling_rate)
            time_end = (i + 1) * int(time_per_bit * self.sampling_rate)
            if bit == 1:
                # Добавление синусоиды для бита "1"
                signal[time_start:time_end] = \
                    self.amplitude * np.sin(2 * np.pi * self.carrier_frequency * time[time_start:time_end] + phase1 * np.pi)
            else:
                # Установка значения сигнала равным половине амплитуды для бита "0"
                signal[time_start:time_end] = \
                    self.amplitude * np.sin(2 * np.pi * self.carrier_frequency * time[time_start:time_end] + phase0 * np.pi)
        return signal
    
    def generate_fm_signal(self,freq0,freq1):
        time_per_bit = 1 / self.bit_rate
        time = np.linspace(0, len(self.bits) * time_per_bit, int(len(self.bits) * time_per_bit * self.sampling_rate), endpoint=False)
        signal = np.zeros(len(time))
        for i, bit in enumerate(self.bits):
            time_start = i * int(time_per_bit * self.sampling_rate)
            time_end = (i + 1) * int(time_per_bit * self.sampling_rate)
            if bit == 1:
                # Добавление синусоиды для бита "1"
                signal[time_start:time_end] = \
                    self.amplitude * np.cos(2 * np.pi * freq1 * time[time_start:time_end])
            else:
                # Установка значения сигнала равным половине амплитуды для бита "0"
                signal[time_start:time_end] = \
                    self.amplitude * np.cos(2 * np.pi * freq0 * time[time_start:time_end])
        return signal

# Функция для сохранения сигнала в файл WAV
    def save_wav(self,filename, signal):
        normalized_signal = signal / np.max(np.abs(signal))  # Нормализация сигнала
        signal = (normalized_signal * 32767).astype(np.int16)  # Приведение к 16-битному целому типу
        with wave.open(filename, 'wb') as wave_file:
            wave_file.setnchannels(1)  # Один канал (моно)
            wave_file.setsampwidth(2)  # 2 байта на сэмпл (16 бит)
            wave_file.setframerate(self.sampling_rate)  # Частота дискретизации
            wave_file.writeframes(signal.tobytes())  # Запись сигнала в файл
