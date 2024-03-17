
import typing
import numpy as np
import wave
import matplotlib.pyplot as plt

class SignalProcessor():
    def __init__(self,bits : typing.List[int],bitrate : int,carrier : int,sampling_rate : int,qstep : int) -> None:
        self.bits = bits
        self.bitrate = bitrate
        self.carrier = carrier
        self.sampling_rate = sampling_rate
        self.qstep = qstep

    # Функция для генерации сигнала на основе битовой последовательности
    def generate_signal(self):
        # Вычисление времени на один бит
        time_per_bit = 1 / self.bitrate
        # Общее время сигнала
        total_time = len(self.bits) * time_per_bit
        # Создание временной оси
        time = np.linspace(0, total_time, int(total_time * self.sampling_rate))
        self.time = time
        # Амплитуда сигнала
        amplitude = 127
        signal = []
        mod = 0.5
        # Формирование сигнала для каждого бита

        for bit in self.bits:
            if bit == 1:
                # Добавление синусоиды для бита "1"
                signal += list(amplitude * np.sin(2 * np.pi * self.carrier * time[:int(time_per_bit * self.sampling_rate)]))
                markers.append(time[:int(time_per_bit * sampling_rate)])
            else:
                # Добавление нулевого сигнала для бита "0"
                signal += list(mod * amplitude * np.sin(2 * np.pi * self.carrier* time[:int(time_per_bit * self.sampling_rate)]))
                #signal += [0] * int(time_per_bit * sampling_rate)

        # Преобразование списка в массив NumPy и установка типа данных int8
        signal = np.array(signal, dtype=np.int8)

        # Квантование сигнала
        quantization_levels = 2**quantization_depth
        #signal = np.round(signal * (quantization_levels - 1) / amplitude)
        signal = np.clip(signal, -128, 127)

        return signal

    # Функция для сохранения сигнала в файл WAV
    def save_wav(self,filename, signal):
        with wave.open(filename, 'w') as wave_file:
            wave_file.setnchannels(1)
            wave_file.setsampwidth(2)
            wave_file.setframerate(self.sampling_rate)
            wave_file.writeframes(signal.tobytes())

    # Функция для построения визуализации сигнала
    def plot_waveform(self,signal):
        # time = np.arange(0, len(signal)) / self.sampling_rate * 1000
        plt.plot(self.time, signal)
       # plt.autoscale(True,"y",None)
        # plt.xlim((0,5))
        plt.xlabel('Время (мс)')
        plt.ylabel('Амплитуда')
        plt.show()

    def process(self):
        signal = self.generate_signal()
        self.plot_waveform(signal)

# Пример использования
bits = [1,1,0,0,1,1,1,0,1,0,1,0]  # Пример битовой последовательности
markers = []
bit_rate = 500  # бит в секунду
carrier_frequency = 2000  # Гц
sampling_rate = 8  # отсчетов в секунду
quantization_depth = 8  # бит

processor = SignalProcessor(bits,bit_rate,carrier_frequency,sampling_rate,quantization_depth)
processor.process()