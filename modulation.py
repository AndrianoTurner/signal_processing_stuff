import numpy as np
import matplotlib.pyplot as plt
import wave
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from scipy.interpolate import interp1d
from itertools import combinations_with_replacement, product



class SignalProcessor():
    def __init__(self,bits,bit_rate,carrier_frequency,sampling_rate) -> None:
        self.bits = bits
        self.carrier_frequency = carrier_frequency
        self.bit_rate = bit_rate
        self.sampling_rate = sampling_rate
        self.time_per_bit = 1 / bit_rate
        self.time = np.linspace(0, len(self.bits) * self.time_per_bit, int(len(self.bits) * self.time_per_bit * self.sampling_rate), endpoint=False)
        self.amplitude = 100
        

    def generate_am_signal(self):
        # Пустой массив для сигнала
        signal = np.zeros(len(self.time))
       
        for i, bit in enumerate(self.bits):
            time_start = i * int(self.time_per_bit * self.sampling_rate)
            time_end = (i + 1) * int(self.time_per_bit * self.sampling_rate)
            if bit == 1:
                # Добавление синусоиды для бита "1"
                signal[time_start:time_end] = \
                    self.amplitude * np.sin(2 * np.pi * self.carrier_frequency * self.time[time_start:time_end])
            else:
                # Установка значения сигнала равным половине амплитуды для бита "0"
                signal[time_start:time_end] = \
                    0.5 * self.amplitude * np.sin(2 * np.pi * self.carrier_frequency * self.time[time_start:time_end])
        return signal
    
    def generate_am_signal_pam(self,mod_depth,bit_count,time,time_per_bit):
       
        signal = np.zeros(len(time))
        mod_depth = 1 - mod_depth
        mod_levels = 2 ** bit_count
        levels = np.linspace(mod_depth,1,num=mod_levels)

        binary_sequence = list(product([0,1],repeat=bit_count))
        encoding_table = {binary_sequence[i]: round(levels[i],3) for i in range(0,len(levels))}

        for i in range(0,len(self.bits),bit_count):
            time_start = i * int(time_per_bit * self.sampling_rate)
            time_end = (i + 1) * int(time_per_bit * self.sampling_rate)
            bit_slice = self.bits[i:i+bit_count]
            m = encoding_table[tuple(bit_slice)]
            signal[time_start:time_end] = \
                m * self.amplitude * np.sin(2 * np.pi * self.carrier_frequency * time[time_start:time_end])
        return signal
    
    def generate_psk_signal(self,phase0,phase1):
        # Пустой массив для сигнала
        signal = np.zeros(len(self.time))
 
        for i, bit in enumerate(self.bits):
            time_start = i * int(self.time_per_bit * self.sampling_rate)
            time_end = (i + 1) * int(self.time_per_bit * self.sampling_rate)
            if bit == 1:
                # Добавление синусоиды для бита "1"
                signal[time_start:time_end] = \
                    self.amplitude * np.sin(2 * np.pi * self.carrier_frequency * self.time[time_start:time_end] + phase1 * np.pi)
            else:
                # Установка значения сигнала равным половине амплитуды для бита "0"
                signal[time_start:time_end] = \
                    self.amplitude * np.sin(2 * np.pi * self.carrier_frequency * self.time[time_start:time_end] + phase0 * np.pi)
        return signal
    
    def generate_fm_signal(self,freq0,freq1):
        signal = np.zeros(len(self.time))
        for i, bit in enumerate(self.bits):
            time_start = i * int(self.time_per_bit * self.sampling_rate)
            time_end = (i + 1) * int(self.time_per_bit * self.sampling_rate)
            if bit == 1:
                # Добавление синусоиды для бита "1"
                signal[time_start:time_end] = \
                    self.amplitude * np.cos(2 * np.pi * freq1 * self.time[time_start:time_end])
            else:
                # Установка значения сигнала равным половине амплитуды для бита "0"
                signal[time_start:time_end] = \
                    self.amplitude * np.cos(2 * np.pi * freq0 * self.time[time_start:time_end])
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


    # Функция для построения визуализации сигнала
    def plot_waveform(self,signal,time,interpolated_signal,time_interp):


        fig,ax = plt.subplots(2,1,num=0,clear=True,sharex=True,sharey=True)
        ax[0].set_xlabel('Время (мс)')
        ax[0].set_ylabel('Амплитуда')

        ax[1].set_xlabel('Время (мс)')
        ax[1].set_ylabel('Амплитуда')

        ax[0].grid(True)
        ax[0].plot(time,signal,'b.', label='Дискретные отсчеты')


        ax[1].grid(True)
        ax[1].plot(time_interp,interpolated_signal,'r-', label='Интерполированный сигнал')

        fig.legend()

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

        processor = SignalProcessor(bits,500,2000,8000)
        bit_count = int(dropdown.get())
        total_parts = len(bits) // bit_count
        time_per_bit_pam = 1 / 500 * total_parts
        time_pam = np.linspace(0, len(processor.bits) * time_per_bit_pam, int(len(processor.bits) * time_per_bit_pam * processor.sampling_rate), endpoint=False)
        signal = processor.generate_am_signal_pam(mod_depth=0.75,bit_count=bit_count,time=time_pam,time_per_bit=time_per_bit_pam)
        time_pam = np.arange(0,len(signal)) * 1000 / processor.sampling_rate
        time_interp_pam = np.linspace(0, time_pam[-1], 10 * len(time_pam)) 
        interp_pam = interp1d(time_pam,signal,"cubic",bounds_error=False)(time_interp_pam)
        signal_am = processor.generate_am_signal()
        # signal_fm = processor.generate_fm_signal(800,1600)
        # signal_psk = processor.generate_psk_signal(0,1)

        time = np.arange(0, len(signal_am)) * 1000 / processor.sampling_rate
        time_interp = np.linspace(0, time[-1], 10 * len(time))  # Временная ось для интерполированного сигнала
        # # Интерполяция сигнала
        interpolator_am = interp1d(time,signal_am,'cubic',bounds_error=False)
        interpolated_am_signal = interpolator_am(time_interp)

        # interpolator_fm = interp1d(time,signal_fm,'cubic',bounds_error=False)
        # interpolated_fm_signal = interpolator_fm(time_interp)

        # interpolator_psk = interp1d(time,signal_psk,'cubic',bounds_error=False)
        # interpolated_psk_signal = interpolator_psk(time_interp)

        #save_wav('output.wav', signal, sampling_rate)

        #messagebox.showinfo("Успех", "Файл успешно сохранен как output.wav")
        processor.plot_waveform(signal, time_pam, interp_pam,time_interp_pam)
 #s       processor.plot_waveform(signal_fm, time, interpolated_fm_signal,time_interp)
        processor.plot_waveform(signal_am, time, interpolated_am_signal,time_interp)
#        processor.plot_waveform(signal_psk, time, interpolated_psk_signal,time_interp)

    except Exception as e:
        print(e)


# Создание графического интерфейса
app = tk.Tk()
app.title("Генератор WAV файла")

label = tk.Label(app, text="Введите битовую последовательность (12 бит):")
label.pack()

entry = tk.Entry(app)
entry.insert(0,"000110001100")
entry.pack()
options = ["1","2","4"]
dropdown = ttk.Combobox(values=options)
dropdown.set("1")
dropdown.pack()
button = tk.Button(app, text="Сгенерировать и сохранить", command=generate_and_save)
button.pack()

app.mainloop()
