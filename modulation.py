import numpy as np
import matplotlib.pyplot as plt
import wave
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from scipy.interpolate import interp1d
from itertools import combinations_with_replacement, product
from SignalProcessor import SignalProcessor




# Функция для построения визуализации сигнала
def plot_waveform(signal,time,interpolated_signal,time_interp):


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
       
       
        signal = processor.generate_am_signal_pam(mod_depth=0.5,bit_count=bit_count)
        time_pam = np.arange(0,len(signal)) * 1000 / processor.sampling_rate
        time_interp_pam = np.linspace(0, time_pam[-1], 10 * len(time_pam)) 
        interp_pam = interp1d(time_pam,signal,"cubic",bounds_error=False)(time_interp_pam)

        plot_waveform(signal, time_pam, interp_pam,time_interp_pam)
    except Exception as e:
        print(e)


# Создание графического интерфейса
app = tk.Tk()
app.title("Генератор WAV файла")

label = tk.Label(app, text="Введите битовую последовательность (12 бит):")
label.pack()

entry = tk.Entry(app)
entry.insert(0,"111000110010")
entry.pack()
options = ["1","2","3","4"]
dropdown = ttk.Combobox(values=options)
dropdown.set("1")
dropdown.pack()
button = tk.Button(app, text="Сгенерировать и сохранить", command=generate_and_save)
button.pack()

app.mainloop()
