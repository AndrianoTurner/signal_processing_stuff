
# signal_processing_stuff

pip install -r requirements.txt

# Алгоритм

1. Ввод текстовой строки битового сообщения
1. Ввод значения глубины модуляции M %
1. Выбор количества бит для одного символа канала **ПДС**
1. Расчет амплитуды сигнала для каждого символа канала **ПДС** относительно M
1. Разбиение строки на фрагменты равные символу **ПДС**
1. Ввод частоты дискретизации
1. Ввод символьной скорости канала **ПДС** (например 500 символов в секунду)
1. Расчет времени передачи одного символа
1. Расчет времени полной передачи сообщения
1. Генерация массива с размерностью количества отсчетов ЦАП для времени равного общему времени.
1. Генерация массива размерностью пункту 10 и значениями абсолютного времени для каждого отсчета.
