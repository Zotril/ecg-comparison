# ECG Signal Processing

Цей проєкт аналізує сигнали ЕКГ за допомогою різних алгоритмів детекції R-вершин. Він дозволяє обробляти дані з двох різних баз даних, порівнювати результати детекції та зберігати графіки для подальшого аналізу.

## Вимоги до середовища

Для запуску проєкту вам необхідно:

1. **Python:** 3.8 або вище.
2. **Бібліотеки:** Встановіть залежності, що вказані нижче
**numpy**
**matplotlib**
**scipy**
**pandas**
**wfdb**
**pywavelets**
**ecg-detectors**

## Структура проєкту

- **`datasets/`**: Містить `.tsv` файли із сигналами ЕКГ для тестування.
- **`results/`**: Приклади результатів виконання обробки сигналів.=
- **`peaks.py`**: Додатковий функціонал для анотування R-вершин власноруч, якщо анотованих міток немає.
- **`main.py`**: Головний файл, який запускає весь функціонал проєкту.

## Як використовувати

1. **Запуск алгоритму для аналізу сигналу:**
   - У файлі `main.py` закоментовано кілька блоків коду, кожен з яких відповідає за різні алгоритми детекції R-вершин, наприклад: `Pan-Tompkins`, `Hamilton`, `Engzee`, тощо.
   - Для виконання аналізу на одному або кількох наборах даних розкоментуйте відповідні алгоритми як вгорі, так і внизу файлу `main.py`.

2. **Приклад виконання:**
   - Використання аналізу для запуску тестового сигналу з датасету https://www.physionet.org/content/butqdb/1.0.0/ 
     `Christov algorithm`:
     ```python
     r_peaks1 = detectors1.christov_detector(unfiltered_ecg)
     ```
   - Використання аналізу для запуску на трьох різних сигналах `Normal`, `MI`, `Hyp` 
   `Christov algorithm`
   ```python
   #-----------------Christov--------------------------------------

   def filter_raw_ecg(raw_ecg):
       total_taps = 0
   
       b = np.ones(int(0.02*fs))
       b = b/int(0.02*fs)
       total_taps += len(b)
       a = [1]
   
       MA1 = signal.lfilter(b, a, raw_ecg)
   
       b = np.ones(int(0.028*fs))
       b = b/int(0.028*fs)
       total_taps += len(b)
       a = [1]
   
       MA2 = signal.lfilter(b, a, MA1)
   
       Y = []
       for i in range(1, len(MA2)-1):
           
           diff = abs(MA2[i+1]-MA2[i-1])
   
           Y.append(diff)
   
       b = np.ones(int(0.040*fs))
       b = b/int(0.040*fs)
       total_taps += len(b)
       a = [1]
   
       MA3 = signal.lfilter(b, a, Y)
   
       MA3[0:total_taps] = 0
       return MA3
   
   filtered_normal_ecg = filter_raw_ecg(raw_normal_ecg)
   filtered_mi_ecg = filter_raw_ecg(raw_mi_ecg)
   filtered_hyp_ecg = filter_raw_ecg(raw_hyp_ecg)
   
   normal_r_peaks = detectors.christov_detector(raw_normal_ecg)
   mi_r_peaks = detectors.christov_detector(raw_mi_ecg)
   hyp_r_peaks = detectors.christov_detector(raw_hyp_ecg)
   
   #---------------------------------------------------------------------
   ```
   - Ананлогічне використання для інших алгоритмів: `pan_tompkins_detector`, `hamilton_detector`, `engzee_detector` і т.д.
   - Також є можливість залишити щось одне, закоментовуючи все інше.

3. **Дані:**
   - Перший набір даних береться з файлу `datasets/ecg.tsv`. Переконайтеся, що сигнал відповідає частоті 250 Hz або 360 Hz.
   - Другий набір треба завантажити 1.7 ГБ датасету з https://physionet.org/content/ptb-xl/1.0.3/#files-panel. Після завантаження та розархівації датасету необхідно подати path у локальну змінну з         файлу main.py 81 рядок data_path = r'put_path_to_dataset'
   - Якщо частота відрізняється, використовуйте додаткову обробку у `templates.py`.
   - Код дозволяє порівняти результати детекції R-вершин на різних наборах даних (наприклад, `Normal`, `MI`, `Hyp`). Розкоментуйте відповідні алгоритми в секціях для кожної бази.

4. **Приклади роботи:**
   - Результати аналізу зберігаються у папці `results/`.

## Приклад запуску

Для запуску основного коду:
У моєму випадку py, python та python3 вказують на одну версію 3.13 тому необхідно зважати на ваші налаштування.
```bash
py .\main.py
```

Графіки результатів будуть автоматично відображені, а також їх можна зберегти або у папці `results/`, або у іншому місці.
