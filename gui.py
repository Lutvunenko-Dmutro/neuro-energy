import tkinter as tk                          # Імпортуємо базовий модуль Tkinter і даємо йому псевдонім tk
from tkinter import ttk                       # Імпортуємо ttk (стилізовані віджети Tkinter)
from static.constants import APP_NAME, VERSION, BG_COLOR, FONT_MAIN  # Константи додатка (назва, версія, кольори, шрифт)
from static.mappings import DATASET_NAMES     # Мапінг назв датасетів (ключі для комбобоксу)
from gui_handlers import (                    # Імпортуємо функції обробників GUI
    run_algorithm, run_all_modes, log, MODE_NAMES,
    save_table_to_csv, clear_log_and_table, set_running
)

# --- Головне вікно ---
root = tk.Tk()                                 # Створюємо головне вікно додатка
root.title(f"{APP_NAME} {VERSION}")            # Встановлюємо заголовок вікна з назвою та версією
root.configure(bg="#f5f5f5")                   # Встановлюємо світлий фон для всього вікна

# --- Стиль ---
style = ttk.Style()                            # Створюємо стиль ttk
style.theme_use("clam")                        # Вмикаємо тему "clam" для сучасного вигляду

# Загальний стиль
style.configure("TLabel", font=("Segoe UI", 10), background="#f5f5f5")  # Стиль для усіх Label: шрифт + фон
style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)    # Стиль для кнопок: жирний шрифт + падінг
style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))      # Стиль заголовків таблиці: жирний шрифт
style.configure("Treeview", rowheight=25, font=("Consolas", 10))        # Стиль рядків таблиці: висота + моноширинний шрифт
style.map("Treeview", background=[("selected", "#cce5ff")])             # Колір виділеного рядка у таблиці

# --- Статус-бар ---
root.status_var = tk.StringVar(value="Готово") # Змінна стану (рядок), за замовчуванням "Готово"
status_bar = ttk.Label(                        # Створюємо статус-бар як Label із прив'язкою до змінної
    root, textvariable=root.status_var, anchor="w",
    background="#e9ecef", font=("Segoe UI", 9)
)
status_bar.grid(row=3, column=0, columnspan=2, sticky="ew")             # Розміщуємо нижче на всю ширину (липне по осі X)

# --- Верхня панель ---
frame_top = ttk.Frame(root, padding=10)        # Контейнер для верхніх контролів із внутрішнім відступом
frame_top.grid(row=0, column=0, columnspan=2, sticky="ew")              # Розміщуємо у верхньому рядку, розтягуємо по ширині

ttk.Label(frame_top, text="Енергосистема:", font=FONT_MAIN).grid(       # Підпис для вибору датасету
    row=0, column=0, padx=5, pady=5
)
dataset_var = tk.StringVar(value=next(iter(DATASET_NAMES.keys()), "s1"))# Змінна для вибраного датасету, дефолтно перший ключ
ttk.Combobox(                                                           # Комбобокс для вибору датасету
    frame_top, textvariable=dataset_var,
    values=list(DATASET_NAMES.keys()), width=25
).grid(row=0, column=1, padx=5, pady=5)

ttk.Label(frame_top, text="Етап аналізу:", font=FONT_MAIN).grid(        # Підпис для вибору етапу GA
    row=0, column=2, padx=5, pady=5
)
mode_var = tk.StringVar(value="features")                               # Змінна для вибраного режиму (за замовчуванням "features")
ttk.Combobox(                                                           # Комбобокс для вибору режиму за технічними ключами
    frame_top, textvariable=mode_var,
    values=list(MODE_NAMES.keys()), width=30
).grid(row=0, column=3, padx=5, pady=5)

ttk.Label(frame_top, text="Кількість поколінь:", font=FONT_MAIN).grid(  # Підпис для поля кількості поколінь
    row=0, column=4, padx=5, pady=5
)
gen_var = tk.StringVar(value="5")                                       # Змінна для кількості поколінь (рядок, конвертуємо пізніше)
ttk.Entry(frame_top, textvariable=gen_var, width=10).grid(              # Поле вводу кількості поколінь
    row=0, column=5, padx=5, pady=5
)

# --- Кнопки ---
frame_buttons = ttk.Frame(root, padding=10)                             # Контейнер для кнопок керування
frame_buttons.grid(row=1, column=0, columnspan=2, sticky="ew")          # Розміщення під верхньою панеллю по ширині

btn_width = 24                                                          # Єдина ширина для кнопок (для акуратного вирівнювання)
root.btn_run_one = ttk.Button(                                          # Кнопка запуску одного етапу
    frame_buttons, text="▶ Запустити етап",
    command=lambda: run_algorithm(root, output, dataset_var, mode_var, gen_var),
    width=btn_width
)
root.btn_run_one.grid(row=0, column=0, padx=5)                          # Розміщення кнопки з відступом

root.btn_run_all = ttk.Button(                                          # Кнопка запуску всіх етапів послідовно
    frame_buttons, text="⏩ Запустити всі етапи",
    command=lambda: run_all_modes(root, output, dataset_var, gen_var),
    width=btn_width
)
root.btn_run_all.grid(row=0, column=1, padx=5)

root.btn_save_csv = ttk.Button(                                         # Кнопка збереження таблиці у CSV
    frame_buttons, text="💾 Зберегти таблицю",
    command=lambda: save_table_to_csv(root, "results/summary.csv"),
    width=btn_width
)
root.btn_save_csv.grid(row=0, column=2, padx=5)

root.btn_clear = ttk.Button(                                            # Кнопка очищення логу і таблиці
    frame_buttons, text="🧹 Очистити",
    command=lambda: clear_log_and_table(root),
    width=btn_width
)
root.btn_clear.grid(row=0, column=3, padx=5)

# --- Основні фрейми ---
frame_left = ttk.Frame(root, padding=10)                                # Лівий контейнер (лог + таблиця)
frame_left.grid(row=2, column=0, sticky="n")                            # Розміщення з прилипанням до верхньої сторони

frame_right = ttk.Frame(root, padding=10)                               # Правий контейнер (резерв на майбутнє)
frame_right.grid(row=2, column=1, sticky="n")                           # Розміщення праворуч
root.frame_right = frame_right                                          # Зберігаємо посилання у root для можливого використання

# --- Лог ---
output = tk.Text(                                                       # Текстове поле для логів
    frame_left, width=88, height=18, font=FONT_MAIN, bg="#ffffff"
)
output.grid(row=0, column=0, padx=10, pady=10)                          # Розміщення з відступами
root.output = output                                                    # Зберігаємо посилання у root (щоб доступитись з handler'ів)

# --- Таблиця результатів ---
columns = ("dataset", "mode", "mae", "rmse", "extra")                   # Визначаємо імена колонок таблиці
results_table = ttk.Treeview(                                           # Створюємо таблицю (Treeview) з головами колонок
    frame_left, columns=columns, show="headings", height=10
)

results_table.heading("dataset", text="Енергосистема")                  # Заголовок колонки "dataset"
results_table.heading("mode", text="Етап аналізу")                      # Заголовок колонки "mode"
results_table.heading("mae", text="MAE")                                # Заголовок колонки "mae"
results_table.heading("rmse", text="RMSE")                              # Заголовок колонки "rmse"
results_table.heading("extra", text="Додатково")                        # Заголовок колонки "extra"

results_table.column("dataset", width=120, anchor="center")             # Налаштування ширини та вирівнювання колонки
results_table.column("mode", width=180, anchor="center")
results_table.column("mae", width=90, anchor="center")
results_table.column("rmse", width=90, anchor="center")
results_table.column("extra", width=420, anchor="w")

results_table.grid(row=1, column=0, padx=10, pady=10, sticky="ew")      # Розміщення таблиці, розтягуємо по ширині

# Горизонтальний скролбар
scroll_x = tk.Scrollbar(                                                # Створюємо горизонтальний скролбар
    frame_left, orient="horizontal", command=results_table.xview
)
results_table.configure(xscrollcommand=scroll_x.set)                    # Прив'язуємо скролбар до таблиці
scroll_x.grid(row=2, column=0, sticky="ew")                             # Розміщуємо під таблицею, розтягуємо по ширині

# Робимо таблицю доступною у gui_handlers
root.results_table = results_table                                      # Зберігаємо посилання у root, щоб обробники могли вставляти рядки

# --- Стартове повідомлення ---
log(root, output,                                                       # Початковий лог, який пояснює користувачу порядок дій
    "🔌 Система готова. Оберіть енергосистему, етап і кількість поколінь, потім натисніть «Запустити».",
    "info"
)
set_running(root, False)                                                # Встановлюємо стан "Готово" (кнопки активні)

root.mainloop()                                                         # Запускаємо головний цикл Tkinter (обробка подій, відображення GUI)
