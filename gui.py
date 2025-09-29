import tkinter as tk
from tkinter import ttk
from static.constants import APP_NAME, VERSION, BG_COLOR, FONT_MAIN
from static.mappings import DATASET_NAMES
from gui_handlers import run_algorithm, run_all_modes, log, MODE_NAMES

# --- GUI ---
root = tk.Tk()
root.title(f"{APP_NAME} {VERSION}")
root.configure(bg=BG_COLOR)

# --- Верхня панель ---
frame_top = ttk.Frame(root, padding=10)
frame_top.grid(row=0, column=0, columnspan=2, sticky="ew")

ttk.Label(frame_top, text="Енергосистема:", font=FONT_MAIN).grid(row=0, column=0, padx=5, pady=5)
dataset_var = tk.StringVar(value="s1")
ttk.Combobox(frame_top, textvariable=dataset_var, values=list(DATASET_NAMES.keys()), width=25).grid(row=0, column=1, padx=5, pady=5)

ttk.Label(frame_top, text="Етап аналізу:", font=FONT_MAIN).grid(row=0, column=2, padx=5, pady=5)
mode_var = tk.StringVar(value="features")
# Використовуємо зрозумілі назви режимів
ttk.Combobox(frame_top, textvariable=mode_var, values=list(MODE_NAMES.keys()), width=30).grid(row=0, column=3, padx=5, pady=5)

ttk.Label(frame_top, text="Кількість поколінь:", font=FONT_MAIN).grid(row=0, column=4, padx=5, pady=5)
gen_var = tk.StringVar(value="5")
ttk.Entry(frame_top, textvariable=gen_var, width=10).grid(row=0, column=5, padx=5, pady=5)

# --- Кнопки ---
frame_buttons = ttk.Frame(root, padding=10)
frame_buttons.grid(row=1, column=0, columnspan=2, sticky="ew")

btn_width = 25
ttk.Button(frame_buttons, text="▶ Запустити аналіз",
           command=lambda: run_algorithm(root, output, dataset_var, mode_var, gen_var),
           width=btn_width).grid(row=0, column=0, padx=5)

ttk.Button(frame_buttons, text="⏩ Запустити всі етапи",
           command=lambda: run_all_modes(root, output, dataset_var, gen_var),
           width=btn_width).grid(row=0, column=1, padx=5)

# --- Основні фрейми ---
frame_left = ttk.Frame(root, padding=10)
frame_left.grid(row=2, column=0, sticky="n")

frame_right = ttk.Frame(root, padding=10)
frame_right.grid(row=2, column=1, sticky="n")

root.frame_right = frame_right  # для можливого використання

# --- Лог ---
output = tk.Text(frame_left, width=80, height=15, font=FONT_MAIN)
output.grid(row=0, column=0, padx=10, pady=10)

# --- Таблиця результатів ---
columns = ("dataset", "mode", "mae", "rmse", "extra")
results_table = ttk.Treeview(frame_left, columns=columns, show="headings", height=8)

results_table.heading("dataset", text="Енергосистема")
results_table.heading("mode", text="Етап аналізу")
results_table.heading("mae", text="MAE")
results_table.heading("rmse", text="RMSE")
results_table.heading("extra", text="Додатково")

results_table.column("dataset", width=120, anchor="center")
results_table.column("mode", width=200, anchor="center")
results_table.column("mae", width=80, anchor="center")
results_table.column("rmse", width=80, anchor="center")
results_table.column("extra", width=400, anchor="w")

results_table.grid(row=1, column=0, padx=10, pady=10)

# Горизонтальний скролбар
scroll_x = tk.Scrollbar(frame_left, orient="horizontal", command=results_table.xview)
results_table.configure(xscrollcommand=scroll_x.set)
scroll_x.grid(row=2, column=0, sticky="ew")

# Робимо таблицю доступною у gui_handlers
root.results_table = results_table

# Стартове повідомлення
log(root, output, "🔌 Система моніторингу навантаження енергосистеми готова до роботи", "info")

root.mainloop()
