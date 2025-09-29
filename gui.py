import tkinter as tk
from tkinter import ttk
from static.constants import APP_NAME, VERSION, BG_COLOR, FONT_MAIN
from static.mappings import DATASET_NAMES
from gui_handlers import (
    run_algorithm, run_all_modes, log, MODE_NAMES,
    save_table_to_csv, clear_log_and_table, set_running
)

# --- Головне вікно ---
root = tk.Tk()
root.title(f"{APP_NAME} {VERSION}")
root.configure(bg="#f5f5f5")  # світлий фон

# --- Стиль ---
style = ttk.Style()
style.theme_use("clam")

# Загальний стиль
style.configure("TLabel", font=("Segoe UI", 10), background="#f5f5f5")
style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
style.configure("Treeview", rowheight=25, font=("Consolas", 10))
style.map("Treeview", background=[("selected", "#cce5ff")])

# --- Статус-бар ---
root.status_var = tk.StringVar(value="Готово")
status_bar = ttk.Label(root, textvariable=root.status_var, anchor="w",
                       background="#e9ecef", font=("Segoe UI", 9))
status_bar.grid(row=3, column=0, columnspan=2, sticky="ew")

# --- Верхня панель ---
frame_top = ttk.Frame(root, padding=10)
frame_top.grid(row=0, column=0, columnspan=2, sticky="ew")

ttk.Label(frame_top, text="Енергосистема:", font=FONT_MAIN).grid(row=0, column=0, padx=5, pady=5)
dataset_var = tk.StringVar(value=next(iter(DATASET_NAMES.keys()), "s1"))
ttk.Combobox(frame_top, textvariable=dataset_var,
             values=list(DATASET_NAMES.keys()), width=25).grid(row=0, column=1, padx=5, pady=5)

ttk.Label(frame_top, text="Етап аналізу:", font=FONT_MAIN).grid(row=0, column=2, padx=5, pady=5)
mode_var = tk.StringVar(value="features")
ttk.Combobox(frame_top, textvariable=mode_var,
             values=list(MODE_NAMES.keys()), width=30).grid(row=0, column=3, padx=5, pady=5)

ttk.Label(frame_top, text="Кількість поколінь:", font=FONT_MAIN).grid(row=0, column=4, padx=5, pady=5)
gen_var = tk.StringVar(value="5")
ttk.Entry(frame_top, textvariable=gen_var, width=10).grid(row=0, column=5, padx=5, pady=5)

# --- Кнопки ---
frame_buttons = ttk.Frame(root, padding=10)
frame_buttons.grid(row=1, column=0, columnspan=2, sticky="ew")

btn_width = 24
root.btn_run_one = ttk.Button(frame_buttons, text="▶ Запустити етап",
                              command=lambda: run_algorithm(root, output, dataset_var, mode_var, gen_var),
                              width=btn_width)
root.btn_run_one.grid(row=0, column=0, padx=5)

root.btn_run_all = ttk.Button(frame_buttons, text="⏩ Запустити всі етапи",
                              command=lambda: run_all_modes(root, output, dataset_var, gen_var),
                              width=btn_width)
root.btn_run_all.grid(row=0, column=1, padx=5)

root.btn_save_csv = ttk.Button(frame_buttons, text="💾 Зберегти таблицю",
                               command=lambda: save_table_to_csv(root, "results/summary.csv"),
                               width=btn_width)
root.btn_save_csv.grid(row=0, column=2, padx=5)

root.btn_clear = ttk.Button(frame_buttons, text="🧹 Очистити",
                            command=lambda: clear_log_and_table(root),
                            width=btn_width)
root.btn_clear.grid(row=0, column=3, padx=5)

# --- Основні фрейми ---
frame_left = ttk.Frame(root, padding=10)
frame_left.grid(row=2, column=0, sticky="n")

frame_right = ttk.Frame(root, padding=10)
frame_right.grid(row=2, column=1, sticky="n")

root.frame_right = frame_right  # зарезервовано на майбутнє

# --- Лог ---
output = tk.Text(frame_left, width=88, height=18, font=FONT_MAIN, bg="#ffffff")
output.grid(row=0, column=0, padx=10, pady=10)
root.output = output  # для доступу з handler'ів

# --- Таблиця результатів ---
columns = ("dataset", "mode", "mae", "rmse", "extra")
results_table = ttk.Treeview(frame_left, columns=columns, show="headings", height=10)

results_table.heading("dataset", text="Енергосистема")
results_table.heading("mode", text="Етап аналізу")
results_table.heading("mae", text="MAE")
results_table.heading("rmse", text="RMSE")
results_table.heading("extra", text="Додатково")

results_table.column("dataset", width=120, anchor="center")
results_table.column("mode", width=180, anchor="center")
results_table.column("mae", width=90, anchor="center")
results_table.column("rmse", width=90, anchor="center")
results_table.column("extra", width=420, anchor="w")

results_table.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

# Горизонтальний скролбар
scroll_x = tk.Scrollbar(frame_left, orient="horizontal", command=results_table.xview)
results_table.configure(xscrollcommand=scroll_x.set)
scroll_x.grid(row=2, column=0, sticky="ew")

# Робимо таблицю доступною у gui_handlers
root.results_table = results_table

# --- Стартове повідомлення ---
log(root, output, "🔌 Система готова. Оберіть енергосистему, етап і кількість поколінь, потім натисніть «Запустити».", "info")
set_running(root, False)

root.mainloop()
