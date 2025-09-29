import tkinter as tk
from tkinter import ttk
import threading
import warnings
from sklearn.exceptions import ConvergenceWarning

from static.constants import APP_NAME, VERSION
from static.styles import BG_COLOR, FONT_MAIN
from static.mappings import DATASET_NAMES, MODE_NAMES
from static.ga_defaults import FAST_MODE, FULL_MODE
from static.table_columns import TABLE_COLUMNS
from static.ui_styles import apply_styles
from gui_handlers import run_algorithm, run_all_modes, log

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --- Функції ---
def configure_table_for_mode(table, mode):
    """Перебудовує таблицю під вибраний режим"""
    for col in table["columns"]:
        table.heading(col, text="")
        table.column(col, width=0)
    table["columns"] = []

    cols = TABLE_COLUMNS.get(mode, [])
    table["columns"] = [c[0] for c in cols]
    for key, label in cols:
        table.heading(key, text=label)
        table.column(key, width=120, anchor="center")

def run_in_thread():
    output.delete("1.0", tk.END)
    params = FAST_MODE if fast_mode_var.get() else FULL_MODE
    t = threading.Thread(
        target=lambda: run_algorithm(root, output, progress,
                                     dataset_var, mode_var, gen_var,
                                     params, results_table, frame_right)
    )
    t.start()

def on_mode_change(event=None):
    configure_table_for_mode(results_table, mode_var.get())

# --- GUI ---
root = tk.Tk()
root.title(f"{APP_NAME} v{VERSION}")
root.configure(bg=BG_COLOR)

# --- Верхня панель (налаштування) ---
frame_top = ttk.Frame(root, padding=10)
frame_top.grid(row=0, column=0, columnspan=2, sticky="ew")

ttk.Label(frame_top, text="Датасет:", font=FONT_MAIN).grid(row=0, column=0, padx=5, pady=5, sticky="w")
dataset_var = tk.StringVar(value="s1")
ttk.Combobox(frame_top, textvariable=dataset_var, values=list(DATASET_NAMES.keys()), width=15).grid(row=0, column=1, padx=5, pady=5)

ttk.Label(frame_top, text="Режим:", font=FONT_MAIN).grid(row=0, column=2, padx=5, pady=5, sticky="w")
mode_var = tk.StringVar(value="features")
mode_menu = ttk.Combobox(frame_top, textvariable=mode_var, values=list(MODE_NAMES.keys()), width=15)
mode_menu.grid(row=0, column=3, padx=5, pady=5)
mode_menu.bind("<<ComboboxSelected>>", on_mode_change)

ttk.Label(frame_top, text="Покоління:", font=FONT_MAIN).grid(row=0, column=4, padx=5, pady=5, sticky="w")
gen_var = tk.StringVar(value="5")
ttk.Entry(frame_top, textvariable=gen_var, width=10).grid(row=0, column=5, padx=5, pady=5)

fast_mode_var = tk.BooleanVar(value=True)
ttk.Checkbutton(frame_top, text="⚡ Fast mode", variable=fast_mode_var).grid(row=0, column=6, padx=10, pady=5)

# --- Панель кнопок ---
frame_buttons = ttk.Frame(root, padding=10)
frame_buttons.grid(row=1, column=0, columnspan=2, sticky="ew")

btn_width = 20
ttk.Button(frame_buttons, text="▶ Запустити", command=run_in_thread, width=btn_width).grid(row=0, column=0, padx=5)
ttk.Button(frame_buttons, text="⏩ Запустити всі режими",
           command=lambda: run_all_modes(root, output, progress,
                                         dataset_var, gen_var,
                                         FAST_MODE if fast_mode_var.get() else FULL_MODE,
                                         results_table, frame_right),
           width=btn_width).grid(row=0, column=1, padx=5)

# --- Прогресбар ---
progress = ttk.Progressbar(root, orient="horizontal", length=500, mode="determinate")
progress.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

# --- Основні фрейми ---
frame_left = ttk.Frame(root, padding=10)
frame_left.grid(row=3, column=0, sticky="n")

frame_right = ttk.Frame(root, padding=10)
frame_right.grid(row=3, column=1, sticky="n")

# Лог
output = tk.Text(frame_left, width=70, height=18, font=FONT_MAIN)
output.grid(row=0, column=0, padx=5, pady=5)

# Таблиця
results_table = ttk.Treeview(frame_left, show="headings", height=10)
results_table.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

# Стилі
apply_styles(root, results_table, output)

# Початкова конфігурація таблиці
configure_table_for_mode(results_table, mode_var.get())

root.mainloop()