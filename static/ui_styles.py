from tkinter import ttk

def apply_styles(root, results_table, output):
    style = ttk.Style(root)

    # Загальний стиль
    style.theme_use("default")
    style.configure(".", font=("Segoe UI", 10))

    # --- Таблиця ---
    style.configure("Treeview",
                    background="white",
                    foreground="black",
                    rowheight=24,
                    fieldbackground="white",
                    font=("Segoe UI", 9))
    style.configure("Treeview.Heading",
                    font=("Segoe UI", 10, "bold"),
                    background="#f0f0f0")
    style.map("Treeview",
              background=[("selected", "#0078D7")],
              foreground=[("selected", "white")])

    # Чергування кольорів рядків
    results_table.tag_configure("oddrow", background="#f9f9f9")
    results_table.tag_configure("evenrow", background="white")

    # --- Лог ---
    output.configure(bg="#fafafa", fg="black", insertbackground="black")
    output.tag_configure("info", foreground="black")
    output.tag_configure("warn", foreground="orange", font=("Segoe UI", 9, "bold"))
    output.tag_configure("ok", foreground="green", font=("Segoe UI", 9, "bold"))

    # --- Кнопки ---
    style.configure("TButton",
                    font=("Segoe UI", 10, "bold"),
                    padding=6)
    style.map("TButton",
              background=[("active", "#0078D7")],
              foreground=[("active", "white")])