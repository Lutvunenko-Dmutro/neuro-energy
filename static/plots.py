"""
Модуль для побудови графіків у GUI.
"""

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

def plot_pareto_front(root, frame, pareto_front):
    """
    Будує графік Парето‑фронту (MAE vs RMSE).
    """
    # Очистимо попередній графік
    for widget in frame.winfo_children():
        widget.destroy()

    if not pareto_front:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    maes = [p["mae"] for p in pareto_front]
    rmses = [p["rmse"] for p in pareto_front]
    labels = [f"{p['layers']}×{p['neurons']}" for p in pareto_front]

    ax.scatter(maes, rmses, c="blue", marker="o")
    for i, label in enumerate(labels):
        ax.text(maes[i], rmses[i], label, fontsize=8, ha="right")

    ax.set_xlabel("MAE (середня абсолютна похибка)")
    ax.set_ylabel("RMSE (середньоквадратична похибка)")
    ax.set_title("Парето‑фронт: компроміс між точністю та складністю")

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)