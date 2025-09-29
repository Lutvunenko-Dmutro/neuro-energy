import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

def init_live_plot(root, mode="features"):
    """Ініціалізація графіка для різних режимів (викликати у головному потоці)"""
    fig, ax = plt.subplots(figsize=(5, 3))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().grid(row=7, column=0, columnspan=2, padx=5, pady=5)

    if mode in ("features", "params", "structure"):
        ax.set_title({
            "features": "Відбір ознак",
            "params": "Параметричний синтез",
            "structure": "Структурний синтез"
        }[mode])
        ax.set_xlabel("Покоління")
        ax.set_ylabel("MAE / RMSE")
        lines = {
            "mae": ax.plot([], [], label="MAE", color="blue")[0],
            "rmse": ax.plot([], [], label="RMSE", color="red")[0]
        }
        ax.legend()

    elif mode == "opt":
        ax.set_title("Парето-фронт")
        ax.set_xlabel("RMSE")
        ax.set_ylabel("MAE")
        lines = {"pareto": None}

    else:
        lines = {}

    return fig, ax, canvas, lines


def update_live_plot(ax, canvas, lines, history, mode="features"):
    """Оновлення графіка (викликати через root.after)"""
    if mode in ["features", "params", "structure"]:
        if not history:
            return
        gens = [h[0] for h in history]
        maes = [h[1] for h in history]
        rmses = [h[2] for h in history]

        lines["mae"].set_data(gens, maes)
        lines["rmse"].set_data(gens, rmses)

        ax.relim()
        ax.autoscale_view()

    elif mode == "opt":
        if not history:
            return
        maes = [h[1] for h in history]
        rmses = [h[2] for h in history]

        order = np.argsort(rmses)
        maes_sorted = np.array(maes)[order]
        rmses_sorted = np.array(rmses)[order]

        ax.clear()
        ax.set_title("Парето-фронт")
        ax.set_xlabel("RMSE")
        ax.set_ylabel("MAE")

        ax.scatter(rmses, maes, color="blue", label="Рішення")
        ax.plot(rmses_sorted, maes_sorted, "-o", color="red", label="Фронт")
        ax.legend()

    canvas.draw_idle()