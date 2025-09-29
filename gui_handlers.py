import tkinter as tk
import threading
from datasets import load_dataset
from static.mode_config import MODE_CONFIG

# Людяні назви режимів
MODE_NAMES = {
    "features": "Відбір ознак",
    "params": "Параметричний синтез",
    "structure": "Структурний синтез",
    "opt": "Оптимізація структури"
}

def log(root, output, msg, tag=None):
    """Додає повідомлення у логове вікно GUI."""
    root.after(0, lambda: (output.insert(tk.END, msg + "\n", tag), output.see(tk.END)))

def format_result(mode, result):
    """Форматує результат у зручний для читання вигляд."""
    if not isinstance(result, dict):
        return str(result)

    if mode == "features":
        return f"Ознаки: {', '.join(result['features'])}"

    elif mode == "params":
        return f"h={result['hidden']}, lr={result['lr']}, α={result['alpha']}"

    elif mode == "structure":
        return f"{result['layers']}×{result['neurons']}"

    elif mode == "opt":
        return f"Парето‑рішень={len(result)}"

    return str(result)

def run_algorithm(root, output, dataset_var, mode_var, gen_var, on_finish=None):
    """
    Запуск одного етапу аналізу.
    Після завершення викликає on_finish (для послідовного запуску).
    """
    dataset = dataset_var.get()
    mode = mode_var.get()
    gens = int(gen_var.get())

    if mode not in MODE_CONFIG:
        log(root, output, f"⚠️ Невідомий режим '{mode}'", "warn")
        if on_finish:
            root.after(0, on_finish)
        return

    desc = MODE_NAMES.get(mode, mode)
    func = MODE_CONFIG[mode]["func"]

    log(root, output, f"⚡ Запуск етапу «{desc}» для енергосистеми {dataset}…", "info")

    def task():
        try:
            X, y, cols = load_dataset(dataset)
        except Exception as e:
            log(root, output, f"❌ Помилка завантаження даних: {e}", "error")
            if on_finish:
                root.after(0, on_finish)
            return

        # Виклик функції GA залежно від режиму
        if mode == "features":
            result = func(
                X, y, cols,
                pop_size=8,
                n_gen=gens,
                mutation_rate=0.2,
                max_iter=100,
                cv_splits=3,
                progress_cb=lambda g, mae, rmse, extra: log(
                    root, output,
                    f"Покоління {g+1} з {gens}: середня похибка = {mae:.3f}, квадратична похибка = {rmse:.3f}. {extra}",
                    "info"
                )
            )
        else:
            result = func(
                X, y,
                pop_size=8 if mode != "opt" else 10,
                n_gen=gens,
                mutation_rate=0.2,
                max_iter=100,
                cv_splits=3,
                progress_cb=lambda g, mae, rmse, extra: log(
                    root, output,
                    f"Покоління {g+1} з {gens}: середня похибка = {mae:.3f}, квадратична похибка = {rmse:.3f}. {extra}",
                    "info"
                )
            )

        # Підсумок
        log(root, output, f"✅ Етап «{desc}» завершено.", "ok")

        # Додаємо результат у таблицю
        if hasattr(root, "results_table"):
            if isinstance(result, dict):
                root.after(0, lambda: root.results_table.insert(
                    "", "end",
                    values=(dataset,
                            desc,
                            f"{result.get('mae', 0):.3f}",
                            f"{result.get('rmse', 0):.3f}",
                            format_result(mode, result))
                ))
            elif isinstance(result, list):
                for p in result:
                    root.after(0, lambda p=p: root.results_table.insert(
                        "", "end",
                        values=(dataset,
                                desc,
                                f"{p.get('mae', 0):.3f}",
                                f"{p.get('rmse', 0):.3f}",
                                f"{p['layers']}×{p['neurons']}")
                    ))

        # Викликаємо callback для переходу до наступного етапу
        if on_finish:
            root.after(0, on_finish)

    threading.Thread(target=task, daemon=True).start()

def run_all_modes(root, output, dataset_var, gen_var):
    """Запуск усіх етапів аналізу послідовно."""
    modes = list(MODE_CONFIG.keys())

    def run_next(i=0):
        if i >= len(modes):
            log(root, output, "✅ Усі етапи аналізу завершені", "ok")
            return

        mode = modes[i]
        desc = MODE_NAMES.get(mode, mode)
        log(root, output, f"▶ Запуск етапу «{desc}»…", "info")

        run_algorithm(
            root, output, dataset_var,
            tk.StringVar(value=mode), gen_var,
            on_finish=lambda: run_next(i+1)
        )

    run_next()
