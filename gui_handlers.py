import tkinter as tk
import threading
from datasets import load_dataset
from static.mode_config import MODE_CONFIG

def log(root, output, msg, tag=None):
    """Додає повідомлення у логове вікно GUI."""
    root.after(0, lambda: (output.insert(tk.END, msg + "\n", tag), output.see(tk.END)))

def format_result(mode, result):
    """Форматує результат у зручний для читання вигляд."""
    if not isinstance(result, dict):
        return str(result)

    if mode == "features":
        return (f"MAE={result['mae']:.3f}, RMSE={result['rmse']:.3f}, "
                f"Відібрано ознак={result['n_features']}, "
                f"Ознаки={', '.join(result['features'])}")

    elif mode == "params":
        return (f"MAE={result['mae']:.3f}, RMSE={result['rmse']:.3f}, "
                f"hidden={result['hidden']}, lr={result['lr']}, alpha={result['alpha']}")

    elif mode == "structure":
        return (f"MAE={result['mae']:.3f}, RMSE={result['rmse']:.3f}, "
                f"шарів={result['layers']}, нейронів={result['neurons']}")

    elif mode == "opt":
        return f"Парето‑рішень={len(result)}"

    return str(result)

def run_algorithm(root, output, dataset_var, mode_var, gen_var):
    """Запуск одного етапу аналізу."""
    dataset = dataset_var.get()
    mode = mode_var.get()
    gens = int(gen_var.get())

    if mode not in MODE_CONFIG:
        log(root, output, f"⚠️ Невідомий режим '{mode}'", "warn")
        return

    desc = MODE_CONFIG[mode]["desc"]
    func = MODE_CONFIG[mode]["func"]

    log(root, output, f"▶ Запуск аналізу ({mode}) – {desc} для енергосистеми {dataset}…", "info")

    def task():
        try:
            X, y, cols = load_dataset(dataset)
        except Exception as e:
            log(root, output, f"❌ Помилка завантаження даних: {e}", "error")
            return

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
                    f"[Gen {g+1}/{gens}] MAE={mae:.3f}, RMSE={rmse:.3f}, {extra}", "info"
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
                    f"[Gen {g+1}/{gens}] MAE={mae:.3f}, RMSE={rmse:.3f}, {extra}", "info"
                )
            )

        pretty = format_result(mode, result)
        log(root, output, f"✅ Завершено ({desc}): {pretty}", "ok")

        if hasattr(root, "results_table"):
            if isinstance(result, dict):
                root.after(0, lambda: root.results_table.insert(
                    "", "end",
                    values=(dataset,
                            mode,
                            f"{result.get('mae', 0):.3f}",
                            f"{result.get('rmse', 0):.3f}",
                            pretty)
                ))
            elif isinstance(result, list):
                for p in result:
                    root.after(0, lambda p=p: root.results_table.insert(
                        "", "end",
                        values=(dataset,
                                mode,
                                f"{p.get('mae', 0):.3f}",
                                f"{p.get('rmse', 0):.3f}",
                                f"{p['layers']}×{p['neurons']}")
                    ))

    threading.Thread(target=task, daemon=True).start()

def run_all_modes(root, output, dataset_var, gen_var):
    """Запуск усіх етапів аналізу послідовно."""
    modes = list(MODE_CONFIG.keys())

    def run_next(i=0):
        if i >= len(modes):
            log(root, output, "✅ Усі етапи аналізу завершені", "ok")
            return

        mode = modes[i]
        log(root, output, f"▶ Запуск етапу {mode}…", "info")

        def task():
            run_algorithm(root, output, dataset_var, tk.StringVar(value=mode), gen_var)
            root.after(200, lambda: run_next(i+1))

        threading.Thread(target=task, daemon=True).start()

    run_next()
