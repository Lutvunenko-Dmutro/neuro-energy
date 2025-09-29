import tkinter as tk
import threading
import time
import csv
from tkinter import filedialog

from static.messages import START_MSG
from static.plots import init_live_plot, update_live_plot
from datasets import load_dataset
from static.mode_config import MODE_CONFIG
from static.mode_log import MODE_LOG

def log(root, output, msg, tag=None):
    root.after(0, lambda: (output.insert(tk.END, msg + "\n", tag), output.see(tk.END)))

def make_progress_callback(mode, history, ax_ref, canvas_ref, lines_ref, root, progress, output):
    def callback(gen, *metrics):
        # Отримуємо поточні посилання на графік (вони встановлюються після init через root.after)
        ax = ax_ref["val"]
        canvas = canvas_ref["val"]
        lines = lines_ref["val"]

        if metrics[0] is None:
            root.after(0, lambda: log(root, output, f"▶ Йде обчислення покоління {gen}…", "info"))
        else:
            if mode == "features":
                mae, rmse, nf = metrics
                root.after(0, lambda: log(root, output,
                                          f"Gen {gen}: MAE={mae:.3f}, RMSE={rmse:.3f}, features={nf}", "info"))
                history.append((gen, mae, rmse))
            elif mode == "params":
                mae, rmse, params_used = metrics
                root.after(0, lambda: log(root, output,
                                          f"Gen {gen}: MAE={mae:.3f}, RMSE={rmse:.3f}, params={params_used}", "info"))
                history.append((gen, mae, rmse))
            elif mode == "structure":
                mae, rmse, n_params, arch = metrics
                root.after(0, lambda: log(root, output,
                                          f"Gen {gen}: MAE={mae:.3f}, RMSE={rmse:.3f}, params={n_params}, arch={arch}", "info"))
                history.append((gen, mae, rmse))
            elif mode == "opt":
                mae, rmse, n_params, arch = metrics
                root.after(0, lambda: log(root, output,
                                          f"[Gen {gen}] MAE={mae:.3f}, RMSE={rmse:.3f}, params={n_params}, arch={arch}", "info"))
                history.append((gen, mae, rmse, n_params, arch))

            # Оновлення графіка та прогресу тільки через головний потік
            if ax is not None and canvas is not None and lines is not None:
                root.after(0, lambda: update_live_plot(ax, canvas, lines, history, mode=mode))
            root.after(0, lambda: progress.step(1))
        return True
    return callback

def add_result_to_table(table, dataset, mode, result):
    if table is None or result is None:
        return

    mae = f"{result.get('mae', 0):.3f}" if result.get("mae") is not None else "-"
    rmse = f"{result.get('rmse', 0):.3f}" if result.get("rmse") is not None else "-"
    elapsed = result.get("time", "-")

    if mode == "features":
        feats = result.get("features", [])
        n_feats = len(feats)
        feat_str = ", ".join(feats[:5]) + ("..." if len(feats) > 5 else "")
        table.insert("", "end", values=(dataset, mode, mae, rmse, n_feats, feat_str, elapsed))

    elif mode == "params":
        params = result.get("params", {})
        if isinstance(params, dict):
            params_str = ", ".join([f"{k}={v}" for k, v in params.items()]) if params else "-"
        elif isinstance(params, (list, tuple)):
            params_str = ", ".join(map(str, params)) if params else "-"
        else:
            params_str = str(params) if params else "-"
        table.insert("", "end", values=(dataset, mode, mae, rmse, params_str, elapsed))

    elif mode in ("structure", "opt"):
        arch = result.get("arch", [])
        n_params = result.get("n_params", "-")
        table.insert("", "end", values=(dataset, mode, mae, rmse, n_params, arch, elapsed))

    else:
        table.insert("", "end", values=(dataset, mode, mae, rmse, "-", "-", elapsed))

def run_algorithm(root, output, progress, dataset_var, mode_var, gen_var, params,
                  results_table=None, frame_right=None):
    dataset = dataset_var.get()
    mode = mode_var.get()
    gens = int(gen_var.get())
    X, y, cols = load_dataset(dataset)

    root.after(0, lambda: progress.config(maximum=gens, value=0))
    root.after(0, lambda: log(root, output, f"{START_MSG} {dataset} ({mode})", "info"))

    history = []

    # Посилання-контейнери для об'єктів графіка (заповняться у головному потоці)
    ax_ref = {"val": None}
    canvas_ref = {"val": None}
    lines_ref = {"val": None}

    # Створюємо графік ТІЛЬКИ у головному потоці
    def init_plot():
        fig, ax, canvas, lines = init_live_plot(frame_right if frame_right else root, mode=mode)
        ax_ref["val"] = ax
        canvas_ref["val"] = canvas
        lines_ref["val"] = lines
    root.after(0, init_plot)

    cb = make_progress_callback(mode, history, ax_ref, canvas_ref, lines_ref, root, progress, output)

    config = MODE_CONFIG.get(mode)
    if not config:
        root.after(0, lambda: log(root, output, f"❌ Невідомий режим: {mode}", "warn"))
        return

    common_args = dict(
        pop_size=params["POP_SIZE"],
        n_gen=params["GENERATIONS"],
        max_iter=params["MAX_ITER"],
        cv_splits=params["CV_SPLITS"],
        progress_cb=cb
    )

    ga_func = config["ga_func"]
    extra_args = config["extra_args"](params, cols)

    start_time = time.time()
    result = ga_func(X, y, **common_args, **extra_args)
    elapsed = round(time.time() - start_time, 2)

    if not result:
        root.after(0, lambda: log(root, output, "⚠️ Алгоритм не повернув результат", "warn"))
        return

    processed = config["postprocess"](result, cols)

    if config["multi"]:  # opt
        root.after(0, lambda: log(root, output, MODE_LOG[mode], "info"))
        for r in processed:
            r["time"] = elapsed
            msg = f"MAE={r['mae']:.3f}, RMSE={r['rmse']:.3f}, params={r['n_params']}, arch={r['arch']}"
            root.after(0, lambda m=msg: log(root, output, m, "info"))
            root.after(0, lambda r=r: add_result_to_table(results_table, dataset, mode, r))
    else:
        processed["time"] = elapsed
        root.after(0, lambda: log(root, output, MODE_LOG[mode].format(**processed), "info"))
        root.after(0, lambda: add_result_to_table(results_table, dataset, mode, processed))

    root.after(0, lambda: log(root, output, f"⏱ Час виконання: {elapsed:.2f} сек", "ok"))

def run_all_modes(root, output, progress, dataset_var, gen_var, params, results_table, frame_right=None):
    modes = list(MODE_CONFIG.keys())

    def run_next(i=0):
        if i >= len(modes):
            root.after(0, lambda: log(root, output, "✅ Усі режими завершені", "ok"))
            return

        mode = modes[i]

        def task():
            run_algorithm(root, output, progress,
                          dataset_var, tk.StringVar(value=mode),
                          gen_var, params, results_table, frame_right)
            root.after(200, lambda: run_next(i+1))

        threading.Thread(target=task, daemon=True).start()

    run_next()

def export_results_to_csv(table):
    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        title="Зберегти результати"
    )
    if not file_path:
        return

    cols = table["columns"]
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for row_id in table.get_children():
            row = table.item(row_id)["values"]
            writer.writerow(row)