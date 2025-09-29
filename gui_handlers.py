import tkinter as tk
import threading
import csv
import os
from datasets import load_dataset
from static.mode_config import MODE_CONFIG

# Людяні назви етапів для показу користувачу
MODE_NAMES = {
    "features": "Відбір ознак",
    "params": "Параметричний синтез",
    "structure": "Структурний синтез",
    "opt": "Оптимізація структури"
}

def log(root, output, msg, tag=None):
    """Додає повідомлення у логове вікно GUI з потокобезпечним оновленням."""
    root.after(0, lambda: (output.insert(tk.END, msg + "\n", tag), output.see(tk.END)))

def format_result_compact(mode, result):
    """Компактний опис для таблиці: без зайвої технічної деталізації."""
    if isinstance(result, dict):
        if mode == "features":
            return f"Ознаки: {', '.join(result.get('features', []))}"
        elif mode == "params":
            return f"h={result.get('hidden')}, lr={result.get('lr')}, α={result.get('alpha')}"
        elif mode == "structure":
            return f"{result.get('layers')}×{result.get('neurons')}"
        else:
            return ""
    elif isinstance(result, list) and mode == "opt":
        return f"Парето‑рішень={len(result)}"
    return ""

def set_running(root, is_running=True):
    """Блокує/розблоковує кнопки під час виконання."""
    def _apply():
        if hasattr(root, "btn_run_one"):
            root.btn_run_one.config(state=("disabled" if is_running else "normal"))
        if hasattr(root, "btn_run_all"):
            root.btn_run_all.config(state=("disabled" if is_running else "normal"))
        if hasattr(root, "btn_save_csv"):
            root.btn_save_csv.config(state=("disabled" if is_running else "normal"))
        if hasattr(root, "btn_clear"):
            root.btn_clear.config(state=("disabled" if is_running else "normal"))
        if hasattr(root, "status_var"):
            root.status_var.set("Виконується…" if is_running else "Готово")
    root.after(0, _apply)

def insert_table_row(root, dataset, mode_desc, mae, rmse, extra):
    """Додає рядок у таблицю та оновлює підсвічування найкращого MAE."""
    def _ins():
        item_id = root.results_table.insert(
            "", "end",
            values=(dataset, mode_desc, f"{mae:.3f}", f"{rmse:.3f}", extra)
        )
        # Підсвічуємо найкращий MAE (мінімальний)
        try:
            items = root.results_table.get_children()
            maes = [(iid, float(root.results_table.set(iid, "mae"))) for iid in items]
            if maes:
                best_id = min(maes, key=lambda x: x[1])[0]
                # Скидаємо стиль усім
                for iid in items:
                    root.results_table.item(iid, tags=())
                # Призначаємо тег "best" найкращому
                root.results_table.item(best_id, tags=("best",))
                # Стиль тегу
                root.results_table.tag_configure("best", background="#e6ffea")  # легкий зелений
        except Exception:
            pass
    root.after(0, _ins)

def save_table_to_csv(root, path="results/summary.csv"):
    """Зберігає поточну таблицю результатів у CSV."""
    def _save():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cols = ("Енергосистема", "Етап аналізу", "MAE", "RMSE", "Додатково")
        rows = []
        for iid in root.results_table.get_children():
            vals = root.results_table.item(iid, "values")
            rows.append(vals)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            writer.writerows(rows)
        log(root, root.output, f"💾 Результати збережено у {path}", "ok")
    root.after(0, _save)

def clear_log_and_table(root):
    """Очищає лог і таблицю."""
    def _clear():
        root.output.delete("1.0", tk.END)
        for iid in root.results_table.get_children():
            root.results_table.delete(iid)
        if hasattr(root, "status_var"):
            root.status_var.set("Готово")
        log(root, root.output, "🧹 Очищено лог і таблицю", "info")
    root.after(0, _clear)

def run_algorithm(root, output, dataset_var, mode_var, gen_var, on_finish=None):
    """
    Запуск одного етапу аналізу для вибраної енергосистеми.
    Після завершення викликає on_finish (для послідовного запуску).
    """
    dataset = dataset_var.get()
    mode = mode_var.get()

    # Валідація кількості поколінь
    try:
        gens = int(gen_var.get())
        if gens <= 0:
            raise ValueError("Кількість поколінь має бути > 0")
    except Exception:
        log(root, output, "⚠️ Некоректна кількість поколінь. Введіть додатнє ціле число.", "warn")
        if on_finish:
            root.after(0, on_finish)
        return

    if mode not in MODE_CONFIG:
        log(root, output, f"⚠️ Невідомий етап '{mode}'", "warn")
        if on_finish:
            root.after(0, on_finish)
        return

    desc = MODE_NAMES.get(mode, mode)
    func = MODE_CONFIG[mode]["func"]

    set_running(root, True)
    log(root, output, f"⚡ Запуск етапу «{desc}» для енергосистеми {dataset}…", "info")

    def task():
        try:
            X, y, cols = load_dataset(dataset)
        except Exception as e:
            log(root, output, f"❌ Помилка завантаження даних: {e}", "error")
            set_running(root, False)
            if on_finish:
                root.after(0, on_finish)
            return

        # Прогрес у людяному форматі
        make_progress = lambda g, mae, rmse, extra: log(
            root, output,
            f"Покоління {g+1} з {gens}: середня похибка = {mae:.3f}, квадратична похибка = {rmse:.3f}. {extra}",
            "info"
        )

        # Виклик функції GA залежно від етапу
        try:
            if mode == "features":
                result = func(
                    X, y, cols,
                    pop_size=8,
                    n_gen=gens,
                    mutation_rate=0.2,
                    max_iter=100,
                    cv_splits=3,
                    progress_cb=make_progress
                )
            else:
                result = func(
                    X, y,
                    pop_size=8 if mode != "opt" else 10,
                    n_gen=gens,
                    mutation_rate=0.2,
                    max_iter=100,
                    cv_splits=3,
                    progress_cb=make_progress
                )
        except Exception as e:
            log(root, output, f"❌ Помилка виконання етапу «{desc}»: {e}", "error")
            set_running(root, False)
            if on_finish:
                root.after(0, on_finish)
            return

        # Підсумок + вставка в таблицю
        if isinstance(result, dict):
            mae = float(result.get("mae", 0.0))
            rmse = float(result.get("rmse", 0.0))
            extra = format_result_compact(mode, result)
            log(root, output, f"✅ Етап «{desc}» завершено. Найкраща модель: MAE={mae:.3f}, RMSE={rmse:.3f}", "ok")
            insert_table_row(root, dataset, desc, mae, rmse, extra)

        elif isinstance(result, list) and mode == "opt":
            log(root, output, f"✅ Етап «{desc}» завершено. Знайдено Парето‑рішень: {len(result)}", "ok")
            # Додаємо всі Парето‑рішення окремими рядками
            for p in result:
                mae = float(p.get("mae", 0.0))
                rmse = float(p.get("rmse", 0.0))
                extra = f"{p.get('layers')}×{p.get('neurons')}"
                insert_table_row(root, dataset, desc, mae, rmse, extra)
        else:
            log(root, output, f"ℹ️ Етап «{desc}» повернув нетиповий результат", "warn")

        set_running(root, False)
        if on_finish:
            root.after(0, on_finish)

    threading.Thread(target=task, daemon=True).start()

def run_all_modes(root, output, dataset_var, gen_var):
    """Запускає всі етапи послідовно: Відбір ознак → Параметри → Структура → Оптимізація."""
    modes = list(MODE_CONFIG.keys())

    def run_next(i=0):
        if i >= len(modes):
            log(root, output, "✅ Усі етапи аналізу завершені", "ok")
            return

        mode = modes[i]
        desc = MODE_NAMES.get(mode, mode)
        log(root, output, f"▶ Готуємо запуск етапу «{desc}»…", "info")

        run_algorithm(
            root, output, dataset_var,
            tk.StringVar(value=mode), gen_var,
            on_finish=lambda: run_next(i+1)
        )

    run_next()
