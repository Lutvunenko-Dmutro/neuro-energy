import tkinter as tk          # Імпортуємо бібліотеку Tkinter для створення GUI, скорочуємо ім'я до tk
import threading              # Імпортуємо модуль для роботи з потоками (щоб GUI не зависав під час обчислень)
import csv                    # Модуль для роботи з CSV-файлами (збереження результатів у таблицю)
import os                     # Модуль для роботи з файловою системою (створення папок, шляхи)
from datasets import load_dataset          # Імпортуємо функцію load_dataset з твого модуля datasets.py
from static.mode_config import MODE_CONFIG # Імпортуємо словник MODE_CONFIG з static/mode_config.py (налаштування режимів)

# Словник з "людяними" назвами етапів для відображення у GUI
MODE_NAMES = {
    "features": "Відбір ознак",             # Технічний ключ "features" → показуємо користувачу "Відбір ознак"
    "params": "Параметричний синтез",       # "params" → "Параметричний синтез"
    "structure": "Структурний синтез",      # "structure" → "Структурний синтез"
    "opt": "Оптимізація структури"          # "opt" → "Оптимізація структури"
}

def log(root, output, msg, tag=None):
    """
    Додає повідомлення у логове вікно GUI з потокобезпечним оновленням.
    root  – головне вікно Tkinter
    output – віджет Text, куди пишемо повідомлення
    msg   – текст повідомлення
    tag   – тег для стилю (наприклад, колір)
    """
    # root.after(0, ...) – виконує дію у головному потоці GUI (через 0 мс), щоб уникнути проблем з потоками
    # lambda: (...) – анонімна функція, яка виконує одразу кілька дій
    root.after(0, lambda: (
        output.insert(tk.END, msg + "\n", tag),  # вставляємо повідомлення у кінець текстового поля
        output.see(tk.END)                       # прокручуємо лог донизу, щоб було видно останній рядок
    ))

def format_result_compact(mode, result):
    """
    Форматує результат у компактний вигляд для таблиці.
    mode   – який етап (features, params, structure, opt)
    result – словник або список з результатами
    """
    if isinstance(result, dict):  # Якщо результат – словник
        if mode == "features":    # Для відбору ознак
            return f"Ознаки: {', '.join(result.get('features', []))}"
        elif mode == "params":    # Для параметричного синтезу
            return f"h={result.get('hidden')}, lr={result.get('lr')}, α={result.get('alpha')}"
        elif mode == "structure": # Для структурного синтезу
            return f"{result.get('layers')}×{result.get('neurons')}"
        else:
            return ""             # Якщо інший словник – повертаємо порожній рядок
    elif isinstance(result, list) and mode == "opt":  # Якщо результат – список і режим "opt"
        return f"Парето‑рішень={len(result)}"         # Показуємо кількість Парето-рішень
    return ""  # Якщо нічого не підійшло – повертаємо порожній рядок

def set_running(root, is_running=True):
    """
    Блокує або розблоковує кнопки під час виконання обчислень.
    root – головне вікно
    is_running=True – якщо True, кнопки блокуються; якщо False – розблоковуються
    """
    def _apply():
        if hasattr(root, "btn_run_one"):   # Якщо у root є кнопка "Запустити етап"
            root.btn_run_one.config(state=("disabled" if is_running else "normal"))
        if hasattr(root, "btn_run_all"):   # Якщо є кнопка "Запустити всі етапи"
            root.btn_run_all.config(state=("disabled" if is_running else "normal"))
        if hasattr(root, "btn_save_csv"):  # Якщо є кнопка "Зберегти таблицю"
            root.btn_save_csv.config(state=("disabled" if is_running else "normal"))
        if hasattr(root, "btn_clear"):     # Якщо є кнопка "Очистити"
            root.btn_clear.config(state=("disabled" if is_running else "normal"))
        if hasattr(root, "status_var"):    # Якщо є змінна статусу
            root.status_var.set("Виконується…" if is_running else "Готово")  # Показуємо стан
    root.after(0, _apply)  # Виконуємо у головному потоці GUI

def insert_table_row(root, dataset, mode_desc, mae, rmse, extra):
    """Додає рядок у таблицю та оновлює підсвічування найкращого MAE."""
    def _ins():
        # Додаємо новий рядок у таблицю (Treeview)
        item_id = root.results_table.insert(
            "", "end",  # "" означає корінь, "end" — вставити в кінець
            values=(dataset, mode_desc, f"{mae:.3f}", f"{rmse:.3f}", extra)  # значення колонок
        )
        # Підсвічуємо найкращий MAE (мінімальний)
        try:
            # Отримуємо всі рядки таблиці
            items = root.results_table.get_children()
            # Створюємо список кортежів (id рядка, значення MAE)
            maes = [(iid, float(root.results_table.set(iid, "mae"))) for iid in items]
            if maes:
                # Знаходимо рядок з мінімальним MAE
                best_id = min(maes, key=lambda x: x[1])[0]
                # Скидаємо стиль у всіх рядків
                for iid in items:
                    root.results_table.item(iid, tags=())
                # Призначаємо тег "best" найкращому рядку
                root.results_table.item(best_id, tags=("best",))
                # Налаштовуємо стиль для тегу "best" (світло-зелений фон)
                root.results_table.tag_configure("best", background="#e6ffea")
        except Exception:
            pass
    # Виконуємо вставку у головному потоці Tkinter
    root.after(0, _ins)


def save_table_to_csv(root, path="results/summary.csv"):
    """Зберігає поточну таблицю результатів у CSV."""
    def _save():
        # Створюємо папку results, якщо її ще немає
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Заголовки колонок
        cols = ("Енергосистема", "Етап аналізу", "MAE", "RMSE", "Додатково")
        rows = []
        # Проходимо по всіх рядках таблиці
        for iid in root.results_table.get_children():
            vals = root.results_table.item(iid, "values")  # отримуємо значення рядка
            rows.append(vals)
        # Записуємо у CSV-файл
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(cols)   # записуємо заголовки
            writer.writerows(rows)  # записуємо всі рядки
        # Лог повідомлення про успішне збереження
        log(root, root.output, f"💾 Результати збережено у {path}", "ok")
    # Виконуємо у головному потоці Tkinter
    root.after(0, _save)


def clear_log_and_table(root):
    """Очищає лог і таблицю."""
    def _clear():
        # Очищаємо текстове поле логів
        root.output.delete("1.0", tk.END)
        # Видаляємо всі рядки з таблиці
        for iid in root.results_table.get_children():
            root.results_table.delete(iid)
        # Скидаємо статус у "Готово"
        if hasattr(root, "status_var"):
            root.status_var.set("Готово")
        # Лог повідомлення про очищення
        log(root, root.output, "🧹 Очищено лог і таблицю", "info")
    # Виконуємо у головному потоці Tkinter
    root.after(0, _clear)

def run_algorithm(root, output, dataset_var, mode_var, gen_var, on_finish=None):
    """
    Запуск одного етапу аналізу для вибраної енергосистеми.
    Після завершення викликає on_finish (для послідовного запуску).
    """
    dataset = dataset_var.get()   # Отримуємо назву вибраного датасету (наприклад, "s1", "s2")
    mode = mode_var.get()         # Отримуємо вибраний режим (features, params, structure, opt)

    # Валідація кількості поколінь
    try:
        gens = int(gen_var.get())   # Пробуємо перетворити введене значення у ціле число
        if gens <= 0:               # Якщо число <= 0
            raise ValueError("Кількість поколінь має бути > 0")  # Викликаємо помилку
    except Exception:
        # Якщо введено некоректне значення
        log(root, output, "⚠️ Некоректна кількість поколінь. Введіть додатнє ціле число.", "warn")
        if on_finish:               # Якщо передано callback on_finish
            root.after(0, on_finish) # Викликаємо його у головному потоці
        return                      # Завершуємо функцію

    if mode not in MODE_CONFIG:     # Якщо режиму немає у словнику MODE_CONFIG
        log(root, output, f"⚠️ Невідомий етап '{mode}'", "warn")
        if on_finish:
            root.after(0, on_finish)
        return

    desc = MODE_NAMES.get(mode, mode)   # Людяна назва етапу (наприклад, "Відбір ознак")
    func = MODE_CONFIG[mode]["func"]    # Функція, яка відповідає цьому етапу

    set_running(root, True)             # Блокуємо кнопки під час виконання
    log(root, output, f"⚡ Запуск етапу «{desc}» для енергосистеми {dataset}…", "info")

    def task():
        try:
            X, y, cols = load_dataset(dataset)  # Завантажуємо дані (X – ознаки, y – ціль, cols – назви ознак)
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
            if mode == "features":   # Якщо режим "Відбір ознак"
                result = func(
                    X, y, cols,      # Передаємо дані та список ознак
                    pop_size=8,      # Розмір популяції
                    n_gen=gens,      # Кількість поколінь
                    mutation_rate=0.2, # Ймовірність мутації
                    max_iter=100,    # Максимальна кількість ітерацій
                    cv_splits=3,     # Кількість фолдів для крос-валідації
                    progress_cb=make_progress # Callback для логування прогресу
                )
            else:                    # Для інших режимів (params, structure, opt)
                result = func(
                    X, y,            # Передаємо дані
                    pop_size=8 if mode != "opt" else 10, # Для opt трохи більша популяція
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
        if isinstance(result, dict):   # Якщо результат – словник (одне найкраще рішення)
            mae = float(result.get("mae", 0.0))   # Дістаємо MAE
            rmse = float(result.get("rmse", 0.0)) # Дістаємо RMSE
            extra = format_result_compact(mode, result) # Форматуємо додаткову інформацію
            log(root, output, f"✅ Етап «{desc}» завершено. Найкраща модель: MAE={mae:.3f}, RMSE={rmse:.3f}", "ok")
            insert_table_row(root, dataset, desc, mae, rmse, extra) # Додаємо рядок у таблицю

        elif isinstance(result, list) and mode == "opt": # Якщо результат – список (Парето-рішення)
            log(root, output, f"✅ Етап «{desc}» завершено. Знайдено Парето‑рішень: {len(result)}", "ok")
            # Додаємо всі Парето‑рішення окремими рядками
            for p in result:
                mae = float(p.get("mae", 0.0))
                rmse = float(p.get("rmse", 0.0))
                extra = f"{p.get('layers')}×{p.get('neurons')}"
                insert_table_row(root, dataset, desc, mae, rmse, extra)
        else:
            log(root, output, f"ℹ️ Етап «{desc}» повернув нетиповий результат", "warn")

        set_running(root, False)       # Розблоковуємо кнопки
        if on_finish:                  # Якщо передано callback
            root.after(0, on_finish)   # Викликаємо його у головному потоці

    threading.Thread(target=task, daemon=True).start() # Запускаємо task у окремому потоці


def run_all_modes(root, output, dataset_var, gen_var):
    """Запускає всі етапи послідовно: Відбір ознак → Параметри → Структура → Оптимізація."""
    modes = list(MODE_CONFIG.keys())   # Отримуємо список усіх режимів

    def run_next(i=0):
        if i >= len(modes):            # Якщо всі етапи пройдені
            log(root, output, "✅ Усі етапи аналізу завершені", "ok")
            return

        mode = modes[i]                # Беремо поточний режим
        desc = MODE_NAMES.get(mode, mode) # Людяна назва
        log(root, output, f"▶ Готуємо запуск етапу «{desc}»…", "info")

        # Запускаємо поточний етап, після завершення викликаємо наступний
        run_algorithm(
            root, output, dataset_var,
            tk.StringVar(value=mode), gen_var,
            on_finish=lambda: run_next(i+1)
        )

    run_next()  # Стартуємо з першого етапу

