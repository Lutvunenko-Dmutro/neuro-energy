# static/mode_log.py
MODE_LOG = {
    "features": "=== Відібрані ознаки ===\nMAE={mae:.3f}, RMSE={rmse:.3f}\n{features}",
    "params":   "=== Результат параметричного синтезу ===\nMAE={mae:.3f}, RMSE={rmse:.3f}\nПараметри: {params}",
    "structure":"=== Результат структурного синтезу ===\nMAE={mae:.3f}, RMSE={rmse:.3f}\nАрхітектура: {arch}",
    "opt":      "=== Парето-фронт ==="
}