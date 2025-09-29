# === Дефолтні параметри для генетичних алгоритмів ===

# --- FAST MODE (для демонстрації) ---
FAST_MODE = {
    "POP_SIZE": 5,
    "GENERATIONS": 5,
    "MUTATION_RATE": 0.2,
    "CROSSOVER_RATE": 0.8,
    "CV_SPLITS": 2,
    "MAX_ITER": 50,
    "HIDDEN_NEURONS_RANGE": (8, 32),
    "LEARNING_RATE_RANGE": (-3.5, -2),   # 3e-4 .. 1e-2
    "ALPHA_RANGE": (-5, -3)              # 1e-5 .. 1e-3
}

# --- FULL MODE (для якісних результатів) ---
FULL_MODE = {
    "POP_SIZE": 10,
    "GENERATIONS": 20,
    "MUTATION_RATE": 0.1,
    "CROSSOVER_RATE": 0.8,
    "CV_SPLITS": 3,
    "MAX_ITER": 200,
    "HIDDEN_NEURONS_RANGE": (16, 128),
    "LEARNING_RATE_RANGE": (-4, -2),     # 1e-4 .. 1e-2
    "ALPHA_RANGE": (-6, -3)              # 1e-6 .. 1e-3
}