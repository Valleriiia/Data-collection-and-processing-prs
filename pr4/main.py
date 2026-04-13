import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# 1. Параметри мережі
N_SENSORS = 20 # кількість IoT-датчиків
N_SERVERS = 4 # кількість серверів-обробників
N_TICKS = 200 # кількість тактів симуляції
MAX_LOAD = 25.0 # максимальна пропускна здатність одного сервера за такт

print("\nБалансування навантаження IoT-мережі")
print(f"\nПараметри мережі:")
print(f"  Датчики:          {N_SENSORS}")
print(f"  Сервери:          {N_SERVERS}")
print(f"  Такти симуляції:  {N_TICKS}")
print(f"  Ємність сервера:  {MAX_LOAD} од./такт")

# 2. Генереція трафіку датчиків
sensor_loads = []
for i in range(N_SENSORS):
    base = np.random.uniform(0.5, 2.0) # базова інтенсивність
    phase = np.random.uniform(0, 2 * np.pi) # зсув фази
    period = np.random.choice([20, 40, 60]) # тривалість циклу
    t = np.arange(N_TICKS)
    load = base + 1.0 * np.sin(2 * np.pi * t / period + phase) \
             + np.random.exponential(0.3, N_TICKS)
    load = np.clip(load, 0.1, 5.0)
    sensor_loads.append(load)

sensor_loads = np.array(sensor_loads)

avg_total = sensor_loads.sum(axis=0).mean()
print(f"  Середнє сумарне навантаження/такт: {avg_total:.2f}")
print(f"  Загальна ємність мережі/такт:      {N_SERVERS * MAX_LOAD:.2f}")

# 3. Стратегія 1 — статична: Round Robin
def static_round_robin(sensor_loads, n_servers):
    n_sensors, n_ticks = sensor_loads.shape
    server_loads = np.zeros((n_servers, n_ticks))
    for sid in range(n_sensors):
        server_loads[sid % n_servers] += sensor_loads[sid]
    dropped = (server_loads > MAX_LOAD).sum()
    return server_loads, int(dropped)

rr_loads, rr_dropped = static_round_robin(sensor_loads, N_SERVERS)

# 4. Стратегія 2 — статична: Random
def static_random(sensor_loads, n_servers):
    n_sensors, n_ticks = sensor_loads.shape
    server_loads = np.zeros((n_servers, n_ticks))
    assignments = np.random.randint(0, n_servers, size=n_sensors)
    for sid in range(n_sensors):
        server_loads[assignments[sid]] += sensor_loads[sid]
    dropped = (server_loads > MAX_LOAD).sum()
    return server_loads, int(dropped)

rand_loads, rand_dropped = static_random(sensor_loads, N_SERVERS)

# 5. Стратегія 3 — інтелектуальна: Least Load (динамічна)
def intelligent_least_load(sensor_loads, n_servers, max_load):
    n_sensors, n_ticks = sensor_loads.shape
    server_loads = np.zeros((n_servers, n_ticks))
    dropped = 0

    for t in range(n_ticks):
        current = np.zeros(n_servers)

        for sid in range(n_sensors):
            packet = sensor_loads[sid, t]
            target = int(np.argmin(current))

            if current[target] + packet <= max_load:
                current[target] += packet
            else:
                current[target] += packet
                dropped += 1

        server_loads[:, t] = current

    return server_loads, dropped

smart_loads, smart_dropped = intelligent_least_load(sensor_loads, N_SERVERS, MAX_LOAD)

# 6. Метрика якості
def compute_metrics(server_loads, max_load, name, dropped):
    mean_load = server_loads.mean()
    std_load = server_loads.std()
    imbalance = server_loads.std(axis=0).mean() # дисбаланс між серверами
    overload_pct = (server_loads.max(axis=0) > max_load).mean() * 100
    peak = server_loads.max()
    utilization = mean_load / max_load * 100 # утилізація ємності

    print(f"\n{name}")
    print(f"  Середнє навантаження:       {mean_load:.3f}")
    print(f"  Утилізація ємності:         {utilization:.1f}%")
    print(f"  Дисбаланс між серверами:    {imbalance:.3f}")
    print(f"  Пік навантаження:           {peak:.3f}")
    print(f"  Такти з перевантаженням:    {overload_pct:.1f}%")
    print(f"  Перевищень ємності:         {dropped}")

    return {
        "name": name,
        "mean": mean_load,
        "utilization": utilization,
        "std": std_load,
        "imbalance": imbalance,
        "overload_pct": overload_pct,
        "peak": peak,
        "dropped": dropped,
    }

print("\nМетрика якості балансування")

m_rr = compute_metrics(rr_loads,    MAX_LOAD, "Round Robin (статичний)", rr_dropped)
m_rand = compute_metrics(rand_loads,  MAX_LOAD, "Random (статичний)",      rand_dropped)
m_smart = compute_metrics(smart_loads, MAX_LOAD, "Least Load (інтелект.)",  smart_dropped)

print("\nПорівняльна таблиця")
all_m = [m_rr, m_rand, m_smart]
print(f"{'Метрика':<32} {'Round Robin':>12} {'Random':>12} {'Least Load':>12}")
print("─" * 70)
for key, label in [
    ("mean",        "Середнє навантаження"),
    ("utilization", "Утилізація, %"),
    ("imbalance",   "Дисбаланс"),
    ("overload_pct","Перевантаження, %"),
    ("peak",        "Пік навантаження"),
    ("dropped",     "Перевищень ємності"),
]:
    vals = [m[key] for m in all_m]
    print(f"  {label:<30} {vals[0]:>12.2f} {vals[1]:>12.2f} {vals[2]:>12.2f}")

print(f"\nНайменший дисбаланс:     {min(all_m, key=lambda m: m['imbalance'])['name']}")
print(f"Найменше перевантажень:  {min(all_m, key=lambda m: m['overload_pct'])['name']}")

# 7. Візуалізація
COLORS = ["#2C7BB6", "#1A9641", "#D7191C", "#984EA3"]

# Графік 1: навантаження серверів у часі
fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
fig.suptitle("Навантаження серверів у часі", fontsize=14, fontweight="bold")

for ax, (loads, title) in zip(axes, [
    (rr_loads,    "Round Robin (статичний)"),
    (rand_loads,  "Random (статичний)"),
    (smart_loads, "Least Load (інтелектуальний)"),
]):
    t = np.arange(N_TICKS)
    for srv in range(N_SERVERS):
        ax.plot(t, loads[srv], alpha=0.8, linewidth=1.2,
                label=f"Сервер {srv+1}", color=COLORS[srv])
    ax.axhline(MAX_LOAD, color="red", linestyle="--",
               linewidth=1.2, alpha=0.7, label=f"Ємність ({MAX_LOAD})")
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_ylabel("Навантаження")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.25)

axes[-1].set_xlabel("Такт симуляції")
plt.tight_layout()
plt.savefig("load_over_time.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nГрафік 1 збережено: load_over_time.png")

# Графік 2: boxplot + дисбаланс у часі
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("Розподіл навантаження та дисбаланс", fontsize=13, fontweight="bold")

bp = axes2[0].boxplot(
    [rr_loads.flatten(), rand_loads.flatten(), smart_loads.flatten()],
    labels=["Round Robin", "Random", "Least Load"],
    patch_artist=True,
    medianprops={"color": "black", "linewidth": 2}
)
for patch, color in zip(bp["boxes"], ["#2C7BB6", "#FDAE61", "#1A9641"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
axes2[0].axhline(MAX_LOAD, color="red", linestyle="--", alpha=0.6, label=f"Ємність ({MAX_LOAD})")
axes2[0].set_title("Розподіл навантаження (boxplot)", fontweight="bold")
axes2[0].set_ylabel("Навантаження на сервер за такт")
axes2[0].legend()
axes2[0].grid(True, alpha=0.25, axis="y")

t = np.arange(N_TICKS)
for loads, label, color in zip(
    [rr_loads, rand_loads, smart_loads],
    ["Round Robin", "Random", "Least Load"],
    ["#2C7BB6", "#FDAE61", "#1A9641"]
):
    axes2[1].plot(t, loads.std(axis=0), alpha=0.8, linewidth=1.2,
                  label=label, color=color)
axes2[1].set_title("Дисбаланс між серверами (std) у часі", fontweight="bold")
axes2[1].set_xlabel("Такт симуляції")
axes2[1].set_ylabel("std навантаження між серверами")
axes2[1].legend(fontsize=9)
axes2[1].grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig("distribution_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("Графік 2 збережено: distribution_analysis.png")

# Графік 3: bar chart метрик + теплова карта
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle("Порівняння метрик та теплова карта навантаження", fontsize=13, fontweight="bold")

metric_names = ["Дисбаланс", "Перевант.\n%", "Пік"]
rr_v   = [m_rr["imbalance"],    m_rr["overload_pct"],    m_rr["peak"]]
rand_v = [m_rand["imbalance"],  m_rand["overload_pct"],  m_rand["peak"]]
sm_v   = [m_smart["imbalance"], m_smart["overload_pct"], m_smart["peak"]]

x = np.arange(len(metric_names))
w = 0.25
axes3[0].bar(x - w, rr_v,   w, label="Round Robin", color="#2C7BB6", alpha=0.85)
axes3[0].bar(x,     rand_v, w, label="Random",       color="#FDAE61", alpha=0.85)
axes3[0].bar(x + w, sm_v,   w, label="Least Load",   color="#1A9641", alpha=0.85)
axes3[0].set_xticks(x)
axes3[0].set_xticklabels(metric_names)
axes3[0].set_title("Порівняння ключових метрик", fontweight="bold")
axes3[0].legend(fontsize=9)
axes3[0].grid(True, alpha=0.25, axis="y")

# Теплова карта: всі три стратегії, середнє навантаження по серверах
hm = np.array([
    rr_loads.mean(axis=1),
    rand_loads.mean(axis=1),
    smart_loads.mean(axis=1),
])
im = axes3[1].imshow(hm, aspect="auto", cmap="YlOrRd", vmin=0, vmax=MAX_LOAD)
axes3[1].set_title("Середнє навантаження по серверах\n(рядок = стратегія, стовпець = сервер)",
                    fontweight="bold")
axes3[1].set_yticks([0, 1, 2])
axes3[1].set_yticklabels(["Round Robin", "Random", "Least Load"])
axes3[1].set_xticks(range(N_SERVERS))
axes3[1].set_xticklabels([f"Srv {i+1}" for i in range(N_SERVERS)])
for i in range(3):
    for j in range(N_SERVERS):
        axes3[1].text(j, i, f"{hm[i,j]:.1f}", ha="center", va="center",
                      fontsize=10, fontweight="bold",
                      color="white" if hm[i,j] > MAX_LOAD * 0.6 else "black")
plt.colorbar(im, ax=axes3[1], label="Навантаження")
plt.tight_layout()
plt.savefig("metrics_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Графік 3 збережено: metrics_comparison.png")

# Графік 4: трафік датчиків
fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
fig4.suptitle("Аналіз трафіку IoT-датчиків", fontsize=13, fontweight="bold")

total_traffic = sensor_loads.sum(axis=0)
axes4[0].fill_between(np.arange(N_TICKS), total_traffic, alpha=0.35, color="#2C7BB6")
axes4[0].plot(np.arange(N_TICKS), total_traffic, color="#2C7BB6", linewidth=1.2)
axes4[0].axhline(N_SERVERS * MAX_LOAD, color="red", linestyle="--",
                  alpha=0.7, label=f"Загальна ємність ({N_SERVERS*MAX_LOAD})")
axes4[0].set_title("Сумарний трафік усіх датчиків у часі", fontweight="bold")
axes4[0].set_xlabel("Такт симуляції")
axes4[0].set_ylabel("Сумарне навантаження")
axes4[0].legend()
axes4[0].grid(True, alpha=0.25)

avg_per_sensor = sensor_loads.mean(axis=1)
bar_colors = plt.cm.RdYlGn_r(avg_per_sensor / avg_per_sensor.max())
axes4[1].bar(range(N_SENSORS), avg_per_sensor, color=bar_colors, edgecolor="white")
axes4[1].set_title("Середнє навантаження кожного датчика", fontweight="bold")
axes4[1].set_xlabel("ID датчика")
axes4[1].set_ylabel("Середнє навантаження")
axes4[1].grid(True, alpha=0.25, axis="y")
plt.tight_layout()
plt.savefig("sensor_traffic.png", dpi=150, bbox_inches="tight")
plt.close()
print("Графік 4 збережено: sensor_traffic.png")