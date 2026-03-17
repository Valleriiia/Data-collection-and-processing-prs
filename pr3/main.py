import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# 1. генерація синтетичного часового ряду
np.random.seed(42)

# 4 роки щомісячних даних (48 точок)
n_months = 48
dates = pd.date_range(start="2020-01-01", periods=n_months, freq="MS")

# Тренд: лінійне зростання споживання
trend_component = np.linspace(200, 320, n_months)

# Сезонність: річний цикл (пік взимку та влітку)
seasonal_component = (
    30 * np.sin(2 * np.pi * np.arange(n_months) / 12)
    + 15 * np.cos(4 * np.pi * np.arange(n_months) / 12)
)

# Шум
noise = np.random.normal(0, 8, n_months)

# Результуючий ряд (МВт·год)
energy = trend_component + seasonal_component + noise

ts = pd.Series(energy, index=dates, name="energy_MWh")

print("\nПерші 12 місяців (МВт·год):")
print(ts.head(12).round(2).to_string())

print("\nОписова статистика:")
print(ts.describe().round(2).to_string())

# 2. декомпозиція seasonal_decompose
print("\nДекомпозиція часового ряду")

decomposition = seasonal_decompose(ts, model="additive", period=12)

trend    = decomposition.trend.dropna()
seasonal = decomposition.seasonal
residual = decomposition.resid.dropna()

print(f"\nТренд   — діапазон: [{trend.min():.2f}, {trend.max():.2f}] МВт·год")
print(f"Сезонність — діапазон: [{seasonal.min():.2f}, {seasonal.max():.2f}] МВт·год")
print(f"Залишки — std: {residual.std():.2f} МВт·год")

# 3. Прогнозування на основі компонентів
print("\nОцінка точності прогнозування")

# Розбиваємо на train (36 міс.) / test (12 міс.)
split_idx = 36
train = ts.iloc[:split_idx]
test  = ts.iloc[split_idx:]

# Декомпозиція тренувальної частини
decomp_train = seasonal_decompose(train, model="additive", period=12)

# Прогноз тренду: лінійна екстраполяція по останніх 6 точках тренду
trend_train = decomp_train.trend.dropna()
x = np.arange(len(trend_train))
trend_coef = np.polyfit(x, trend_train.values, deg=1)
trend_line = np.poly1d(trend_coef)

future_x = np.arange(len(trend_train), len(trend_train) + len(test))
trend_forecast = trend_line(future_x)

# Прогноз сезонності: повторення останнього повного сезонного циклу
season_pattern = decomp_train.seasonal.values[-12:]
seasonal_forecast = np.tile(season_pattern, int(np.ceil(len(test) / 12)))[:len(test)]

# Прогноз = тренд + сезонність
forecast = trend_forecast + seasonal_forecast
forecast_series = pd.Series(forecast, index=test.index)

# Базовий прогноз (naive): повторення останніх 12 значень train
naive_forecast = pd.Series(
    train.values[-12:],
    index=test.index
)

# Метрики
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mae_decomp  = mean_absolute_error(test, forecast_series)
rmse_decomp = rmse(test, forecast_series)
mape_decomp = mape(test.values, forecast_series.values)

mae_naive   = mean_absolute_error(test, naive_forecast)
rmse_naive  = rmse(test, naive_forecast)
mape_naive  = mape(test.values, naive_forecast.values)

print(f"\n{'Метрика':<10} {'Декомпозиція':>16} {'Наївний (baseline)':>20}")
print("-" * 48)
print(f"{'MAE':<10} {mae_decomp:>16.3f} {mae_naive:>20.3f}")
print(f"{'RMSE':<10} {rmse_decomp:>16.3f} {rmse_naive:>20.3f}")
print(f"{'MAPE, %':<10} {mape_decomp:>16.3f} {mape_naive:>20.3f}")

improvement_mape = (mape_naive - mape_decomp) / mape_naive * 100
print(f"\nПрогноз через декомпозицію точніший за наївний на {improvement_mape:.1f}% (MAPE)")

# 4. візуалізація

# Графік 1: декомпозиція
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(4, 1, hspace=0.5)

ax0 = fig.add_subplot(gs[0])
ax0.plot(ts, color="#2C7BB6", linewidth=1.5)
ax0.set_title("Оригінальний ряд (енергоспоживання, МВт·год)", fontweight="bold")
ax0.set_ylabel("МВт·год")
ax0.grid(True, alpha=0.3)

ax1 = fig.add_subplot(gs[1])
ax1.plot(decomposition.trend, color="#D7191C", linewidth=2)
ax1.set_title("Тренд", fontweight="bold")
ax1.set_ylabel("МВт·год")
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[2])
ax2.plot(decomposition.seasonal, color="#1A9641", linewidth=1.5)
ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax2.set_title("Сезонна складова", fontweight="bold")
ax2.set_ylabel("МВт·год")
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[3])
ax3.plot(decomposition.resid, color="#FDAE61", linewidth=1.2)
ax3.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax3.set_title("Залишки (шум)", fontweight="bold")
ax3.set_ylabel("МВт·год")
ax3.grid(True, alpha=0.3)

fig.suptitle("Адитивна декомпозиція часового ряду енергоспоживання",
             fontsize=14, fontweight="bold", y=1.01)
plt.savefig("decomposition.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nГрафік декомпозиції збережено: decomposition.png")

fig2, ax = plt.subplots(figsize=(14, 5))

ax.plot(train.index, train.values, color="#2C7BB6", linewidth=1.5, label="Тренування (train)")
ax.plot(test.index,  test.values,  color="#1A9641", linewidth=1.5, label="Факт (test)")
ax.plot(forecast_series.index, forecast_series.values,
        color="#D7191C", linewidth=2, linestyle="--",
        label=f"Декомпозиція (MAPE={mape_decomp:.1f}%)")
ax.plot(naive_forecast.index, naive_forecast.values,
        color="#FDAE61", linewidth=1.5, linestyle=":",
        label=f"Наївний baseline (MAPE={mape_naive:.1f}%)")

ax.axvline(test.index[0], color="gray", linestyle="--", alpha=0.6, label="Межа train/test")
ax.fill_between(test.index,
                forecast_series - rmse_decomp,
                forecast_series + rmse_decomp,
                alpha=0.15, color="#D7191C", label="±RMSE коридор")

ax.set_title("Прогнозування енергоспоживання: декомпозиція vs наівний baseline",
             fontsize=13, fontweight="bold")
ax.set_ylabel("МВт·год")
ax.set_xlabel("Дата")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("forecast.png", dpi=150, bbox_inches="tight")
plt.close()
print("Графік прогнозу збережено: forecast.png")

# Графік 3: сезонний профіль
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 4))

months_ua = ["Січ", "Лют", "Бер", "Кві", "Тра", "Чер",
             "Лип", "Сер", "Вер", "Жов", "Лис", "Гру"]
seasonal_profile = decomposition.seasonal.groupby(decomposition.seasonal.index.month).mean()

axes3[0].bar(range(1, 13), seasonal_profile.values,
             color=["#D7191C" if v > 0 else "#2C7BB6" for v in seasonal_profile.values],
             edgecolor="white", linewidth=0.5)
axes3[0].set_xticks(range(1, 13))
axes3[0].set_xticklabels(months_ua, fontsize=9)
axes3[0].axhline(0, color="black", linewidth=0.8)
axes3[0].set_title("Середній сезонний профіль (по місяцях)", fontweight="bold")
axes3[0].set_ylabel("Відхилення від тренду (МВт·год)")
axes3[0].grid(True, alpha=0.3, axis="y")

errors = (test - forecast_series).values
axes3[1].bar(range(1, len(errors) + 1), errors,
             color=["#D7191C" if e > 0 else "#2C7BB6" for e in errors],
             edgecolor="white", linewidth=0.5)
axes3[1].axhline(0, color="black", linewidth=0.8)
axes3[1].set_title("Помилки прогнозу по місяцях тестового періоду", fontweight="bold")
axes3[1].set_xlabel("Місяць (тестовий період)")
axes3[1].set_ylabel("Помилка (МВт·год)")
axes3[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("seasonal_profile.png", dpi=150, bbox_inches="tight")
plt.close()
print("Графік сезонного профілю збережено: seasonal_profile.png")