"""PhysAudit-AI Lab — школьный цифровой физический практикум.

Загрузка CSV телеметрии (Data Harvest, до 1000 Гц) → сглаживание → метрики
(пик, длительность, площадь под кривой) → Socratic AI Auditor (OpenAI или
офлайн-фоллбэк) → 1-страничный PDF «Цифровой паспорт исследования».
"""

from __future__ import annotations

import io
import json
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from fpdf import FPDF
from scipy.integrate import trapezoid
from scipy.signal import savgol_filter

APP_TITLE = "PhysAudit-AI Lab"
DEFAULT_MODEL = "gpt-4o-mini"

# --------------------------------------------------------------------------
# Типы экспериментов: распознавание колонок, единицы, шаблоны фоллбэка
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Experiment:
    key: str
    title: str
    symbol: str  # обозначение величины
    unit: str
    mode: str  # "pulse" — одиночный процесс, "oscillation" — колебания
    area_label: str
    area_unit: str
    patterns: tuple[str, ...]  # регулярные выражения для заголовка колонки
    question: str
    analysis: str
    keywords: tuple[str, ...]  # ключевые понятия для проверки ответа


EXPERIMENTS: dict[str, Experiment] = {
    "force": Experiment(
        "force", "Краш-тест (сила удара)", "F", "Н", "pulse",
        "Импульс силы J = ∫F dt", "Н·с",
        (r"force", r"сила", r"^f\b", r"\(n\)", r"\(н\)"),
        question=(
            "Пик силы удара F_max = {peak} Н, длительность удара Δt = {dt} с, "
            "импульс J = {area} Н·с. Если тележку снабдить более мягким (эластичным) "
            "барьером, а импульс J останется прежним, как изменятся F_max и Δt? "
            "За счёт чего эластичный барьер снижает силу удара?"
        ),
        analysis=(
            "Площадь под графиком F(t) — это импульс силы J = Δp = m·Δv. При неизменном "
            "изменении импульса тележки увеличение времени взаимодействия Δt уменьшает "
            "среднюю силу ⟨F⟩ = J/Δt ≈ {avg} Н. Именно поэтому зоны деформации, подушки "
            "безопасности и мягкие барьеры «растягивают» удар во времени. Проверьте: "
            "F_max ({peak} Н) должно быть больше ⟨F⟩ — форма импульса не прямоугольная."
        ),
        keywords=("импульс", "врем", "длительн", "дольше", "растяг", "Δt", "dt"),
    ),
    "position": Experiment(
        "position", "Колебания маятника (координата)", "x", "м", "oscillation",
        "∫|x| dt", "м·с",
        (r"position", r"displacement", r"distance", r"координат", r"положени",
         r"смещени", r"^x\b", r"\(m\)", r"\(м\)"),
        question=(
            "Амплитуда маятника x_max = {peak} м, период T = {dt} с. Как изменится "
            "период, если длину нити увеличить в 4 раза? Почему амплитуда со временем "
            "уменьшается, а период почти не меняется?"
        ),
        analysis=(
            "Для математического маятника T = 2π√(l/g), значит длина нити ≈ g·T²/(4π²) ≈ "
            "{length} м. Период не зависит от амплитуды (при малых углах), поэтому "
            "затухание из-за сопротивления воздуха уменьшает x_max, но почти не меняет T. "
            "При увеличении длины в 4 раза период вырастет в 2 раза."
        ),
        keywords=("длин", "корень", "√", "2 раза", "вдвое", "сопротивл", "трени", "затух"),
    ),
    "velocity": Experiment(
        "velocity", "Движение тележки (скорость)", "v", "м/с", "pulse",
        "Путь s = ∫v dt", "м",
        (r"velocity", r"speed", r"скорост", r"^v\b", r"m/s", r"м/с"),
        question=(
            "Максимальная скорость тележки v_max = {peak} м/с, движение длилось "
            "Δt = {dt} с, пройденный путь s = {area} м. Какой была средняя скорость и "
            "почему она меньше v_max?"
        ),
        analysis=(
            "Площадь под графиком v(t) равна пройденному пути. Средняя скорость "
            "⟨v⟩ = s/Δt ≈ {avg} м/с меньше v_max = {peak} м/с, так как тележка часть "
            "времени разгонялась и тормозила."
        ),
        keywords=("путь", "площад", "разгон", "тормож", "средн"),
    ),
    "sound": Experiment(
        "sound", "Акустический шум (уровень звука)", "L", "дБ", "pulse",
        "∫(L − L_фон) dt", "дБ·с",
        (r"sound", r"noise", r"level", r"звук", r"шум", r"\(db\)", r"\(дб\)", r"\bdb\b"),
        question=(
            "Пиковый уровень звука L_max = {peak} дБ, громкий звук длился Δt = {dt} с. "
            "Во сколько раз интенсивность звука в пике больше фоновой ({base} дБ)? "
            "Опасен ли такой уровень для слуха при длительном воздействии?"
        ),
        analysis=(
            "Шкала децибел логарифмическая: разница {diff} дБ означает отношение "
            "интенсивностей 10^({diff}/10) ≈ {ratio}. Уровни выше 85 дБ при длительном "
            "воздействии вредны для слуха."
        ),
        keywords=("логариф", "10", "раз", "интенсивн", "85"),
    ),
    "temperature": Experiment(
        "temperature", "Нагревание (температура)", "T", "°C", "pulse",
        "∫(T − T₀) dt", "°C·с",
        (r"temp", r"температур", r"^t\s*\(", r"°c", r"\(c\)", r"\(с\)$"),
        question=(
            "Температура выросла с {base} °C до T_max = {peak} °C за Δt = {dt} с. "
            "Какую мощность получила вода массой 0,2 кг (c = 4200 Дж/(кг·°C))? "
            "Почему рост температуры замедляется к концу?"
        ),
        analysis=(
            "Q = c·m·ΔT = 4200·0,2·{diff} ≈ {heat} Дж, средняя мощность ≈ {power} Вт. "
            "Замедление роста связано с увеличением теплопотерь в окружающую среду при "
            "росте разности температур."
        ),
        keywords=("теплоёмк", "теплоемк", "q", "потер", "мощност"),
    ),
    "voltage": Experiment(
        "voltage", "Электрическая цепь (напряжение)", "U", "В", "pulse",
        "∫U dt", "В·с",
        (r"voltage", r"напряжен", r"^u\b", r"\(v\)", r"\(в\)"),
        question=(
            "Напряжение на конденсаторе достигло U_max = {peak} В, процесс занял "
            "Δt = {dt} с. Как изменится время зарядки, если сопротивление резистора "
            "увеличить вдвое?"
        ),
        analysis=(
            "Время заряда/разряда RC-цепи определяется постоянной τ = R·C: удвоение R "
            "удваивает τ и длительность процесса. Форма кривой — экспонента, а не прямая."
        ),
        keywords=("rc", "τ", "постоянн", "экспонент", "вдвое", "2 раза"),
    ),
}

GENERIC = Experiment(
    "generic", "Произвольная величина", "y", "", "pulse",
    "Площадь ∫(y − y₀) dt", "ед.·с", (),
    question=(
        "Пиковое значение {peak}, длительность процесса Δt = {dt} с, площадь под кривой "
        "{area}. Какой физический смысл имеет площадь под этим графиком?"
    ),
    analysis=(
        "Площадь под графиком величины по времени — это накопленный эффект процесса "
        "(например, импульс для силы, путь для скорости, заряд для тока). Сравните пик "
        "со средним значением ≈ {avg}."
    ),
    keywords=("площад", "интеграл", "накоп"),
)


def detect_experiment(column_name: str) -> Experiment:
    name = column_name.strip().lower()
    for exp in EXPERIMENTS.values():
        if any(re.search(p, name) for p in exp.patterns):
            return exp
    return GENERIC


# --------------------------------------------------------------------------
# Данные: парсинг CSV и демо-генераторы
# --------------------------------------------------------------------------


def parse_csv(raw: bytes) -> pd.DataFrame:
    """Читает CSV Data Harvest: авто-разделитель, десятичная запятая, рус/англ заголовки."""
    text = raw.decode("utf-8-sig", errors="replace")
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except Exception:
        df = pd.read_csv(io.StringIO(text), sep=";")
    if df.shape[1] < 2:
        raise ValueError("В файле должно быть минимум 2 колонки: время и измеряемая величина.")
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".", regex=False).str.strip(),
                errors="coerce",
            )
    df = df.dropna(how="all", axis=1)
    return df


def demo_data(kind: str, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fs = 1000  # Гц
    if kind == "force":
        t = np.arange(0, 1.0, 1 / fs)
        f = np.zeros_like(t)
        m = (t >= 0.40) & (t <= 0.46)
        f[m] = 120 * np.sin(np.pi * (t[m] - 0.40) / 0.06)
        y, col = f + rng.normal(0, 4, t.size), "Force (N)"
    elif kind == "position":
        t = np.arange(0, 10.0, 1 / fs)
        x = 0.10 * np.exp(-0.08 * t) * np.cos(2 * np.pi * t / 1.42)
        y, col = x + rng.normal(0, 0.004, t.size), "Position (m)"
    elif kind == "velocity":
        t = np.arange(0, 4.0, 1 / fs)
        v = np.where(t < 0.5, 0, np.where(t < 1.5, 1.2 * (t - 0.5),
                     np.where(t < 2.5, 1.2, np.clip(1.2 - 1.0 * (t - 2.5), 0, None))))
        y, col = v + rng.normal(0, 0.03, t.size), "Velocity (m/s)"
    elif kind == "sound":
        t = np.arange(0, 2.0, 1 / fs)
        l = 40 + np.where(t >= 0.7, 55 * np.exp(-(t - 0.7) / 0.12), 0)
        y, col = l + rng.normal(0, 1.5, t.size), "Sound (dB)"
    elif kind == "temperature":
        t = np.arange(0, 30.0, 1 / fs)
        temp = 20 + 45 * (1 - np.exp(-t / 10))
        y, col = temp + rng.normal(0, 0.3, t.size), "Temp (C)"
    else:  # voltage
        t = np.arange(0, 3.0, 1 / fs)
        u = np.where(t < 0.2, 0, 5 * (1 - np.exp(-(t - 0.2) / 0.3)))
        u = np.where(t > 1.8, u[np.searchsorted(t, 1.8)] * np.exp(-(t - 1.8) / 0.3), u)
        y, col = u + rng.normal(0, 0.08, t.size), "Voltage (V)"
    return pd.DataFrame({"Time (s)": np.round(t, 4), col: np.round(y, 4)})


# --------------------------------------------------------------------------
# Обработка сигнала
# --------------------------------------------------------------------------


def smooth(y: np.ndarray, fs: float, method: str, window_ms: float) -> np.ndarray:
    n = max(3, int(round(window_ms / 1000 * fs)))
    n = min(n, len(y) - (1 - len(y) % 2))  # окно не длиннее сигнала
    if n % 2 == 0:
        n += 1
    if n < 3:
        return y.copy()
    if method.startswith("Савицкого"):
        return savgol_filter(y, n, polyorder=min(3, n - 1))
    return pd.Series(y).rolling(n, center=True, min_periods=1).mean().to_numpy()


@dataclass
class Metrics:
    peak: float
    t_peak: float
    dt: float
    dt_label: str
    area: float
    baseline: float
    t_start: float
    t_end: float
    fs: float


def analyze(t: np.ndarray, y: np.ndarray, exp: Experiment) -> Metrics:
    fs = 1 / np.median(np.diff(t))
    if exp.mode == "oscillation":
        center = np.median(y)
        yc = y - center
        i_peak = int(np.argmax(np.abs(yc)))
        # период по пересечениям нуля снизу вверх
        cross = np.where((yc[:-1] < 0) & (yc[1:] >= 0))[0]
        t_cross = t[cross] - yc[cross] * (t[cross + 1] - t[cross]) / (yc[cross + 1] - yc[cross])
        periods = np.diff(t_cross)
        periods = periods[periods > 0.2 * np.median(periods)] if periods.size else periods
        period = float(np.median(periods)) if periods.size else float("nan")
        return Metrics(
            peak=float(np.abs(yc[i_peak])), t_peak=float(t[i_peak]), dt=period,
            dt_label="Период T", area=float(trapezoid(np.abs(yc), t)), baseline=float(center),
            t_start=float(t[0]), t_end=float(t[-1]), fs=fs,
        )

    n0 = max(5, len(y) // 20)
    baseline = float(np.median(y[:n0]))
    i_peak = int(np.argmax(y))
    peak = float(y[i_peak])
    thr = baseline + 0.1 * (peak - baseline)
    above = y >= thr
    i0 = i_peak
    while i0 > 0 and above[i0 - 1]:
        i0 -= 1
    i1 = i_peak
    while i1 < len(y) - 1 and above[i1 + 1]:
        i1 += 1
    area = float(trapezoid(y[i0:i1 + 1] - baseline, t[i0:i1 + 1])) if i1 > i0 else 0.0
    return Metrics(
        peak=peak, t_peak=float(t[i_peak]), dt=float(t[i1] - t[i0]),
        dt_label="Длительность Δt", area=area, baseline=baseline,
        t_start=float(t[i0]), t_end=float(t[i1]), fs=fs,
    )


def fmt(x: float, digits: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{digits}g}".replace(".", ",") if abs(x) < 1e5 else f"{x:.3e}"


def make_figure(t, y_raw, y_smooth, m: Metrics, exp: Experiment, ylabel: str):
    fig, ax = plt.subplots(figsize=(9, 4), dpi=110)
    ax.plot(t, y_raw, color="#9aa5b1", lw=0.6, alpha=0.7, label="Исходный сигнал (шум)")
    ax.plot(t, y_smooth, color="#d1495b", lw=1.8, label="Очищенный сигнал")
    if exp.mode == "pulse" and m.t_end > m.t_start:
        sel = (t >= m.t_start) & (t <= m.t_end)
        ax.fill_between(t[sel], m.baseline, y_smooth[sel], color="#d1495b", alpha=0.15,
                        label=exp.area_label)
    sign_peak = m.peak if exp.mode == "pulse" else y_smooth[np.searchsorted(t, m.t_peak)]
    ax.scatter([m.t_peak], [sign_peak], color="#00798c", zorder=5, s=30,
               label=f"Пик: {fmt(m.peak)} {exp.unit}")
    ax.set_xlabel("Время t, с")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# ИИ-модуль: Socratic AI Auditor
# --------------------------------------------------------------------------


def get_api_key() -> str | None:
    try:
        key = st.secrets.get("OPENAI_API_KEY")
    except Exception:  # нет secrets.toml
        key = None
    return key or os.environ.get("OPENAI_API_KEY") or None


def get_model() -> str:
    try:
        return st.secrets.get("OPENAI_MODEL", DEFAULT_MODEL)
    except Exception:
        return os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)


def template_values(m: Metrics, exp: Experiment) -> dict[str, str]:
    avg = m.area / m.dt if m.dt and np.isfinite(m.dt) and m.dt > 0 else float("nan")
    diff = m.peak - m.baseline
    vals = {
        "peak": fmt(m.peak), "dt": fmt(m.dt), "area": fmt(m.area), "avg": fmt(avg),
        "base": fmt(m.baseline), "diff": fmt(diff),
        "ratio": fmt(10 ** (diff / 10)) if exp.key == "sound" else "—",
        "length": fmt(9.81 * m.dt ** 2 / (4 * np.pi ** 2)) if np.isfinite(m.dt) else "—",
        "heat": fmt(4200 * 0.2 * diff), "power": fmt(4200 * 0.2 * diff / m.dt) if m.dt else "—",
    }
    return vals


def metrics_payload(m: Metrics, exp: Experiment) -> dict:
    return {
        "эксперимент": exp.title,
        "величина": f"{exp.symbol}, {exp.unit}",
        "пик": round(m.peak, 4),
        "время_пика_с": round(m.t_peak, 4),
        m.dt_label: round(m.dt, 4) if np.isfinite(m.dt) else None,
        exp.area_label: f"{round(m.area, 4)} {exp.area_unit}",
        "фон/среднее": round(m.baseline, 4),
        "частота_дискретизации_Гц": round(m.fs),
    }


def fallback_question(m: Metrics, exp: Experiment) -> dict:
    v = template_values(m, exp)
    return {"question": exp.question.format(**v), "analysis": exp.analysis.format(**v),
            "source": "офлайн-шаблон"}


def ai_question(m: Metrics, exp: Experiment) -> dict:
    key = get_api_key()
    if not key:
        return fallback_question(m, exp)
    try:
        from openai import OpenAI

        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=get_model(),
            temperature=0.6,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": (
                    "Ты — Socratic AI Auditor школьного физического практикума (8–11 класс). "
                    "По метрикам эксперимента составь ОДИН вопрос в формате PISA (контекст из "
                    "жизни, требует рассуждения, а не вычисления по формуле) и краткий критический "
                    "анализ графика (3–5 предложений): что означают пик, длительность и площадь, "
                    "какие погрешности возможны. Используй ТОЛЬКО переданные числа, не выдумывай "
                    "данных. Без LaTeX и markdown. Ответ — JSON: {\"question\": str, \"analysis\": str}."
                )},
                {"role": "user", "content": json.dumps(metrics_payload(m, exp), ensure_ascii=False)},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        return {"question": data["question"], "analysis": data["analysis"],
                "source": f"OpenAI {get_model()}"}
    except Exception as e:  # сеть, лимиты, неверный ключ — не роняем приложение
        out = fallback_question(m, exp)
        out["source"] = f"офлайн-шаблон (ошибка API: {type(e).__name__})"
        return out


NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")
HINTS = {
    "force": "как связаны импульс силы J = F·Δt и время удара?",
    "position": "от чего зависит период маятника и почему затухают колебания?",
    "velocity": "что означает площадь под графиком скорости?",
    "sound": "шкала децибел — логарифмическая.",
    "temperature": "вспомните формулу Q = c·m·ΔT и теплопотери.",
    "voltage": "что определяет постоянная времени τ = R·C?",
    "generic": "какой физический смысл у площади под графиком?",
}


def fallback_check(answer: str, m: Metrics, exp: Experiment) -> str:
    """Проверка без API: сверяет числа из ответа с метриками и ищет ключевые понятия."""
    refs = {
        "пик": m.peak, "длительность/период": m.dt, "площадь": m.area,
        "длительность в мс": m.dt * 1000, "фон": m.baseline,
    }
    refs = {k: v for k, v in refs.items() if np.isfinite(v) and v != 0}
    numbers = [float(n.replace(",", ".")) for n in NUM_RE.findall(answer)]
    confirmed, suspicious = [], []
    for x in numbers:
        if x in (0, 1, 2, 4, 10) or abs(x) < 1e-12:  # служебные числа («в 2 раза»)
            continue
        hit = next((k for k, v in refs.items() if abs(x - v) <= 0.15 * abs(v)), None)
        (confirmed if hit else suspicious).append((x, hit))
    low = answer.lower()
    concepts = [k for k in exp.keywords if k.lower() in low]

    lines = []
    if confirmed:
        lines.append("✅ Подтверждаются данными: " + ", ".join(
            f"{fmt(x)} ({k})" for x, k in confirmed) + ".")
    if suspicious:
        lines.append("⚠️ Возможная «галлюцинация»: числа " + ", ".join(fmt(x) for x, _ in suspicious)
                     + " не совпадают ни с одной измеренной метрикой (допуск ±15 %). "
                     "Откуда они взялись?")
    if not numbers:
        lines.append("ℹ️ В ответе нет чисел — подкрепите вывод значениями с графика.")
    if concepts:
        lines.append("✅ Использованы ключевые понятия: " + ", ".join(concepts) + ".")
    else:
        lines.append("❓ Не названо ключевое физическое понятие. Подсказка: "
                     + HINTS.get(exp.key, HINTS["generic"]))
    verdict = "Ответ согласуется с данными." if concepts and not suspicious else \
        "Ответ требует доработки."
    return "\n\n".join(lines + [f"**Итог:** {verdict}"])


def ai_check(answer: str, question: str, m: Metrics, exp: Experiment) -> tuple[str, str]:
    key = get_api_key()
    if not key:
        return fallback_check(answer, m, exp), "офлайн-проверка"
    try:
        from openai import OpenAI

        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=get_model(),
            temperature=0.2,
            messages=[
                {"role": "system", "content": (
                    "Ты — Socratic AI Auditor. Проверь ответ ученика на вопрос к графику. "
                    "Сверь каждое число и утверждение ученика с измеренными метриками; всё, "
                    "что не следует из данных или противоречит физике, пометь как «галлюцинацию». "
                    "Не давай готового ответа — задай 1 наводящий сократовский вопрос. "
                    "Формат: 3–6 коротких пунктов на русском, в конце строка «Итог: …»."
                )},
                {"role": "user", "content": (
                    f"Метрики: {json.dumps(metrics_payload(m, exp), ensure_ascii=False)}\n"
                    f"Вопрос: {question}\nОтвет ученика: {answer}"
                )},
            ],
        )
        return resp.choices[0].message.content.strip(), f"OpenAI {get_model()}"
    except Exception as e:
        return fallback_check(answer, m, exp), f"офлайн-проверка (ошибка API: {type(e).__name__})"


# --------------------------------------------------------------------------
# PDF «Цифровой паспорт исследования»
# --------------------------------------------------------------------------


def font_paths() -> tuple[Path, Path]:
    """DejaVu с кириллицей: из папки fonts/ репозитория или из комплекта matplotlib."""
    local = Path(__file__).parent / "fonts"
    mpl = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for base in (local, mpl):
        reg, bold = base / "DejaVuSans.ttf", base / "DejaVuSans-Bold.ttf"
        if reg.exists() and bold.exists():
            return reg, bold
    raise FileNotFoundError("Не найден шрифт DejaVuSans.ttf")


def clean_for_pdf(text: str) -> str:
    text = re.sub(r"[*_`#$]", "", text)
    text = re.sub(r"[✅⚠️❓ℹ️️]", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def build_pdf(fig, m: Metrics, exp: Experiment, info: dict, ai: dict | None,
              check: str | None) -> bytes:
    reg, bold = font_paths()
    pdf = FPDF(format="A4", unit="mm")
    pdf.set_auto_page_break(False)
    pdf.set_margins(15, 12, 15)
    pdf.add_page()
    pdf.add_font("DejaVu", "", str(reg))
    pdf.add_font("DejaVu", "B", str(bold))
    W = pdf.w - 30

    # Заголовок
    pdf.set_fill_color(209, 73, 91)
    pdf.rect(0, 0, pdf.w, 4, "F")
    pdf.set_y(10)
    pdf.set_font("DejaVu", "B", 17)
    pdf.cell(W, 8, "Цифровой паспорт исследования", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("DejaVu", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(W, 5, f"{APP_TITLE} · школьный цифровой физический практикум",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    # Реквизиты
    pdf.set_font("DejaVu", "", 10)
    rows = [
        ("Дата", info["date"]), ("Ученик / класс", info["student"] or "______________________"),
        ("Эксперимент", exp.title), ("Данные", info["source"]),
    ]
    for k, v in rows:
        pdf.set_font("DejaVu", "B", 10)
        pdf.cell(38, 5.5, k + ":")
        pdf.set_font("DejaVu", "", 10)
        pdf.cell(W - 38, 5.5, v, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # График
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    buf.seek(0)
    img_h = W * 4 / 9
    pdf.image(buf, x=15, y=pdf.get_y(), w=W, h=img_h)
    pdf.set_y(pdf.get_y() + img_h + 3)

    # 3 метрики в рамке
    y0 = pdf.get_y()
    cards = [
        (f"Пик {exp.symbol}_max", f"{fmt(m.peak)} {exp.unit}"),
        (m.dt_label, f"{fmt(m.dt)} с"),
        (exp.area_label, f"{fmt(m.area)} {exp.area_unit}"),
    ]
    cw = W / 3
    pdf.set_draw_color(209, 73, 91)
    pdf.set_line_width(0.5)
    pdf.rect(15, y0, W, 18)
    for i, (label, value) in enumerate(cards):
        x = 15 + i * cw
        if i:
            pdf.line(x, y0 + 2, x, y0 + 16)
        pdf.set_xy(x, y0 + 2)
        pdf.set_font("DejaVu", "", 8.5)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(cw, 5, label, align="C")
        pdf.set_xy(x, y0 + 8)
        pdf.set_font("DejaVu", "B", 13)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(cw, 8, value, align="C")
    pdf.set_line_width(0.2)
    pdf.set_draw_color(0, 0, 0)
    pdf.set_y(y0 + 22)

    # Вывод ИИ — строго в оставшееся место до блока подписи
    sign_top = pdf.h - 40
    pdf.set_font("DejaVu", "B", 11)
    src = f" ({ai['source']})" if ai else ""
    pdf.cell(W, 6, "Вывод ИИ-аудитора" + src, new_x="LMARGIN", new_y="NEXT")
    if ai:
        parts = [f"Вопрос (PISA): {ai['question']}", f"Анализ: {ai['analysis']}"]
        if info.get("answer"):
            parts.append(f"Ответ ученика: {info['answer']}")
        if check:
            parts.append(f"Проверка ответа: {check}")
        body = clean_for_pdf("\n".join(parts))
    else:
        body = "ИИ-анализ не запускался."
    pdf.set_font("DejaVu", "", 9.5)
    lh = 4.6
    lines = pdf.multi_cell(W, lh, body, dry_run=True, output="LINES")
    max_lines = int((sign_top - 3 - pdf.get_y()) // lh)
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [lines[max_lines - 1].rstrip()[:-3] + "…"]
    for line in lines:
        pdf.cell(W, lh, line, new_x="LMARGIN", new_y="NEXT")

    # Оценка и подпись учителя
    pdf.set_y(sign_top)
    pdf.set_draw_color(150, 150, 150)
    pdf.rect(15, sign_top, W, 28)
    pdf.set_xy(18, sign_top + 3)
    pdf.set_font("DejaVu", "B", 10)
    pdf.cell(W, 5, "Заключение учителя", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("DejaVu", "", 10)
    pdf.set_xy(18, sign_top + 11)
    pdf.cell(W - 6, 5, "Комментарий: " + "_" * 70)
    pdf.set_xy(18, sign_top + 20)
    pdf.cell(60, 5, "Оценка: ________")
    pdf.cell(70, 5, "Подпись учителя: ______________")
    pdf.cell(40, 5, "Дата: ________")
    return bytes(pdf.output())


# --------------------------------------------------------------------------
# Интерфейс
# --------------------------------------------------------------------------

st.set_page_config(page_title=APP_TITLE, page_icon="🧪", layout="wide")
st.title("🧪 PhysAudit-AI Lab")
st.caption("Цифровой физический практикум: телеметрия датчиков → очистка сигнала → "
           "метрики → Socratic AI Auditor → PDF-паспорт исследования")

with st.sidebar:
    st.header("1. Данные")
    uploaded = st.file_uploader("CSV-файл с датчика (Data Harvest)", type=["csv", "txt"])
    demo_kind = None
    if uploaded is None:
        demo_kind = st.selectbox(
            "Демо-режим: синтетический эксперимент (1000 Гц + шум)",
            list(EXPERIMENTS), format_func=lambda k: EXPERIMENTS[k].title,
        )

    st.header("2. Очистка сигнала")
    method = st.radio("Фильтр", ["Савицкого–Голея", "Скользящее среднее"])
    window_ms = st.slider("Окно фильтра, мс", 3, 501, 21, step=2)

    st.header("3. Паспорт")
    student = st.text_input("Ученик, класс", placeholder="Иванов И., 9 «А»")

# --- загрузка
try:
    if uploaded is not None:
        df = parse_csv(uploaded.getvalue())
        source = uploaded.name
    else:
        df = demo_data(demo_kind)
        source = "демо-данные (синтетические)"
except Exception as e:
    st.error(f"Не удалось прочитать файл: {e}")
    st.stop()

cols = list(df.columns)
with st.sidebar:
    if len(cols) > 2:
        y_col = st.selectbox("Колонка величины", cols[1:])
    else:
        y_col = cols[1]
    st.download_button("⬇️ Скачать пример CSV", demo_data(demo_kind or "force").to_csv(
        index=False).encode("utf-8"), file_name="example_telemetry.csv", mime="text/csv")

t_col = cols[0]
data = df[[t_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
data = data.sort_values(t_col).drop_duplicates(t_col)
if len(data) < 10:
    st.error(f"Недостаточно числовых данных: колонка времени «{t_col}», величины «{y_col}». "
             "Ожидается: 1-я колонка — время (с), 2-я — измеряемая величина.")
    st.stop()

exp = EXPERIMENTS[demo_kind] if demo_kind else detect_experiment(str(y_col))
t = data[t_col].to_numpy(float)
y_raw = data[y_col].to_numpy(float)
fs = 1 / np.median(np.diff(t))
y_s = smooth(y_raw, fs, method, window_ms)
m = analyze(t, y_s, exp)

st.success(f"**{exp.title}** · колонки: «{t_col}» (время), «{y_col}» "
           f"({exp.symbol}) · {len(t)} точек · ≈{fs:.0f} Гц · источник: {source}")

# --- метрики и график
c1, c2, c3 = st.columns(3)
c1.metric(f"Пиковое значение {exp.symbol}_max", f"{fmt(m.peak)} {exp.unit}",
          help=f"По очищенному сигналу; в момент t = {fmt(m.t_peak)} с")
c2.metric(m.dt_label, f"{fmt(m.dt)} с",
          help="Интервал, где сигнал выше фона на 10 % от пика" if exp.mode == "pulse"
          else "Медианный интервал между пересечениями положения равновесия")
c3.metric(exp.area_label, f"{fmt(m.area)} {exp.area_unit}",
          help="Численное интегрирование методом трапеций")

ylabel = f"{exp.symbol}, {exp.unit}" if exp.unit else str(y_col)
fig = make_figure(t, y_raw, y_s, m, exp, ylabel)
st.pyplot(fig)

with st.expander("Таблица данных"):
    st.dataframe(pd.DataFrame({t_col: t, f"{y_col} (исходный)": y_raw,
                               f"{y_col} (очищенный)": y_s}), height=250)

# --- ИИ-модуль
st.subheader("🤖 Socratic AI Auditor")
sig = (source, exp.key, y_col, method, window_ms, len(t))
if st.session_state.get("sig") != sig:
    st.session_state.update(sig=sig, ai=None, check=None, check_src=None)

if get_api_key() is None:
    st.info("API-ключ OpenAI не задан — работает офлайн-режим с заготовленными "
            "PISA-вопросами по типу эксперимента.", icon="ℹ️")

if st.button("Сгенерировать PISA-вопрос и анализ", type="primary"):
    with st.spinner("ИИ анализирует график…"):
        st.session_state.ai = ai_question(m, exp)
        st.session_state.check = None

ai = st.session_state.ai
answer = ""
if ai:
    st.markdown(f"**Вопрос (PISA):** {ai['question']}")
    st.markdown(f"**Анализ графика:** {ai['analysis']}")
    st.caption(f"Источник: {ai['source']}")
    answer = st.text_area("Ответ ученика", placeholder="Напишите своё объяснение с опорой на график…")
    if st.button("Проверить ответ на «галлюцинации»", disabled=not answer.strip()):
        with st.spinner("Сверяю ответ с данными…"):
            st.session_state.check, st.session_state.check_src = ai_check(
                answer, ai["question"], m, exp)
    if st.session_state.check:
        st.markdown(st.session_state.check)
        st.caption(f"Источник: {st.session_state.check_src}")

# --- PDF
st.subheader("📄 Цифровой паспорт исследования")
info = {"date": date.today().strftime("%d.%m.%Y"), "student": student.strip(),
        "source": source, "answer": answer.strip()}
pdf_bytes = build_pdf(fig, m, exp, info, ai, st.session_state.check)
st.download_button("⬇️ Скачать PDF (1 страница A4)", pdf_bytes,
                   file_name=f"passport_{exp.key}_{date.today():%Y%m%d}.pdf",
                   mime="application/pdf")
plt.close(fig)
