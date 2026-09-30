import streamlit as st
import pandas as pd
import plotly.express as px
from catboost import CatBoostRegressor

st.title("Прогноз стоимости автомобилей с пробегом")
st.write(
    "Интерактивный дашборд дипломного проекта: анализ факторов "
    "ценообразования и прогноз цены на основе модели CatBoost."
)

# Загрузка данных
df = pd.read_csv("../data/processed/cars_clean.csv")
st.caption(f"Данные: {df.shape[0]} объявлений, {df.shape[1]} признаков.")
# Загружаем обученную модель
model = CatBoostRegressor()
model.load_model("../models/catboost_model.cbm")
st.caption("Модель CatBoost загружена успешно.")

# Простой интерактивный элемент
st.sidebar.header("Фильтры")
selected_brand = st.sidebar.selectbox(
    "Выберите марку:",
    df['марка'].unique()
)

# Фильтрация данных
filtered_df = df[df['марка'] == selected_brand]

st.write(f"Показываем данные по марке: {selected_brand} ({len(filtered_df)} шт.)")
st.dataframe(filtered_df[['модель', 'год', 'цена', 'пробег']].head(10))

# --- График 1: Цена vs год ---
st.subheader("Зависимость цены от года выпуска")

fig1 = px.scatter(
    filtered_df,
    x="год",
    y="цена",
    color="цена",
    color_continuous_scale="Viridis",
    hover_data={
        "марка": True,
        "модель": True,
        "пробег": ":,",
        "мощность": True,
        "цена": ":,",
        "год": True,
    },
    opacity=0.6,
    title=f"Цена vs год выпуска — {selected_brand} ({len(filtered_df)} объявлений)",
)

fig1.update_layout(
    xaxis_title="Год выпуска",
    yaxis_title="Цена, руб.",
    height=550,
    coloraxis_showscale=False,
)

st.plotly_chart(fig1, use_container_width=True)

# --- График 2: Цена vs пробег ---
st.subheader("Зависимость цены от пробега")

fig2 = px.scatter(
    filtered_df,
    x="пробег",
    y="цена",
    color="цена",
    color_continuous_scale="Plasma",
    hover_data={
        "марка": True,
        "модель": True,
        "год": True,
        "мощность": True,
        "пробег": ":,",
        "цена": ":,",
    },
    opacity=0.6,
    title=f"Цена vs пробег — {selected_brand} ({len(filtered_df)} объявлений)",
)

fig2.update_layout(
    xaxis_title="Пробег, км",
    yaxis_title="Цена, руб.",
    height=550,
    coloraxis_showscale=False,
)

st.plotly_chart(fig2, use_container_width=True)

# --- График 3: Важность признаков (CatBoost) ---
st.subheader("Важность признаков (что влияет на цену сильнее всего)")

# Получаем важность признаков из модели
importances = model.get_feature_importance()
feature_names = model.feature_names_

# Собираем в DataFrame и сортируем по убыванию
importance_df = pd.DataFrame({
    "признак": feature_names,
    "важность": importances
}).sort_values("важность", ascending=True)  # по возрастанию — для горизонтального графика

# Горизонтальный бар-чарт
fig3 = px.bar(
    importance_df,
    x="важность",
    y="признак",
    orientation="h",
    color="важность",
    color_continuous_scale="Teal",
    title="Топ признаков по вкладу в прогноз цены",
    text="важность",
)

fig3.update_traces(
    texttemplate="%{text:.1f}",
    textposition="outside",
)

fig3.update_layout(
    xaxis_title="Важность, %",
    yaxis_title="",
    height=600,
    coloraxis_showscale=False,
    margin=dict(l=120),   # отступ слева для подписей признаков
)

st.plotly_chart(fig3, use_container_width=True)

# ============================================
# ФОРМА ПРОГНОЗА
# ============================================

st.divider()
st.subheader("Прогноз цены")
st.write("Заполните характеристики автомобиля и получите оценку рыночной стоимости.")

col_left, col_right = st.columns(2)

with col_left:
    input_марка = st.selectbox("Марка", sorted(df["марка"].dropna().unique()))
    доступные_модели = sorted(df[df["марка"] == input_марка]["модель"].dropna().unique())
    input_модель = st.selectbox("Модель", доступные_модели)
    input_год = st.number_input("Год выпуска", min_value=1990, max_value=2025, value=2018, step=1)
    input_мощность = st.number_input("Мощность, л.с.", min_value=30, max_value=1500, value=150, step=5)
    input_пробег = st.number_input("Пробег, км", min_value=0, max_value=1_000_000, value=100_000, step=5000)
    input_объём = st.number_input("Объём двигателя, л", min_value=0.5, max_value=8.0, value=2.0, step=0.1)
    input_владельцы = st.number_input("Владельцев по ПТС", min_value=1.0, max_value=10.0, value=1.0, step=1.0)

with col_right:
    input_коробка = st.selectbox("Коробка передач",
                                 ["АКПП", "механика", "вариатор", "робот", "неизвестно"])
    input_привод = st.selectbox("Привод",
                                ["передний", "4WD", "задний", "неизвестно"])
    input_топливо = st.selectbox("Топливо",
                                 ["Бензин", "Дизель", "Гибрид", "Электро", "Неизвестно"])
    input_руль = st.selectbox("Руль", ["левый", "правый", "неизвестно"])
    input_цвет = st.selectbox("Цвет", sorted(df["цвет"].dropna().unique()))
    input_город = st.selectbox("Город", df["город"].value_counts().head(20).index.tolist())
    input_поколение = st.selectbox("Поколение",
                                   df["поколение"].value_counts().head(30).index.tolist())
    input_комплектация = st.text_input("Комплектация", value="")
    input_гбо = st.checkbox("ГБО установлено", value=False)

if st.button("Предсказать цену", type="primary", use_container_width=True):
    input_row = pd.DataFrame([{
        "год": int(input_год),
        "мощность": float(input_мощность),
        "коробка": input_коробка,
        "привод": input_привод,
        "цвет": input_цвет,
        "пробег": int(input_пробег),
        "владельцы": float(input_владельцы),
        "руль": input_руль,
        "поколение": input_поколение,
        "комплектация": input_комплектация,
        "город": input_город,
        "объём": float(input_объём),
        "топливо": input_топливо,
        "гбо": bool(input_гбо),
        "марка": input_марка,
        "модель": input_модель,
    }])

    prediction = model.predict(input_row)[0]

    st.success(f"### Прогноз цены: {prediction:,.0f} руб.")
    st.caption(
        f"Средняя ошибка модели (MAE) — около 233 000 руб. "
        f"Реалистичный диапазон: от {max(0, prediction - 233000):,.0f} "
        f"до {prediction + 233000:,.0f} руб."
    )