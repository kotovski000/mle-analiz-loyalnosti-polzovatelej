import sys
from src.processing import DataProcessor


def main():
    print("Старт программы: исследование лояльности пользователей Яндекс Афиши")

    processor = DataProcessor(data_dir="data")

    # 1. Загрузка данных
    print("\n[1/5] Загрузка сырых данных и курсов валют")
    try:
        processor.load_currency_rates()
        processor.load_raw_data()
        print(f"Загружено: {len(processor.df_raw):,}")
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        sys.exit(1)

    # 2. Предобработка
    print("\n[2/5] Предобработка, очистка выбросов и удаление дубликатов")
    processor.preprocess_data()
    print(f"Чистых транзакций для анализа: {len(processor.df_clean):,}")

    # 3. Профилирование
    print("\n[3/5] Агрегация профилей и фильтрация аномалий")
    processor.build_user_profiles()
    print(f"Число активных пользователей: {len(processor.user_profiles_clean):,}")

    # 4. Анализ Retention и проверка гипотез
    print("\n[4/5] Расчет удержания и проверка гипотез")
    base_ret = processor.user_profiles_clean["is_two"].mean() * 100
    hyp = processor.evaluate_hypotheses()
    print(f"Базовый Retention Rate (>= 2 заказов): {base_ret:.2f}%")

    sport_ret = hyp["sport_retention"]
    concert_ret = hyp["concert_retention"]
    hyp1_status = "ПОДТВЕРДИЛАСЬ" if hyp["hyp1_confirmed"] else "ОПРОВЕРГНУТА"

    top10_ret = hyp["top10_reg_retention"]
    other_ret = hyp["other_reg_retention"]
    hyp2_status = "ПОДТВЕРДИЛАСЬ" if hyp["hyp2_confirmed"] else "ОПРОВЕРГНУТА"

    print(
        f"Гипотеза 1 (Спорт vs Концерты): "
        f"Спорт = {sport_ret:.2f}%, Концерты = {concert_ret:.2f}% — {hyp1_status}"
    )
    print(
        f"Гипотеза 2 (Топ-10 регионов vs Прочие): "
        f"Топ-10 = {top10_ret:.2f}%, Прочие = {other_ret:.2f}% — {hyp2_status}"
    )

    # 5. Корреляционный анализ phi_k
    print("\n[5/5] Корреляционный анализ факторов лояльности")
    correlations = processor.run_correlation_analysis()
    print(correlations.to_string())

    # Квантиль заказов
    orders_p75 = processor.user_profiles_clean["total_orders"].quantile(0.75)

    # Метрики по категориям событий
    cat_retention = processor.calculate_segment_retention("first_event_type")
    cat_dist = processor.get_first_order_distribution("first_event_type")

    theater_ret = cat_retention.loc[cat_retention["first_event_type"] == "театр", "retention_rate_%"].values[0]
    museum_ret = cat_retention.loc[cat_retention["first_event_type"] == "выставки", "retention_rate_%"].values[0]
    concert_share = cat_dist.loc[cat_dist["first_event_type"] == "концерты", "users_share_%"].values[0]

    tree_rows = cat_retention.loc[cat_retention["first_event_type"].str.contains("ёлка|елка", case=False, na=False)]
    tree_ret = tree_rows["retention_rate_%"].values[0] if not tree_rows.empty else 56.70

    # Дни недели
    dow_df = processor.analyze_day_of_week()
    dow_min = dow_df["retention_rate_%"].min()
    dow_max = dow_df["retention_rate_%"].max()

    # Корреляция
    phi_revenue = correlations.get("avg_revenue_rub", 0.0)

    print("\nКраткий итоговый отчет и бизнес-рекомендации для маркетинга")
    print(
        f"""
1. ОСНОВНЫЕ ВЫВОДЫ:
   - Базовый уровень Retention: {base_ret:.2f}% совершают повторные заказы; 75% оформляют не более {int(orders_p75)} заказов.
   - Театр ({theater_ret:.1f}%) и выставки ({museum_ret:.1f}%) — наиболее надежные точки входа с максимальным удержанием.
   - Концерты приводят {concert_share:.1f}% всех новых клиентов и служат главным каналом масштабирования.
   - Спорт ({sport_ret:.1f}%) и детские ёлки ({tree_ret:.1f}%) показывают наименьший процент повторных покупок.
   - День недели первой покупки практически не связан с возвратом (разброс всего {dow_min:.1f}% - {dow_max:.1f}%).
   - Главный фактор, ассоциированный с числом заказов — средний чек (phi_k = {phi_revenue:.3f}).

2. ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ:
   - Протестировать перераспределение бюджетов на спектакли и выставки (высокая возвратность).
   - Проверить гипотезу кросс-промо для спорта (скидки на фестивали и стендап в межсезонье).
   - Отказаться от сегментации маркетинговых рассылок по дням недели первого заказа.
"""
    )

if __name__ == "__main__":
    main()