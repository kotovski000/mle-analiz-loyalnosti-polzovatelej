import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
import phik


def get_db_engine():
    """ Подключение к базе данных PostgreSQL на основе переменных окружения из .env."""
    load_dotenv()
    user = os.getenv("DB_USER")
    pwd = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    db = os.getenv("DB_NAME")

    connection_string = f"postgresql://{user}:{pwd}@{host}:{port}/{db}"
    return create_engine(connection_string)


class DataProcessor:
    """Класс для загрузки, предобработки и анализа данных Яндекс Афиши."""

    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.df_raw = pd.DataFrame()
        self.df_clean = pd.DataFrame()
        self.rates_df = pd.DataFrame()
        self.user_profiles = pd.DataFrame()
        self.user_profiles_clean = pd.DataFrame()

    def load_raw_data(self, file_name="raw_afisha_data.csv"):
        """
        Задача 3.1: Загрузка сырых данных.
        Если локальный CSV уже существует в папке data/ — считывать его.
        Если файла нет — выполнять SQL-запрос к БД, сохранять результат в CSV и возвращать DataFrame.
        """
        csv_path = os.path.join(self.data_dir, file_name)

        if os.path.exists(csv_path):
            print(f"Найден локальный файл '{csv_path}'. Загрузка из CSV")
            self.df_raw = pd.read_csv(csv_path)
            self.df_raw["order_dt"] = pd.to_datetime(self.df_raw["order_dt"])
            self.df_raw["order_ts"] = pd.to_datetime(self.df_raw["order_ts"])
            return self.df_raw

        print(f"Файл '{csv_path}' не найден. Выполняется SQL-запрос к базе данных")
        engine = get_db_engine()
        query = '''
        WITH set_config_precode AS (
          SELECT set_config('synchronize_seqscans', 'off', true)
        )
        SELECT 
            p.user_id,
            p.device_type_canonical,
            p.order_id,
            p.created_dt_msk AS order_dt,
            p.created_ts_msk AS order_ts,
            p.currency_code,
            p.revenue,
            p.tickets_count,
            (p.created_dt_msk::date - LAG(p.created_dt_msk::date) OVER (
                PARTITION BY p.user_id 
                ORDER BY p.created_dt_msk
            ))::int AS days_since_prev,
            p.event_id,
            p.service_name,
            e.event_type_main,
            r.region_name,
            c.city_name
        FROM afisha.purchases p
        JOIN afisha.events e ON p.event_id = e.event_id
        JOIN afisha.city c ON e.city_id = c.city_id
        JOIN afisha.regions r ON c.region_id = r.region_id
        WHERE p.device_type_canonical IN ('mobile', 'desktop')
          AND e.event_type_main != 'фильм'
        ORDER BY p.user_id ASC;
        '''
        self.df_raw = pd.read_sql_query(query, con=engine)
        self.df_raw["order_dt"] = pd.to_datetime(self.df_raw["order_dt"])
        self.df_raw["order_ts"] = pd.to_datetime(self.df_raw["order_ts"])

        os.makedirs(self.data_dir, exist_ok=True)
        self.df_raw.to_csv(csv_path, index=False)
        print(f"Данные успешно выгружены из БД и сохранены в '{csv_path}'.")

        return self.df_raw

    def load_currency_rates(self, file_name="final_tickets_tenge_df.csv"):
        """Задача 3.1: Загрузка справочника курсов казахстанского тенге."""
        path = os.path.join(self.data_dir, file_name)
        if os.path.exists(path):
            self.rates_df = pd.read_csv(path)
        else:
            url = "https://code.s3.yandex.net/datasets/final_tickets_tenge_df.csv"
            self.rates_df = pd.read_csv(url)

        self.rates_df["data"] = pd.to_datetime(self.rates_df["data"])
        return self.rates_df

    def preprocess_data(self):
        """Задача 3.1: Конвертация валют, нормализация типов, чистка выбросов и удаление дубликатов."""
        df = self.df_raw.copy()

        # 1. Присоединение курса валют и расчет выручки в рублях
        rows_before = len(df)
        df = df.merge(
            self.rates_df[["data", "curs"]],
            left_on="order_dt",
            right_on="data",
            how="left",
        ).drop(columns=["data"])

        assert len(df) == rows_before, "Количество строк изменилось после слияния!"
        assert df["curs"].isna().sum() == 0, "Обнаружены пропуски в курсе валют!"

        df["revenue_rub"] = df["revenue"]
        kzt_mask = df["currency_code"] == "kzt"
        df.loc[kzt_mask, "revenue_rub"] = (
            df.loc[kzt_mask, "revenue"] * (df.loc[kzt_mask, "curs"] / 100)
        ).round(2)
        df = df.drop(columns=["curs"])

        # 2. Нормализация текста и приведение типов
        str_cols = ["device_type_canonical", "service_name", "event_type_main", "region_name", "city_name"]
        for col in str_cols:
            df[col] = df[col].astype(str).str.strip().str.lower()

        df["days_since_prev"] = df["days_since_prev"].astype("Int64")
        df["tickets_count"] = df["tickets_count"].astype("int16")

        # 3. Фильтрация возвратов (<= 0) и срезание по 99 перцентилю
        p99_revenue = df["revenue_rub"].quantile(0.99)
        df_filtered = df[(df["revenue_rub"] > 0) & (df["revenue_rub"] <= p99_revenue)].copy()

        # 4. Удаление дубликатов
        dup_cols = ["user_id", "order_ts", "device_type_canonical", "service_name", "event_type_main", "revenue_rub"]
        df_filtered = df_filtered.drop_duplicates(subset=dup_cols, keep="first").reset_index(drop=True)

        self.df_clean = df_filtered
        return self.df_clean

    def build_user_profiles(self):
        """Задача 3.2: Агрегация профилей."""
        df = self.df_clean.sort_values(by=["user_id", "order_ts"]).reset_index(drop=True)

        df["prev_order_ts"] = df.groupby("user_id")["order_ts"].shift(1)
        df["days_since_prev_clean"] = (
            (df["order_ts"] - df["prev_order_ts"]).dt.total_seconds() / 86400.0
        )

        profiles = df.groupby("user_id").agg(
            first_order_dt=("order_dt", "min"),
            last_order_dt=("order_dt", "max"),
            first_device=("device_type_canonical", "first"),
            first_region=("region_name", "first"),
            first_event_type=("event_type_main", "first"),
            total_orders=("order_id", "count"),
            avg_revenue_rub=("revenue_rub", "mean"),
            avg_days_between_orders=("days_since_prev_clean", "mean"),
        ).reset_index()

        profiles["is_two"] = (profiles["total_orders"] >= 2).astype(int)
        self.user_profiles = profiles

        # Фильтрация аномальных профилей
        p99_orders = profiles["total_orders"].quantile(0.99)
        self.user_profiles_clean = profiles[profiles["total_orders"] <= p99_orders].copy()
        return self.user_profiles_clean

    def get_first_order_distribution(self, column):
        """Расчет долей пользователей по характеристикам первого заказа."""
        dist = self.user_profiles_clean.groupby(column)["user_id"].count().reset_index()
        dist.columns = [column, "users_count"]
        dist["users_share_%"] = (dist["users_count"] / len(self.user_profiles_clean) * 100).round(2)
        return dist.sort_values(by="users_count", ascending=False).reset_index(drop=True)

    def calculate_segment_retention(self, column, min_users=50):
        """Задача 3.2: Расчет Retention по сегментам."""
        res = self.user_profiles_clean.groupby(column).agg(
            users_count=("user_id", "count"),
            repeat_users=("is_two", "sum"),
            retention_rate=("is_two", "mean"),
        ).reset_index()

        res["retention_rate_%"] = (res["retention_rate"] * 100).round(2)
        res = res[res["users_count"] >= min_users].sort_values(by="retention_rate_%", ascending=False).reset_index(drop=True)
        return res

    def evaluate_hypotheses(self):
        """Задача 3.2: Проверка гипотез по жанрам и регионам."""
        event_ret = self.calculate_segment_retention("first_event_type")
        sport_rate = event_ret.loc[event_ret["first_event_type"] == "спорт", "retention_rate_%"].values[0]
        concert_rate = event_ret.loc[event_ret["first_event_type"] == "концерты", "retention_rate_%"].values[0]

        top10_regions = self.user_profiles_clean["first_region"].value_counts().head(10).index
        df_p = self.user_profiles_clean.copy()
        df_p["region_group"] = np.where(df_p["first_region"].isin(top10_regions), "топ-10 регионов", "прочие регионы")
        reg_group = df_p.groupby("region_group")["is_two"].mean() * 100

        return {
            "sport_retention": sport_rate,
            "concert_retention": concert_rate,
            "hyp1_confirmed": sport_rate > concert_rate,
            "top10_reg_retention": round(reg_group["топ-10 регионов"], 2),
            "other_reg_retention": round(reg_group["прочие регионы"], 2),
            "hyp2_confirmed": reg_group["топ-10 регионов"] > reg_group["прочие регионы"],
        }

    def analyze_revenue_cohorts(self):
        """Сравнение параметров чеков когорт (1 заказ vs 2+ заказов)."""
        stats = self.user_profiles_clean.groupby("is_two")["avg_revenue_rub"].agg(
            users_count="count",
            mean="mean",
            median="median",
            p25=lambda x: x.quantile(0.25),
            p75=lambda x: x.quantile(0.75),
            std="std",
        ).reset_index()
        stats["cohort"] = stats["is_two"].map({0: "1 заказ", 1: ">= 2 заказов"})
        return stats[["cohort", "users_count", "mean", "median", "p25", "p75", "std"]].round(2)

    def analyze_day_of_week(self):
        """Расчет активности первого визита и удержания по дням недели."""
        df_p = self.user_profiles_clean.copy()
        df_p["day_of_week_num"] = df_p["first_order_dt"].dt.dayofweek
        days_map = {
            0: "Понедельник", 1: "Вторник", 2: "Среда",
            3: "Четверг", 4: "Пятница", 5: "Суббота", 6: "Воскресенье",
        }
        df_p["day_of_week"] = df_p["day_of_week_num"].map(days_map)

        dow = df_p.groupby(["day_of_week_num", "day_of_week"], as_index=False).agg(
            users_count=("user_id", "count"),
            repeat_users=("is_two", "sum"),
            retention_rate=("is_two", "mean"),
        )
        dow["users_share_%"] = (dow["users_count"] / len(df_p) * 100).round(2)
        dow["retention_rate_%"] = (dow["retention_rate"] * 100).round(2)
        return dow.sort_values(by="day_of_week_num").drop(columns=["day_of_week_num", "retention_rate"]).reset_index(drop=True)

    def run_correlation_analysis(self):
        """Задача 3.2: Корреляционный анализ phi_k по 4 сегментам заказов."""
        df_p = self.user_profiles_clean.copy()
        df_p["day_of_week"] = df_p["first_order_dt"].dt.day_name()

        # Разметка 4 бизнес-сегментов
        conditions = [
            df_p["total_orders"] == 1,
            df_p["total_orders"] == 2,
            (df_p["total_orders"] >= 3) & (df_p["total_orders"] <= 4),
            df_p["total_orders"] >= 5,
        ]
        choices = ["1 заказ", "2 заказа", "от 3 до 4 заказов", "от 5 и выше"]
        df_p["orders_segment"] = np.select(conditions, choices, default="другое")

        cols_seg = [
            "first_device", "first_event_type", "first_region",
            "day_of_week", "avg_revenue_rub", "orders_segment",
        ]
        phik_seg = df_p[cols_seg].phik_matrix(interval_cols=["avg_revenue_rub"])
        return phik_seg["orders_segment"].sort_values(ascending=False).round(3)