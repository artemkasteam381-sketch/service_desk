from src.analytics.data_preparation import prepare_data
from src.analytics.eda_basic import (
    basic_kpi,
    processing_time_analysis,
)
from src.analytics.line_analysis import (
    analyze_lines,
    line_work_time,
)
from src.analytics.clarification_analysis import (
    clarification_analysis,
    clarification_distribution,
)
from src.analytics.sla_risk import calculate_risk, line_risk
from src.analytics.filters import filter_data


def test_prepared_data():
    df = prepare_data()

    assert len(df) == 1931
    assert df["Номер запроса"].is_unique
    assert int(df["is_overdue"].sum()) == 25
    assert int(df["sla_actual_seconds"].isna().sum()) == 1
    assert (df["sla_actual_seconds"].dropna() >= 0).all()

    assert (
        df["recorded_lines_count"]
        .value_counts()
        .sort_index()
        .to_dict()
        == {0: 372, 1: 945, 2: 502, 3: 108, 4: 4}
    )


def test_basic_kpi():
    result = basic_kpi(prepare_data())

    assert result.iloc[0]["Значение"] == 1931
    assert result.iloc[1]["Значение"] == 25
    assert result.iloc[2]["Значение"] == 1.29


def test_processing_time_totals():
    df = prepare_data()

    for column, title in [
        ("service_analytics", "Услуга"),
        ("Вид запроса", "Вид запроса"),
        ("Приоритет", "Приоритет"),
    ]:
        result = processing_time_analysis(df, column, title)

        assert int(result["Обращения"].sum()) == 1931
        assert int(result["Обращения_с_временем"].sum()) == 1930
        assert int(result["Пропуски_времени"].sum()) == 1
        assert (
            result["Малая выборка"]
            == result["Обращения"].lt(30)
        ).all()


def test_line_analytics():
    df = prepare_data()
    participation = analyze_lines(df)

    counts = dict(
        zip(
            participation[
                "Количество линий с зафиксированным временем"
            ],
            participation["Количество обращений"],
        )
    )

    assert counts == {0: 372, 1: 945, 2: 502, 3: 108, 4: 4}

    work_time = line_work_time(df)
    assert work_time["С зафиксированным временем"].tolist() == [
        1399, 702, 183, 5
    ]


def test_clarification_analytics():
    df = prepare_data()

    summary = clarification_analysis(df)
    distribution = clarification_distribution(df)

    assert int(summary["Обращения"].sum()) == 1931
    assert int(distribution["Количество обращений"].sum()) == 1931
    assert distribution["Количество обращений"].tolist() == [
        847, 700, 221, 88, 75
    ]


def test_sla_risk():
    df = prepare_data()

    service_risk = calculate_risk(df, "service_analytics")
    assert int(service_risk["Обращения"].sum()) == 1931
    assert int(service_risk["Просрочки"].sum()) == 25

    expected = (
        service_risk["Просрочки"]
        / service_risk["Обращения"]
        * 100
    ).round(2)

    assert service_risk["Риск SLA (%)"].equals(expected)

    lines = line_risk(df)
    assert len(lines) == 4
    assert lines.loc[
        lines["Линия"].eq("4 линия"),
        "Малая выборка",
    ].all()



def test_filters():
    df = prepare_data()

    assert len(filter_data(df)) == 1931
    assert len(filter_data(
        df,
        services=["Портал pochta.ru"],
    )) == 620
    assert len(filter_data(
        df,
        request_types=["Личный кабинет"],
    )) == 44
    assert len(filter_data(
        df,
        priorities=["(2) Высокий"],
    )) == 111
    assert len(filter_data(
        df,
        lines=["4 линия"],
    )) == 5

    first_date = df["Дата регистрации"].min().date()
    expected = df["Дата регистрации"].dt.date.eq(first_date).sum()

    assert len(filter_data(
        df,
        start_date=first_date,
        end_date=first_date,
    )) == expected


def test_data_quality_flags():
    from src.analytics.data_quality import (
        build_quality_flags,
        issue_records,
        quality_summary,
    )

    df = prepare_data()
    flags = build_quality_flags(df)

    counts = (
        quality_summary(flags)
        .set_index("Проверка")["Количество"]
        .to_dict()
    )

    assert counts["Нет длительности SLA"] == 1
    assert counts["Неопределённая услуга"] == 19
    assert counts["Нет времени линий"] == 372
    assert counts["Уточнения без времени"] == 57
    assert counts["Расхождение с длительностью SLA"] == 5
    assert counts["Расхождение с крайним сроком"] == 3
    assert len(issue_records(flags)) == 431



def test_monthly_dynamics():
    from src.analytics.time_dynamics import monthly_dynamics

    result = monthly_dynamics(prepare_data())

    assert result["Месяц"].tolist() == [
        "2025-01",
        "2025-02",
        "2025-03",
        "2025-04",
        "2025-05",
        "2025-06",
        "2025-07",
        "2025-08",
    ]

    assert int(result["Обращения"].sum()) == 1931
    assert int(result["Просрочки"].sum()) == 25
    assert int(result["Пропуски_времени"].sum()) == 1
    assert int(result["Нет данных"].sum()) == 3
    assert int(result["Малая выборка"].sum()) == 3
