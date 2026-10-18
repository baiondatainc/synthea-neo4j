# Future Advisor — Ask Alpha question set

Questions the `futureadvisor-cypher` model and the DoF knowledge graph answer today. Grouped by what the graph holds; every group lists the labels and the provenance each answer carries. Use the plain-text block at the end as `data/faqs/dof.txt` for `python main.py warm`.

Graph as imported 2026-10-18: `Kpi`, `Country` (ISO alpha-3 + IMF groups), `Source`, `Series` (summaries; `table` = Postgres table, `measure` = indicator), `KpiObservation`, `KpiObservationRevision`, `Dataset`, `Document`, `IngestRun`, `DeScenarioDriver`, `DeScenarioDriverValue`, `DeScenarioElasticity`, `DeMeasure`, `ScenarioEngineRun`, `ScenarioEngineConfig`, `BacktestRun`, `BspCycle`, `BspStage`.

---

## 1. UAE / Dubai macro — `Series` → `Country {code:'ARE'}`

| # | Question | Returns |
|---|---|---|
| 1 | Latest UAE real GDP growth | value, period, source table, since |
| 2 | What is inflation in the UAE? | cpi_yoy and cpi_monthly_yoy, latest, period |
| 3 | UAE unemployment rate and population | latest per indicator |
| 4 | UAE policy rate from BIS | latest, period, since, months of history |
| 5 | UAE fiscal balance and government debt as share of GDP | pct of GDP per indicator |
| 6 | Which indicators do we have for the UAE and from which tables? | indicator × table, first/last period, points |
| 7 | UAE exports and FDI from the World Bank | latest per indicator |
| 8 | Since when do we have UAE data? | earliest, latest period, series count |
| 9 | UAE nominal GDP and GDP per capita | latest, unit |
| 10 | UAE current account balance | pct of GDP, period |

## 2. United States — `Series` → `Country {code:'USA'}`

| # | Question | Returns |
|---|---|---|
| 11 | What is the current US policy rate? | latest, period, table (gold_fred) |
| 12 | US 10-year treasury yield latest and range | latest, min, max, mean |
| 13 | US dollar index now | latest, period |
| 14 | US inflation monthly | cpi_monthly_yoy from OECD |
| 15 | US GDP growth, inflation and unemployment from the IMF | three indicators |
| 16 | Everything we track for the United States | indicator × table |
| 17 | Fed rate versus UAE rate | policy_rate for USA and ARE |
| 18 | How much US data history do we have on rates? | first/last period, n_points |

## 3. Comparisons, partners, oil

| # | Question | Returns |
|---|---|---|
| 19 | Compare GDP growth for UAE, Saudi Arabia and the US | ARE, SAU, USA ordered |
| 20 | GDP growth of Dubai's trading partners | IND, CHN, SAU, USA, GBR |
| 21 | Advanced versus emerging economies growth | ADVANCED, EMERGING, DEVELOPING |
| 22 | Inflation across all countries, highest first | top 50 |
| 23 | GDP growth of Germany and Japan | DEU, JPN |
| 24 | Which countries are in the graph? | code, series count |
| 25 | How many countries have IMF data? | count |
| 26 | Brent oil price latest and range | latest, min, max (no country) |
| 27 | Oil price forecast from EIA | brent and WTI forecast horizon |
| 28 | WTI versus Brent spot | both latest |
| 29 | Euro-dollar exchange rate | fx_eur_usd from ECB |
| 30 | Which Gulf countries have policy-rate data? | ARE, SAU, QAT, KWT, BHR, OMN |

## 4. Dubai fiscal — `Series` where `table STARTS WITH 'gold.de_'`

| # | Question | Returns |
|---|---|---|
| 31 | Total revenue and total expenditure history | first/last period, latest, mean (AED m) |
| 32 | Revenue by entity, budget versus actual | entity_code × measure |
| 33 | Expenditure by entity, recurrent and capital | entity_code × measure |
| 34 | Spending on health and education | COFOG 07, 09 |
| 35 | Revenue accounts — fees and VAT share | account_code × measure |
| 36 | Dubai GDP by sector | sector_code × measure |
| 37 | Which Dubai fiscal series do we have? | table, count, period range |
| 38 | Dubai macro baseline used by the engine | indicator, latest, horizon |
| 39 | Scenario lines under baseline and downside | measure, latest, horizon |
| 40 | Expenditure by function — all COFOG codes | cofog_code × latest |
| 41 | Which entities have both revenue and expenditure data? | entity codes |
| 42 | Policy measure impacts under each scenario | de_measure_impact summaries |

## 5. Observations and revisions — `KpiObservation`, `KpiObservationRevision`

| # | Question | Returns |
|---|---|---|
| 43 | Latest inflation observation for the UAE | value, period, source, retrieved_at |
| 44 | US observations recorded this year | indicator, period, value, source |
| 45 | GDP growth observations for the UAE by year | year, value, source |
| 46 | Observations below 0.8 confidence not yet validated | review queue |
| 47 | Which values were revised and by how much? | old, new, source, revised_at |
| 48 | How many observations per source? | source, count |
| 49 | Observations for Saudi Arabia from the IMF | indicator, period, value |
| 50 | Latest observation per indicator for the UAE | one row per kpi |

## 6. Sources and pipeline — `Source`, `IngestRun`, `Dataset`, `Document`

| # | Question | Returns |
|---|---|---|
| 51 | List all data sources and their status | code, name, kind, tier, enabled, last status |
| 52 | Which sources publish inflation for Saudi Arabia? | source, indicator, tier |
| 53 | Which data sources are overdue or failed? | source, status, started |
| 54 | Last ingest run per source | status, finished, inserted, updated, revised |
| 55 | Which datasets cover the UAE? | table, source, kpi codes, period range, rows |
| 56 | Documents fetched from DSC | title, kind, parse status |
| 57 | What does each KPI mean and in what unit? | code, name, unit, frequency, category |
| 58 | Which sources are tier 1? | T1 sources |
| 59 | When is the next expected release per source? | next_expected_release |
| 60 | How many rows did the last IMF ingest insert? | rows_inserted |

## 7. Scenarios, drivers, engine — `DeScenarioDriver*`, `DeScenarioElasticity`, `DeMeasure`, `*Run`, `*Config`

| # | Question | Returns |
|---|---|---|
| 61 | Which drivers can the scenario builder move? | code, label, unit, min, max, step |
| 62 | Driver values under the downside scenario | driver, value, unit |
| 63 | Compare driver values baseline versus downside | driver × scenario |
| 64 | How sensitive is each fiscal line to its drivers? | line, driver, elasticity |
| 65 | Policy measures available | id, measure, family, sector, rate |
| 66 | Latest scenario engine runs | run id, preset, focus year, created by |
| 67 | Backtest results | run id, target, variant, metrics |
| 68 | Which engine configurations are current? | config id, version, built_at |
| 69 | Which drivers belong to the oil price group? | drivers in driver_group |
| 70 | Which lines does the oil price driver affect? | elasticities on that driver |

## 8. BSP cycle — `BspCycle`, `BspStage`

| # | Question | Returns |
|---|---|---|
| 71 | BSP stages and their owners | cycle, stage, milestone, owner, due, status |
| 72 | Overdue BSP stages | due < now and not done |
| 73 | Which stages had rework loops? | rework_loops > 0 |
| 74 | What is due in the next 30 days? | stages with due_at within 30 days |

## 9. Arabic

| # | Question | Equivalent |
|---|---|---|
| 75 | ما هو آخر معدل تضخم في الإمارات؟ | #2 |
| 76 | نمو الناتج المحلي في الإمارات | #1 |
| 77 | سعر الفائدة الأمريكي الحالي | #11 |
| 78 | إيرادات الجهات الحكومية في دبي | #32 |
| 79 | آخر سعر لخام برنت | #26 |
| 80 | ما هي مصادر البيانات المتأخرة؟ | #53 |
| 81 | مقارنة نمو الناتج بين الإمارات والسعودية وأمريكا | #19 |
| 82 | الإنفاق على الصحة والتعليم | #34 |

---

## Not answerable from the graph (by design)

| Question type | Why | Route |
|---|---|---|
| "GDP growth 2015–2029 year by year" | Series are summaries; points stay in PostgreSQL | SQL route (`gold.*` tables) |
| "Growth rate between 2020 and 2024", "VAT share of revenue" | Computation | Engine / SQL route |
| "Forecast revenue under an oil shock" | Needs a run, not a lookup | Scenario Engine |
| "Why did Fees fall?" | Attribution lives in run tables not yet imported | After run tables + graph sync |
| Insurance, patients, policies | Outside allowed topics | Blocked by guardrail |

## Expected weak spots

Labels with zero rows in the pilot graph return empty results, not errors: `DeScenarioElasticity`, `DeMeasure`, `BspCycle`/`BspStage`, `ScenarioEngineRun`, `BacktestRun` (as of the 2026-10-18 import). Questions 64–74 will come alive when those tables are populated and the graph rebuilt.

---

## Plain list for `data/faqs/dof.txt`

```
Latest UAE real GDP growth
What is inflation in the UAE?
UAE unemployment rate and population
UAE policy rate from BIS
UAE fiscal balance and government debt as share of GDP
Which indicators do we have for the UAE and from which tables?
UAE exports and FDI from the World Bank
Since when do we have UAE data?
UAE nominal GDP and GDP per capita
UAE current account balance
What is the current US policy rate?
US 10-year treasury yield latest and range
US dollar index now
US inflation monthly
US GDP growth, inflation and unemployment from the IMF
Everything we track for the United States
Fed rate versus UAE rate
How much US data history do we have on rates?
Compare GDP growth for UAE, Saudi Arabia and the US
GDP growth of Dubai's trading partners
Advanced versus emerging economies growth
Inflation across all countries, highest first
GDP growth of Germany and Japan
Which countries are in the graph?
How many countries have IMF data?
Brent oil price latest and range
Oil price forecast from EIA
WTI versus Brent spot
Euro-dollar exchange rate
Which Gulf countries have policy-rate data?
Total revenue and total expenditure history
Revenue by entity, budget versus actual
Expenditure by entity, recurrent and capital
Spending on health and education
Revenue accounts — fees and VAT share
Dubai GDP by sector
Which Dubai fiscal series do we have?
Dubai macro baseline used by the engine
Scenario lines under baseline and downside
Expenditure by function — all COFOG codes
Which entities have both revenue and expenditure data?
Policy measure impacts under each scenario
Latest inflation observation for the UAE
US observations recorded this year
GDP growth observations for the UAE by year
Observations below 0.8 confidence not yet validated
Which values were revised and by how much?
How many observations per source?
Observations for Saudi Arabia from the IMF
Latest observation per indicator for the UAE
List all data sources and their status
Which sources publish inflation for Saudi Arabia?
Which data sources are overdue or failed?
Last ingest run per source
Which datasets cover the UAE?
Documents fetched from DSC
What does each KPI mean and in what unit?
Which sources are tier 1?
When is the next expected release per source?
How many rows did the last IMF ingest insert?
Which drivers can the scenario builder move?
Driver values under the downside scenario
Compare driver values baseline versus downside
How sensitive is each fiscal line to its drivers?
Policy measures available
Latest scenario engine runs
Backtest results
Which engine configurations are current?
Which drivers belong to the oil price group?
Which lines does the oil price driver affect?
BSP stages and their owners
Overdue BSP stages
Which stages had rework loops?
What is due in the next 30 days?
ما هو آخر معدل تضخم في الإمارات؟
نمو الناتج المحلي في الإمارات
سعر الفائدة الأمريكي الحالي
إيرادات الجهات الحكومية في دبي
آخر سعر لخام برنت
ما هي مصادر البيانات المتأخرة؟
مقارنة نمو الناتج بين الإمارات والسعودية وأمريكا
الإنفاق على الصحة والتعليم
```
