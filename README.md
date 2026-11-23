![Поток-М: Edge](docs/banner.svg)

# potok-m-edge

Бортовой инференс на Raspberry Pi 4 B: приём запроса от STM32, запуск модели, отдача поправки и диагностики.

## Граница

Raspberry Pi — советчик. STM32 владеет фазами полёта, ПИД, интерлоками и сервами. На двигателе ИИ выключен; три пропуска или `ai_healthy = 0` отключают ассистента до конца полёта.

## Что внутри

- `src/model_v3.rs` — канонический читатель v3 export (`POTOKM3\0`) с проверкой SHA-256; v2-читатель — `src/format_v2.rs`.
- `src/speculative_v3.rs` — student каждый кадр, teacher по исключению, чистый watchguard (одна поправка, latch после трёх пропусков); v2-диспетчер остаётся в `src/speculative.rs`.
- `src/calibration_v3.rs` — temperature/isotonic/split-conformal калибровка confidence; порог из JSON или `1/(1+q_hat)`.
- `src/lowlat.rs` — профиль UART 42/16 байт на 2 Мбит/с.
- `tests/` — побайтовая сверка с Python, safety-тесты, детерминированные замеры hot path (`tests/lowlat_perf_hard_tests.rs`).

## Контракт

Вход: 14 физических признаков + 6 conditioning (`mass_kg`, `area_m2`, `cd0`, `static_margin_cal`, `thrust_peak_n`, `gyro_noise`). Выход: две поправки в м/с², confidence, `ai_healthy`. Калибровочный порог data-derived — из `calibration.json` для temperature/isotonic и `1/(1+q_hat)` для split-conformal. Читатель есть в библиотеке, но dispatch пока берёт свой порог `DEFAULT_CONFIDENCE_THRESHOLD = 0.7` и не потребляет `Calibration` (§4 контракта). Файл модели в репозиторий не попадает: бинарный export приходит из `potok-m-ai` при деплое и загружается в RAM; в репозитории только детерминированные тестовые фикстуры.

## Запуск

```sh
cargo fmt --all -- --check
cargo clippy --all-targets
cargo test --release --all-targets
```

systemd-юнит закреплён на отдельном ядре с SCHED_FIFO и `CAP_SYS_NICE`.

## Статус

Хост-тесты зелёные в release; замер p99 на реальном Pi не проведён, NEON только compile-checked. Детерминированные lowlat perf gates (`tests/lowlat_perf_hard_tests.rs`) проверяют измеритель, а не время Pi; в debug-сборке один integration-тест даёт p99 ≈ 11.2 ms против бюджета 10 ms (release зелёный). Модель не прошла лётную квалификацию до положительного closed-loop A/B. Подробности и честные оговорки — `docs/INFERENCE_CONTRACT.md`.

## Репозитории

[potok-m-ai] · [potok-m-protocol] · [potok-m-sim] · [potok-m-firmware] · [potok-m]

Лицензия: GPL-3.0-or-later (или коммерческая).
