# План миграции на V3 schema

Этот документ ведётся в течение всей миграции и отражает её состояние.

## 1. Снимок начального состояния

- Schema mode: **V1**
- Нод всего: **7** (все на V1 через `NODE_CLASS_MAPPINGS`).
- Vendored директории: `cosyvoice/`, `matcha/` (upstream Alibaba CosyVoice + Matcha-TTS — не трогать).
- Не vendored: `configs/` фактически содержит только наш `ts_emotion_presets.json`.
- `requires-comfyui`: **не задано** в `pyproject.toml` (поднимается до `>=0.3.40` в Phase 1).
- Пакетная версия: **0.6.0**.
- `CLAUDE.md` статус: **V1-only с явным запретом миграции** → переводится в transitional (Phase 1) → V3-only (Phase 4).
- `comfy_api.v0_0_2` подтверждён доступным в текущей установке ComfyUI.
- `tests/`, `docs/`, `workflows/` — отсутствовали, создаются в Phase 1.

## 2. Список нод (Phase 0)

| # | node_id | display | category | OUTPUT_NODE | stateful | сложность |
|---|---------|---------|----------|-------------|----------|-----------|
| 1 | `TS_CosyVoice3_ModelLoader` | TS CosyVoice Model Loader | TS CosyVoice3/Loaders | – | module-level `_MODEL_CACHE` в `utils/ts_model_manager.py` | средняя |
| 2 | `TS_CosyVoice3_Instruct2` | TS CosyVoice Text to Voice | TS CosyVoice3/Synthesis | – | module-level `INSTRUCT_PRESETS` (read-only) | средняя |
| 3 | `TS_CosyVoice3_SpeakerInstruct2` | TS CosyVoice Speaker Text To Voice | TS CosyVoice3/Synthesis | – | module-level `INSTRUCT_PRESETS` (read-only) | средняя |
| 4 | `TS_CosyVoice3_CrossLingual` | TS CosyVoice Cross-Language | TS CosyVoice3/Synthesis | – | None | низкая |
| 5 | `TS_CosyVoice3_VoiceConversion` | TS CosyVoice Voice To Voice | TS CosyVoice3/Synthesis | – | None | высокая (chunking + crossfade + pitch shift) |
| 6 | `TS_CosyVoice3_SaveSpeaker` | TS CosyVoice Save Speaker | TS CosyVoice3/Utilities | **True** | module-level `_WHISPER_MODEL` в `utils/ts_whisper_utils.py` | средняя |
| 7 | `TS_CosyVoice3_Dialog` | TS CosyVoice Dialog | TS CosyVoice3/Synthesis | – | module-level `_WHISPER_MODEL` (через utils) | высокая (multi-speaker + Whisper + parsing) |

## 3. Custom IO types

- `COSYVOICE_MODEL` — единственный custom тип пака. Используется как выход у `ModelLoader` и как вход у остальных 6 нод. После Phase 2 объявляется один раз в `nodes/_v3_types.py` (`CosyVoiceModel = IO.Custom("COSYVOICE_MODEL")`).

## 4. План миграции по нодам (Phase 3)

Порядок продиктован принципом «paired loader + one consumer first»:

1. **`TS_CosyVoice3_ModelLoader` + `TS_CosyVoice3_Instruct2`** (paired) — статус: **pending**.
   Loader без потребителя непроверяем — мигрируем в одном коммите.
2. **`TS_CosyVoice3_SpeakerInstruct2`** — статус: pending. Похож на Instruct2 по структуре.
3. **`TS_CosyVoice3_CrossLingual`** — статус: pending. Самая простая нода синтеза.
4. **`TS_CosyVoice3_VoiceConversion`** — статус: pending. Высокая сложность из-за chunking-логики, но stateless.
5. **`TS_CosyVoice3_SaveSpeaker`** — статус: pending. `OUTPUT_NODE = True` — переносится через `is_output_node=True` + `IO.NodeOutput(saved_path, ui=...)` (если хотим UI feedback).
6. **`TS_CosyVoice3_Dialog`** — статус: pending. Многовыходная (6 outputs), требует особого внимания к порядку RETURN_TYPES.

После каждой ноды:
- Перезапуск pytest (`tests/test_node_contracts.py`).
- Перезапуск `python -m compileall .` для всего пакета.
- В коммит-сообщении явно указывается, какие контракты сохранены.

## 5. Прогресс

- [x] **Phase 0** (детектирование состояния) — 2026-05-07.
- [x] **Phase 1** (safety net + ADR + transitional `CLAUDE.md`) — 2026-05-07.
- [x] **Phase 2** (shared primitives `_v3_types.py` + `ComfyExtension` scaffolding) — 2026-05-07.
- [x] **Phase 3a** Migrate paired ModelLoader + Instruct2 — 2026-05-07.
- [x] **Phase 3b** Migrate SpeakerInstruct2 — 2026-05-07.
- [x] **Phase 3c** Migrate CrossLingual — 2026-05-07.
- [x] **Phase 3d** Migrate VoiceConversion — 2026-05-07.
- [x] **Phase 3e** Migrate SaveSpeaker (OUTPUT_NODE) — 2026-05-07.
- [x] **Phase 3f** Migrate Dialog — 2026-05-07.
- [x] **Phase 4** (cleanup, версия 0.6.0 → 0.7.0, README, CHANGELOG, V3-only `CLAUDE.md`) — 2026-05-07.

**Итого:** 7 нод мигрировано, V1 mappings удалены, контрактные тесты зелёные (7/7).

## 6. Ограничения текущей safety net

Workflow JSON regression-тесты **отсутствуют** — пользователь не предоставил
сохранённые workflow до миграции. Покрытие safety net ограничено:

- статическим контрактным тестом `tests/test_node_contracts.py` против
  `tests/contracts/v1_baseline.json` (проверяет `node_id`, имена входов/выходов,
  порядок `RETURN_TYPES`, `is_output_node`).
- `python -m compileall .` для синтаксической корректности.

Перед merge Phase 4 пользователь должен вручную прогнать в ComfyUI каждый workflow,
который у него уже есть, и убедиться, что граф загружается без красных рамок и
что результаты совпадают с pre-migration.

## 7. Ссылки

- Промпт миграции: см. сообщение пользователя, давшее старт работе.
- ADR: [docs/adr/0001-migrate-v1-to-v3.md](docs/adr/0001-migrate-v1-to-v3.md).
- Baseline контрактов: [tests/contracts/v1_baseline.json](tests/contracts/v1_baseline.json).
- Контрактный тест: [tests/test_node_contracts.py](tests/test_node_contracts.py).
- `CLAUDE.md` (правила проекта): [CLAUDE.md](CLAUDE.md).
