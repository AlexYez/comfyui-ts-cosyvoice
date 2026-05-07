# CLAUDE.md — Engineering Rules for comfyui-ts-cosyvoice

Этот файл — операционные правила для Claude (и других AI-ассистентов) при работе над `comfyui-ts-cosyvoice` (TS CosyVoice3). Пак на ComfyUI **V3 API** (`IO.ComfyNode` / `define_schema` / `execute` / `IO.NodeOutput`, регистрация через `ComfyExtension.get_node_list()`). Миграция V1 → V3 завершена в релизе 0.7 — см. [CHANGELOG.md](CHANGELOG.md), [MIGRATION_PLAN.md](MIGRATION_PLAN.md) и [docs/adr/0001-migrate-v1-to-v3.md](docs/adr/0001-migrate-v1-to-v3.md). V1-паттерны больше **не используются** в коде пака.

Главная цель: **максимальное качество итогового кода без поломки старых ComfyUI workflows и без расхождения с upstream CosyVoice**.

Кредо проекта:

> Readable. Stable. Testable.

ComfyUI-кредо:

> Stability. Identity. Scalability. Predictability.

Принцип рефакторинга:

> Primum non nocere — first, do no harm.

Принцип vendored-кода:

> Don't fork what you can pull.

Язык общения с пользователем — **русский**. Код, идентификаторы, docstrings — английский.

---

## 0. Иерархия инструкций

Этот `CLAUDE.md` — единственный источник правил для всего пакета. Если когда-нибудь появится `nodes/AGENTS.md` или `doc/AGENTS.md` — он лишь уточняет; конфликт решается в пользу `CLAUDE.md`.

При конфликте: workflow compatibility > upstream parity (vendored) > безопасность > тесты > всё остальное.

---

## 1. Контекст проекта

`comfyui-ts-cosyvoice` — production-quality пак custom nodes для ComfyUI поверх семейства моделей **CosyVoice 3** (TTS / voice cloning / cross-lingual / voice conversion / dialog).

- Версия: `0.7.0` (`pyproject.toml`).
- Репозиторий: https://github.com/AlexYez/comfyui-ts-cosyvoice.
- Основная модель: `Fun-CosyVoice3-0.5B` (HuggingFace / ModelScope).
- 7 нод, все на ComfyUI **V3 API** (`IO.ComfyNode` + `define_schema` + `execute` + `IO.NodeOutput`, регистрация через `ComfyExtension.get_node_list()`).
- Все V3 импорты идут через `from comfy_api.v0_0_2 import IO, ComfyExtension`. **Никогда** не импортировать из `comfy_api.latest` — это unstable алиас.
- `requires-comfyui` в `pyproject.toml`: `>=0.3.40` (минимальная версия со стабильным `comfy_api.v0_0_2`).
- Корневой загрузчик: [`__init__.py`](__init__.py) — явный список импортов нод, `_V3_NODES`, `_PackExtension` и `comfy_entrypoint()`. `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS` удалены при cleanup'е миграции — ComfyUI регистрирует пак автоматически через `comfy_entrypoint()`.
- Custom IO type `COSYVOICE_MODEL` объявляется один раз в [`nodes/_v3_types.py`](nodes/_v3_types.py) (`CosyVoiceModel = IO.Custom("COSYVOICE_MODEL")`) и импортируется во все 6 нод-потребителей. Дублирование `IO.Custom("COSYVOICE_MODEL")` в нескольких файлах — баг рассинхронизации.
- Frontend (`js/`) **отсутствует** — `WEB_DIRECTORY` не объявлен.
- Логгер: [`utils/ts_logging.py`](utils/ts_logging.py), фабрика `get_logger(name)` с префиксом `[TS CosyVoice3 ...]`.

Layout (актуальный):

```text
comfyui-ts-cosyvoice/
├─ CLAUDE.md
├─ MIGRATION_PLAN.md      # история миграции V1 → V3 (для архивных целей)
├─ CHANGELOG.md           # релизные заметки, BREAKING помечены
├─ __init__.py            # _PackExtension + comfy_entrypoint() (V3-only после релиза 0.7.0)
├─ pyproject.toml         # name, version, dependencies, ComfyRegistry metadata, requires-comfyui
├─ pytest.ini
├─ requirements.txt
├─ README.md              # двуязычный (RU / EN)
├─ icon.svg
├─ docs/
│  └─ adr/
│     └─ 0001-migrate-v1-to-v3.md
├─ nodes/                 # 7 публичных нод, плоский layout
│  ├─ _v3_types.py        # IO.Custom("COSYVOICE_MODEL") — единственное объявление в паке
│  ├─ ts_cosyvoice_model_loader_node.py
│  ├─ ts_cosyvoice_text_to_voice_node.py
│  ├─ ts_cosyvoice_save_speaker_node.py
│  ├─ ts_cosyvoice_speaker_text_to_voice_node.py
│  ├─ ts_cosyvoice_cross_language_node.py
│  ├─ ts_cosyvoice_voice_to_voice_node.py
│  └─ ts_cosyvoice_dialog_node.py
├─ utils/                 # пакет helpers (логирование, audio prep, model manager и т.п.)
│  └─ ts_logging.py
├─ tests/
│  ├─ contracts/
│  │  └─ v1_baseline.json # snapshot контрактов V1 (источник истины для contract preservation)
│  ├─ workflows/          # пользовательские workflow JSON для regression-тестов (если есть)
│  └─ test_node_contracts.py
├─ configs/               # локальные конфиги (например, ts_emotion_presets.json — наш файл)
├─ cosyvoice/             # vendored upstream код CosyVoice (Alibaba) — не трогать
├─ matcha/                # vendored upstream код Matcha-TTS (HiFi-GAN и пр.) — не трогать
├─ workflows/             # пример ComfyUI workflows (.json)
├─ img/                   # README screenshots
└─ .github/workflows/     # ComfyRegistry publish
```

Ноды (7 шт.) и публичные контракты:

| Class (V1)                           | node_id                              | Display name                       | Category                       |
|--------------------------------------|--------------------------------------|------------------------------------|--------------------------------|
| `TS_CosyVoice3_ModelLoader`          | `TS_CosyVoice3_ModelLoader`          | `TS CosyVoice Model Loader`        | `TS CosyVoice3/Loaders`        |
| `TS_CosyVoice3_Instruct2`            | `TS_CosyVoice3_Instruct2`            | `TS CosyVoice Text to Voice`       | `TS CosyVoice3/Synthesis`      |
| `TS_CosyVoice3_SaveSpeaker`          | `TS_CosyVoice3_SaveSpeaker`          | `TS CosyVoice Save Speaker`        | `TS CosyVoice3/Utilities`      |
| `TS_CosyVoice3_SpeakerInstruct2`     | `TS_CosyVoice3_SpeakerInstruct2`     | `TS CosyVoice Speaker Text To Voice` | `TS CosyVoice3/Synthesis`    |
| `TS_CosyVoice3_CrossLingual`         | `TS_CosyVoice3_CrossLingual`         | `TS CosyVoice Cross-Language`      | `TS CosyVoice3/Synthesis`      |
| `TS_CosyVoice3_VoiceConversion`      | `TS_CosyVoice3_VoiceConversion`      | `TS CosyVoice Voice To Voice`      | `TS CosyVoice3/Synthesis`      |
| `TS_CosyVoice3_Dialog`               | `TS_CosyVoice3_Dialog`               | `TS CosyVoice Dialog`              | `TS CosyVoice3/Synthesis`      |

> Категории — точные значения из исходников (формат `TS CosyVoice3/<Subcategory>`, с пробелом и слэшем). Перед изменением реальной строки `CATEGORY` сверься с файлом ноды — это публичный контракт, влияющий на поиск в UI.

Loader-правила:

- `__init__.py` импортирует ноды **по имени файла**, без рекурсивного скана. При добавлении новой ноды правится сам `__init__.py`: добавляется `from .nodes.ts_cosyvoice_<feature>_node import TS_CosyVoice3_<Name>` плюс две записи в `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`.
- `__init__.py` поддерживает оба способа загрузки (relative и absolute) — обе ветки `try/except ImportError` обновляются синхронно.
- `sys.path.insert(0, os.path.dirname(__file__))` нужен, чтобы vendored `cosyvoice/`, `matcha/`, `configs/` импортировались как top-level пакеты (так же, как они написаны в upstream). Не убирать.

---

## 2. Языковые правила

Обязательно:

- Всегда отвечай пользователю **на русском**.
- Все планы, пояснения, отчёты, риски, verification summary — на русском.
- Не заявляй, что задача завершена, если проверки не запускались (или явно объясни, почему).
- Если тесты невозможно запустить — честно скажи и дай точные команды для локальной проверки.
- Для сложных задач сначала дай короткий план.
- Для рефакторинга всегда пиши: **Анализ / Риски / Изменение / Проверка**.

Разрешено английским:

- Код, имена файлов/классов/переменных, docstrings.
- Технические идентификаторы ComfyUI / Python / PyTorch / CosyVoice.
- `tooltip` пишем двуязычно: короткое English-описание + русский перевод (как уже сделано в существующих нодах).

---

## 3. Operating Loop: Plan → Implement → Verify → Review

Для любой нетривиальной задачи:

1. **Plan**
   - Прочитай релевантные файлы. Для нод — обязательно сам файл ноды + `utils/ts_logging.py` + соответствующие модули в `cosyvoice/cli/` (если задача про синтез).
   - Отметь, является ли затронутый код vendored (`cosyvoice/`, `matcha/`, `configs/`) — если да, см. §4.5.
   - Выпиши публичные контракты: класс, `node_id`, `INPUT_TYPES`, `RETURN_TYPES`, `RETURN_NAMES`, `FUNCTION`, `CATEGORY`, дефолтные значения, custom IO types (`COSYVOICE_MODEL`, `AUDIO`, `STRING`).
   - Выбери стратегию тестирования заранее (CPU smoke vs full GPU run).
   - Выбери минимальное безопасное изменение.

2. **Implement**
   - Делай только запрошенное.
   - Не добавляй unrelated features.
   - Не переписывай рабочий код ради «чистой архитектуры».
   - Сохраняй правило: одна публичная нода = один `.py` в `nodes/`.

3. **Verify**
   - Запусти все доступные релевантные проверки (см. §10).
   - Если тест падает — исправь минимально и повтори.
   - Если проверка невозможна (нет GPU / нет модели / нет интернета) — явно напиши почему.

4. **Review**
   - Перечитай diff.
   - Проверь workflow compatibility (см. §4).
   - Проверь, не задет ли vendored код (см. §4.5).
   - Проверь безопасность, hidden side effects, dead code, tensor mutation.

Если план оказался неверным — остановись и перепланируй.

---

## 4. Non-Negotiable: Workflow Compatibility

Никогда не меняй без явного запроса/миграции:

- Python class names (`TS_CosyVoice3_*`).
- Ключи в `NODE_CLASS_MAPPINGS`.
- Display names в `NODE_DISPLAY_NAME_MAPPINGS`.
- Имена и типы inputs/outputs (включая `RETURN_TYPES`, `RETURN_NAMES`).
- Порядок outputs.
- Имя метода из `FUNCTION`.
- Имена параметров `FUNCTION`-метода (ComfyUI передаёт их по именам).
- Default widget values, диапазоны (`min`/`max`/`step`).
- `CATEGORY`-строки (формат `TS CosyVoice3/<Sub>` — с пробелом, со слэшем).
- Существующую семантику поведения и формат custom типа `COSYVOICE_MODEL`.
- Имена и формат файлов speaker preset в `models/cosyvoice/speaker/`.

Если переименование действительно необходимо:

- Сохрани старый ключ в `NODE_CLASS_MAPPINGS` как alias на новый класс.
- Display name можно менять (не ломает workflow).
- Класс `Instruct2` / `SpeakerInstruct2` назван так из-за upstream API CosyVoice (`inference_instruct2`); переименовывать без обновления привязки к upstream — не надо.
- Добавь docs migration note в README.
- Если меняется формат speaker preset — добавь backward-compat загрузчик старого формата (нельзя ломать пользовательские пресеты).

---

## 4.5. Vendored upstream: `cosyvoice/`, `matcha/`, `configs/`

`cosyvoice/`, `matcha/`, `configs/` — это **vendored copy** из upstream-репозиториев (Alibaba CosyVoice + Matcha-TTS). Они импортируются напрямую с `sys.path` и должны оставаться максимально близко к upstream, чтобы можно было легко подтянуть багфиксы.

Правила работы с vendored кодом:

- **По умолчанию — не править**. Если функционал требует изменений в этих папках, сначала спроси пользователя, действительно ли надо локально патчить, или баг достоин upstream PR.
- **Если правка неизбежна** — оформляй как минимально-инвазивный патч и помечай комментарием:
  ```python
  # TS-PATCH: <короткое объяснение почему>; revert if upstream fixes <ссылка/коммит>.
  ```
- **Не «причесывай»** vendored файлы: не переименовывай, не меняй стиль форматирования, не добавляй type hints в чужой код, не выкидывай неиспользуемые импорты. Любая косметическая правка увеличивает diff против upstream и усложняет последующий merge.
- Импорты CosyVoice в нодах должны идти как `from cosyvoice.cli.cosyvoice import ...`, а не как `from .cosyvoice...` — это требует наличия `sys.path.insert(0, ...)` в `__init__.py`.
- Логика, которая *оборачивает* CosyVoice (download, выбор device, fp16, кэширование модели, prep референс-аудио, post-processing), пишется в `nodes/` или `utils/`, а не внутри `cosyvoice/` / `matcha/`.
- Эталон для обёртки — текущий `TS_CosyVoice3_ModelLoader` и helpers в `utils/`.

---

## 4.6. Обязательные инструменты (skills)

### 4.6.1. Skill `comfyui-custom-nodes`

Перед любой нетривиальной работой над нодой подключай skill **`/comfyui-custom-nodes`**. Это база знаний по ComfyUI custom nodes API. Особенно полезные модули:

- `comfyui-custom-nodes:comfyui-node-basics` — базовая структура ноды (V1 + V3).
- `comfyui-custom-nodes:comfyui-node-inputs` — типы входов, виджеты, скрытые/опциональные/lazy.
- `comfyui-custom-nodes:comfyui-node-outputs` — `RETURN_TYPES`, UI outputs, preview-ноды, OUTPUT_NODE.
- `comfyui-custom-nodes:comfyui-node-datatypes` — IMAGE / MASK / LATENT / **AUDIO** conventions, custom типы.
- `comfyui-custom-nodes:comfyui-node-lifecycle` — caching, `IS_CHANGED`, `VALIDATE_INPUTS`.
- `comfyui-custom-nodes:comfyui-node-advanced` — wildcard inputs, dynamic combos.
- `comfyui-custom-nodes:comfyui-node-packaging` — структура пакета, `__init__.py`, registration.

Не пропускай этот skill для задач, где задеваются: `INPUT_TYPES`, `RETURN_TYPES`, `IS_CHANGED`, custom types (`COSYVOICE_MODEL`), `OUTPUT_NODE`, audio dict-формат.

### 4.6.2. ComfyUI Python для запуска

Системный/test Python, скорее всего, не содержит `torch`, `torchaudio`, `librosa`, `onnxruntime`, `transformers` — без них ноды не загрузятся, а большинство тестов будет скипаться.

Чтобы реально проверить ноды (синтез, voice conversion, GPU inference), используй **ComfyUI portable Python** (на Windows portable обычно `python_embeded/python.exe` рядом с ComfyUI).

```bash
python -m compileall .
python -m pytest -v
```

Этот Python содержит `numpy`, `torch` (с CUDA), `torchaudio`, `librosa`, `transformers`, `huggingface_hub`, `modelscope`, `onnxruntime`, `comfy.model_management`, `folder_paths` и весь runtime ComfyUI.

В verification summary всегда указывай, под каким Python запускались тесты:

```text
Проверено (под ComfyUI portable Python):
- python -m compileall .
- python -m pytest -v
```

### 4.6.3. Скачивание моделей

`TS_CosyVoice3_ModelLoader` уже умеет скачивать `Fun-CosyVoice3-0.5B` через `huggingface_hub` или `modelscope` в `ComfyUI/models/cosyvoice/`. При тестировании:

- Не качай модели в CI без явной необходимости — они тяжёлые (~1.5 ГБ) и upstream может рейтлимитить.
- Для CPU smoke / контрактных тестов модель не нужна — мокируй `cosyvoice.cli.cosyvoice.CosyVoice3` через `monkeypatch`.
- Реальный inference прогоняй локально на машине с уже скачанной моделью.

---

## 5. API: V3 (единственный поддерживаемый schema пака)

### 5.0. Базовые правила V3

- **Импорты только из `comfy_api.v0_0_2`** — `from comfy_api.v0_0_2 import IO, ComfyExtension`. **Никогда** из `comfy_api.latest` — это unstable алиас, который ломается между релизами ComfyUI. (`io`/`ui` в нижнем регистре — это псевдонимы тех же модулей, проект придерживается заглавного `IO`/`UI` для согласованности.)
- **`node_id` — публичный контракт** и должен совпадать с тем, что был в исходных `NODE_CLASS_MAPPINGS` пользовательских workflows. Любые исторические переименования регистрируются через `search_aliases=[old_name_1, old_name_2]`, а не путём смены `node_id`.
- **Custom тип `COSYVOICE_MODEL`** объявляется один раз в `nodes/_v3_types.py` (`CosyVoiceModel = IO.Custom("COSYVOICE_MODEL")`) и импортируется во все потребители. Дублирование `IO.Custom("COSYVOICE_MODEL")` где-либо ещё — запрещено и расценивается как баг рассинхронизации.
- **Module-level state, а не class-level**. V3 классы залочены `comfy_api.internal.lock_class` — `cls.cached_X = ...` упадёт `AttributeError`. Кэши и shared state живут в module-level переменных (`_MODEL_CACHE` в `utils/ts_model_manager.py`, `_WHISPER_MODEL` в `utils/ts_whisper_utils.py`).
- **`execute(cls, ...)`** — `@classmethod`, возвращает `IO.NodeOutput(...)`. Параметры — по именам, совпадающим с `id` входов из `define_schema`.
- **`is_output_node=True`** — только для нод с побочным эффектом на диск (в этом паке — `TS_CosyVoice3_SaveSpeaker`).
- **`validate_inputs(cls, ...)` / `fingerprint_inputs(cls, ...)`** — V3-эквиваленты `VALIDATE_INPUTS` / `IS_CHANGED`.

### 5.1. V3 шаблон ноды

```python
import logging
from comfy_api.v0_0_2 import IO

from utils.ts_logging import get_logger
from .. _v3_types import CosyVoiceModel  # IO.Custom("COSYVOICE_MODEL"), один раз на пакет

LOGGER = get_logger("TS CosyVoice3 Example")
LOG_PREFIX = "[TS CosyVoice3 Example]"


class TS_CosyVoice3_Example(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_CosyVoice3_Example",            # ОБЯЗАН совпадать с прежним V1 ключом
            display_name="TS CosyVoice Example",        # был NODE_DISPLAY_NAME_MAPPINGS value
            category="TS CosyVoice3/Synthesis",         # был CATEGORY (формат сохраняется)
            description="Short English description shown in node search.",
            inputs=[
                CosyVoiceModel.Input("model", tooltip="Загруженная модель CosyVoice из ноды загрузчика."),
                IO.String.Input(
                    "text", default="", multiline=True,
                    tooltip="Текст для синтеза.",
                ),
                IO.Audio.Input("reference_audio", tooltip="Референсное аудио (5–15 c, mono)."),
                IO.Float.Input(
                    "speed", default=1.0, min=0.5, max=2.0, step=0.05, optional=True,
                    tooltip="Скорость речи.",
                ),
            ],
            outputs=[
                IO.Audio.Output(display_name="audio"),
            ],
            search_aliases=[],   # заполнить, если нода когда-либо переименовывалась
            is_output_node=False,
        )

    @classmethod
    def execute(cls, model, text, reference_audio, speed=1.0) -> IO.NodeOutput:
        LOGGER.info("%s started: text='%s'", LOG_PREFIX, text[:64])
        # ... actual synthesis ...
        result_audio = {"waveform": waveform, "sample_rate": 24000}
        return IO.NodeOutput(result_audio)
```

Регистрация в `__init__.py` для V3-нод:

```python
from comfy_api.v0_0_2 import ComfyExtension

class _PackExtension(ComfyExtension):
    async def get_node_list(self):
        return [
            TS_CosyVoice3_Example,
            # остальные V3 ноды
        ]

async def comfy_entrypoint() -> ComfyExtension:
    return _PackExtension()
```

### 5.2. Важные детали именно этого пака

- **Custom IO type `COSYVOICE_MODEL`** — словарь с уже загруженной моделью, который течёт от `TS_CosyVoice3_ModelLoader` ко всем остальным нодам синтеза. Формат словаря — публичный контракт; менять только синхронно во всех 7 нодах. Объявление типа лежит в `nodes/_v3_types.py` и импортируется как `from ._v3_types import CosyVoiceModel`.
- **`AUDIO` type** в ComfyUI — `{"waveform": Tensor [B, C, T], "sample_rate": int}`. CosyVoice работает с **24 kHz mono** — ноды должны ресемплить вход и приводить к mono перед передачей в модель. В V3 — `IO.Audio.Input(...)` / `IO.Audio.Output(...)`.
- `tooltip` — двуязычный, как уже принято в коде (`description` на английском, `tooltip` на русском или комбинированный). Каждый `IO.<Type>.Input(...)` принимает `tooltip="..."` напрямую.
- Module-level state, а не class-level (V3 классы залочены).
- Slider-виджеты задаются через `display_mode=IO.NumberDisplay.slider`.
- Optional input → `optional=True` в его `IO.<Type>.Input(...)`. Хидден инпуты — через поле `hidden=[IO.Hidden.prompt, ...]` в `IO.Schema(...)` (в текущем паке не используются).

### 5.3. Legacy V1 — только для исторической справки

До релиза 0.7.0 пак использовал V1 schema (`INPUT_TYPES` + `RETURN_TYPES` + `FUNCTION` + `NODE_CLASS_MAPPINGS`). Если нужно посмотреть, как нода выглядела до миграции, — `git log --follow nodes/<имя>.py`. **Новый V1-код в этом паке писать запрещено**; все 7 нод уже на V3.

---

## 6. Naming Convention

| Артефакт                  | Пример                                              |
| ------------------------- | --------------------------------------------------- |
| Файл Python (нода)        | `ts_cosyvoice_<feature>_node.py`                    |
| Класс Python              | `TS_CosyVoice3_<Name>`                              |
| `IO.Schema(node_id=...)`  | `"TS_CosyVoice3_<Name>"` (ровно как имя класса)     |
| `display_name`            | `"TS CosyVoice <Name>"` (без `3` в публичном имени) |
| `category`                | `"TS CosyVoice3/<Subcategory>"` (с пробелом, слэшем) |
| Logger name               | `"TS CosyVoice3"` или `"TS CosyVoice3 <Node>"`      |
| Log prefix                | `"[TS CosyVoice3 <NodeName>]"`                      |
| Custom IO type            | `"COSYVOICE_MODEL"` (объявлен в `nodes/_v3_types.py`) |

Существующие исключения (унаследованы из upstream `Instruct2` / `SpeakerInstruct2` и сохраняются для совместимости):

- `TS_CosyVoice3_Instruct2` — display name `TS CosyVoice Text to Voice`.
- `TS_CosyVoice3_SpeakerInstruct2` — display name `TS CosyVoice Speaker Text To Voice`.
- `TS_CosyVoice3_VoiceConversion` — display name `TS CosyVoice Voice To Voice`.
- `TS_CosyVoice3_CrossLingual` — display name `TS CosyVoice Cross-Language`.

Не «исправлять» эти имена в порядке cleanup — это публичный контракт.

---

## 7. One Node = One File

> Одна публичная ComfyUI-нода = один `.py` файл в `nodes/`. Имя файла — `ts_cosyvoice_<feature>_node.py`.

Запрещено:

```text
nodes/ts_cosyvoice_example/schema.py    # не дробим одну ноду на много файлов
nodes/ts_cosyvoice_example/execute.py
```

```text
nodes/ts_cosyvoice_combined.py          # не объединяем несколько нод в один файл
class TS_CosyVoice3_NodeA: ...
class TS_CosyVoice3_NodeB: ...
```

Подкатегории внутри `nodes/` сейчас не используются — layout плоский. Не плоди `nodes/synthesis/`, `nodes/utilities/` без явной договорённости с пользователем.

Shared-логика разрешена в `utils/`. Существующий пример: `utils/ts_logging.py`. По мере роста проекта в `utils/` уместны:

- `ts_audio_prep.py` — ресемплинг, mono-down-mix, нормализация для подачи в CosyVoice.
- `ts_model_manager.py` — download/cache моделей через `folder_paths.models_dir`.
- `ts_speaker_io.py` — сохранение/загрузка speaker preset.

Правила для `utils/`:

- Никаких `INPUT_TYPES`/`RETURN_TYPES` — это не ноды.
- Чисто функции и небольшие классы. Никаких `__init__.py`-сайд-эффектов.
- Не импортируй `torch`/`librosa`/`transformers` на module level, если модуль может быть импортирован без них (см. §14).

Frontend (`js/`) в этом паке отсутствует. Если когда-нибудь добавим UI-расширения — будем писать отдельный регламент в этом же файле.

---

## 8. Audio & GPU Rules

ComfyUI-конвенции (актуальные для этого пака):

```text
AUDIO  -> {"waveform": Tensor [B, C, T], "sample_rate": int}
STRING -> str
INT/FLOAT -> int / float
COSYVOICE_MODEL -> dict (внутренний контракт пака)
```

CosyVoice 3 ожидает **24 kHz mono**. Любой входной `AUDIO`:

- проверяй `waveform.ndim == 3` (`[B, C, T]`);
- если `C > 1` — приводи к mono (`mean` по channel-оси) или делай документированный выбор канала;
- если `sample_rate != 24000` — ресемпли через `torchaudio.functional.resample` (или эквивалент); не молчи об этом, логируй;
- всегда сохраняй batch dim (`B`) — даже если на практике `B == 1`;
- референс для модели обрезай по длительности (например, до 30 c — как уже делает `TS_CosyVoice3_SaveSpeaker` с пометкой «max 30 seconds»);
- для длинного `source_audio` в voice conversion используй чанкинг с попыткой не резать речь посреди слова.

Tensor-правила:

- Не мутируй входные тензоры. `clone()` перед любой in-place операцией.
- Сохраняй dtype/device при всех преобразованиях, кроме явного `.to(device)` для inference.
- Не предполагай CUDA. Используй `comfy.model_management.get_torch_device()` (или флаг `device` из загрузчика).
- Не хардкодь `cuda:0`.
- Не загружай большие модели на module level — только лениво в загрузчике / при первом вызове.
- Используй `torch.no_grad()` для всего inference; `torch.inference_mode()` — если уверен, что inputs не понадобятся в autograd.
- Для fp16 — только если `device.type == "cuda"` и пользователь явно включил (через input ноды или флаг загрузчика).

Пример:

```python
import comfy.model_management as mm
import torch
import torchaudio.functional as F


def prep_reference(audio: dict, target_sr: int = 24000) -> torch.Tensor:
    waveform = audio["waveform"]
    if waveform.ndim != 3:
        raise ValueError("[TS CosyVoice3] AUDIO waveform must be [B, C, T]")
    if waveform.shape[1] > 1:
        waveform = waveform.mean(dim=1, keepdim=True)
    sr = int(audio["sample_rate"])
    if sr != target_sr:
        waveform = F.resample(waveform, orig_freq=sr, new_freq=target_sr)
    return waveform  # [B, 1, T] @ target_sr


device = mm.get_torch_device()
with torch.no_grad():
    out = model.inference(reference.to(device), text)
```

---

## 9. Validation, Caching, Side Effects

V1-эквиваленты:

- `VALIDATE_INPUTS(cls, ...)` — ранняя проверка путей, имён preset, валидности языкового кода. Возвращай `True` или строку с понятной ошибкой (с TS-префиксом).
- `IS_CHANGED(cls, ...)` — детерминированный хеш для cache invalidation. Никогда не возвращай константу, если результат зависит от внешнего файла. Для speaker preset считай `mtime + size + name`. Для текста — сам текст.
- `OUTPUT_NODE = True` — только для нод с побочным эффектом (`TS_CosyVoice3_SaveSpeaker` пишет в `models/cosyvoice/speaker/`).

Side-effects-чеклист для нод:

- Запись в `models/cosyvoice/...` — только через `folder_paths.models_dir` + helper; никаких абсолютных путей.
- Скачивание моделей — только в `TS_CosyVoice3_ModelLoader`. Остальные ноды не должны качать ничего сами.
- Whisper-транскрипция (используется в `TS_CosyVoice3_SaveSpeaker` для авто-`reference_text`) — допустимо, но падать gracefully, если whisper отсутствует или транскрипция упала; reference_text при этом остаётся пустым.

---

## 10. Mandatory Verification

Минимум перед заявлением «готово»:

```bash
python -m compileall .
python -m pytest -v
```

Используй ComfyUI-овский Python (см. §4.6.2). Под обычным test Python тесты, требующие `torch`/`librosa`/`onnxruntime`, скипаются — это **не** настоящая проверка.

Если доступны:

```bash
python -m ruff check .
python -m ruff format --check .
```

> ⚠️ **Важно:** не запускай `ruff` / `black` / любые автоформаттеры по `cosyvoice/` и `matcha/` — это vendored код, форматирование увеличит diff против upstream. При вызове linter ограничивай scope: `ruff check nodes utils __init__.py`.

Smoke-импорт пакета (быстрая проверка, что `__init__.py` не сломан) в migration window:

```bash
# V1 ноды (legacy mappings):
python -c "import sys; sys.path.insert(0, '.'); import comfyui_ts_cosyvoice as p; print(list(p.NODE_CLASS_MAPPINGS))"

# V3 ноды (через ComfyExtension):
python -c "import asyncio, sys; sys.path.insert(0, '.'); import comfyui_ts_cosyvoice as p; ext = asyncio.run(p.comfy_entrypoint()); nodes = asyncio.run(ext.get_node_list()); print([n.define_schema().node_id for n in nodes])"
```

(или из родительского каталога ComfyUI: `python -c "from custom_nodes.comfyui_ts_cosyvoice import NODE_CLASS_MAPPINGS; print(list(NODE_CLASS_MAPPINGS))"`).

Для реального прогона нод нужен запущенный ComfyUI с уже скачанной `Fun-CosyVoice3-0.5B`. Это явно фиксируем в verification summary.

Verification summary в ответе пользователю — обязателен и пишется на русском:

```text
Проверено:
- python -m compileall .
- python -m pytest -v

Не проверено:
- Реальный inference TS_CosyVoice3_Instruct2 — модель Fun-CosyVoice3-0.5B локально не скачана.
- Запуск ComfyUI на 127.0.0.1:8188 — недоступен в текущей среде.
```

Никогда не пиши «готово» только потому, что код выглядит правильным.

---

## 11. Tests

Тесты должны быть CPU-safe, без интернета, без скачивания моделей.

- Контрактный тест каждой ноды — проверка наличия `INPUT_TYPES`, `RETURN_TYPES`, `RETURN_NAMES`, `FUNCTION`, `CATEGORY`, и того, что `FUNCTION`-метод имеет правильную сигнатуру.
- Behaviour-тесты для helpers в `utils/` (audio prep, speaker preset I/O) — без `torch` где возможно, либо через `pytest.importorskip("torch")`.
- Real-inference тесты помечай `@pytest.mark.gpu` и/или `@pytest.mark.requires_model`, чтобы CI / portable Python мог их пропустить, когда модели нет.
- Для тестов нод, требующих CosyVoice, мокируй `cosyvoice.cli.cosyvoice.CosyVoice3` через `monkeypatch` — это даёт smoke-проверку без 1.5 ГБ весов.
- Snapshot контрактов нод (если потребуется) — в `tests/contracts/node_contracts.json` плюс генератор в `tools/`. Сейчас этого нет — добавлять только при явной необходимости.

---

## 12. Frontend

Frontend в этом паке **отсутствует**. `WEB_DIRECTORY` не объявляется, `js/` не существует.

Если когда-нибудь понадобится UI (например, кнопка прослушивания preset, in-node плеер, drag-drop reference audio):

- Добавь регламент по DOM widgets / E2E в этот файл отдельной секцией.
- Не изобретай новый layout — следуй существующим практикам ComfyUI (`app.registerExtension`, `addDOMWidget`).

Без явного запроса пользователя `js/` не создаём.

---

## 13. Python Standards

Используй:

- Python 3.10+.
- Type hints для публичных функций и helpers.
- Google-style docstrings для нетривиальных функций.
- `pathlib.Path` для путей.
- `logging.getLogger(...)` через `utils.ts_logging.get_logger(...)` — префиксы `[TS CosyVoice3 <Node>]`, plain text, без ANSI/emoji/секретов.
- Lazy imports для тяжёлых зависимостей: `import torch` / `import librosa` / `import whisper` — внутри функций или метода ноды, **не** на module level.
- Specific exception types, понятные сообщения с TS-префиксом.

Запрещено:

- `print()` для логирования (только настоящий `logging`).
- Голый `except:` или `except Exception:` без полезной обработки.
- `global` и скрытое мутируемое module state (кроме явного кэша модели в загрузчике).
- Side effects на module-level (загрузка моделей, открытие файлов, скачивание).
- Авто-установка пакетов из кода (`pip install ...`).
- `eval`, `exec`, `subprocess` с пользовательским вводом, `os.system`.
- `pickle.load` без проверки источника / без `weights_only=True` где применимо.
- Хардкод абсолютных путей.

---

## 14. Dependency Resilience

Зависимости пака (`pyproject.toml` / `requirements.txt`) тяжёлые: `torch`, `torchaudio`, `transformers`, `librosa`, `onnxruntime`, `huggingface_hub`, `modelscope`, `openai-whisper`, `diffusers`, `WeTextProcessing`, `pyworld`, `HyperPyYAML`, `hydra-core` и т.д.

Политика:

1. Все объявленные в `pyproject.toml` зависимости считаются **обязательными** при реальном использовании пака. Пользователь устанавливает `requirements.txt` перед стартом.
2. На module level в `nodes/` импортируй только то, что не валит загрузку без модели: `import logging`, `from utils.ts_logging import get_logger`, базовые `os` / `pathlib`. Тяжёлые импорты (`torch`, `torchaudio`, `librosa`, `whisper`, `cosyvoice.cli.cosyvoice`) — **внутри** `INPUT_TYPES`/`FUNCTION`-метода или через ленивую функцию `_lazy_import_*`.
3. Если зависимость отсутствует — кидай `RuntimeError` с TS-префиксом и понятным сообщением: что не установлено, какая команда чинит.
4. Ноды не должны падать на этапе импорта пакета `__init__.py` — если упало здесь, ComfyUI не зарегистрирует ни одну ноду пака.
5. Авто-установка через `pip` из кода ноды запрещена.

Пример lazy-import:

```python
def _load_cosyvoice():
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice3
    except ImportError as exc:
        raise RuntimeError(
            "[TS CosyVoice3 ModelLoader] CosyVoice runtime is not available. "
            "Install requirements.txt and restart ComfyUI."
        ) from exc
    return CosyVoice3
```

---

## 15. Logging

```python
from utils.ts_logging import get_logger

LOGGER = get_logger("TS CosyVoice3 Example")
LOG_PREFIX = "[TS CosyVoice3 Example]"

LOGGER.info("%s started: text='%s'", LOG_PREFIX, text[:64])
LOGGER.warning("%s falling back to CPU", LOG_PREFIX)
```

- DEBUG — внутреннее, INFO — операции, WARNING — recoverable, ERROR — failure.
- Plain text, без ANSI, без emoji, без секретов.
- Не утекать полные приватные пути пользователя — обрезай до `~/.../speaker/<name>.pt` или показывай только имя файла.
- Не логируй сырой текст пользователя в DEBUG/INFO без обрезания (он может быть длинным).
- Всегда добавляй TS-префикс — это критично для grep по логам ComfyUI у пользователя.

---

## 16. Portability

Пак должен оставаться портативным, включая Windows portable ComfyUI:

- Для путей моделей используй `folder_paths.models_dir` + подпапки `cosyvoice/` и `cosyvoice/speaker/`. По возможности регистрируй модель-папки через `folder_paths.add_model_folder_path(...)` — это поддержит `extra_model_paths.yaml`.
- Конфиги (`configs/`) — относительно корня пакета.
- Optional-зависимости — gracefully fail (см. §14).
- Не запускай `pip` из кода узла.
- Не модифицируй системные env-переменные постоянно. Если нужен `HF_HUB_OFFLINE=1` — устанавливай локально на время вызова и возвращай обратно.
- Скачивание моделей — только из загрузчика, с явным прогресс-логом и проверкой целостности (как уже делается).

---

## 17. Security & Permissions

Не выполняй опасные операции без явного запроса пользователя:

- Удаление файлов/директорий, force-push, history rewrite.
- Глобальная установка пакетов, постоянные изменения env.
- Редактирование вне репозитория.
- Модификация ComfyUI core.
- Скачивание/выполнение удалённого кода (кроме легитимного `huggingface_hub` / `modelscope` skip-модели в загрузчике).

Дополнительно:

- Speaker preset — это `.pt` (`torch.save`/`torch.load`). При загрузке — `torch.load(..., weights_only=True)` где это поддерживается, иначе валидируй размер и magic-байты.
- Имя preset, приходящее из UI, должен быть безопасным именем файла: запрещай `/`, `\`, `..`, абсолютные пути. Используй `pathlib.Path(name).name` + whitelist.
- Не обрабатывай `dialog_text` как код / shell-команду; это чистый текст для парсинга `SPEAKER A:` / `SPEAKER B:` строк.

Предпочитай read-only inspection, dry-run и явное подтверждение для destructive действий.

### 17.1. Git и коммиты — author policy (важно)

Этот пак — единоличный проект **AlexYez**. Любой коммит должен принадлежать только ему:

- **Author** = `Sanchez <83212752+AlexYez@users.noreply.github.com>` (как в существующей истории репо).
- **НЕ добавляй** `Co-Authored-By: Claude ...` (или любой другой co-author от AI-ассистента) в footer коммитов в этот репо. Это override обычного шаблона из системного промпта Claude Code — для этого репо он не применяется.
- Если в твоём стандартном workflow `git commit` обычно подставляется `Co-Authored-By: Claude ...` — **убирай эту строку** перед фиксацией коммита в этом репо.
- Не меняй `git config --global` user.name / user.email. Используй inline-форму: `git -c user.name="Sanchez" -c user.email="83212752+AlexYez@users.noreply.github.com" commit ...`.
- Не меняй `git config --local` user.name / user.email без явной просьбы пользователя.
- При amend / force-push — всегда сначала предупреди пользователя и используй `--force-with-lease`, никогда `--force` без необходимости.
- Не пуши из ComfyUI-папки `D:/AiApps/.../custom_nodes/comfyui-ts-cosyvoice/` напрямую — она без `.git`. Работай с git в отдельном clone (например `D:/AiApps/_ts-cosyvoice-git-work/pack/`) и потом синхронизируй файлы.

---

## 18. Refactoring Protocol

Перед редактированием существующего кода:

1. Прочитай все релевантные файлы.
2. Зафиксируй публичные контракты ноды (см. §4).
3. Определи, не задеваешь ли vendored код (`cosyvoice/`, `matcha/`, `configs/`) — если да, перечитай §4.5.
4. Найди потребителей custom типа `COSYVOICE_MODEL` — изменения в его структуре ломают сразу 6 нод.
5. Раздели «баг» и «refactor opportunity».
6. Выбери минимальное безопасное изменение.

Safe:

- Извлечь private helpers в тот же файл или в `utils/`.
- Перенести только реально shared-логику (используется 2+ нодами) в `utils/`.
- Добавить type hints / docstrings в **наш** код (не в vendored).
- Заменить `print()` на logging.
- Улучшить validation и tensor-операции.
- Починить resource leaks (закрыть файлы, освободить модель).

Запрещено в рамках «чистого refactor»:

- Дробить ноду на много файлов.
- Переписывать стабильный код ради чистоты.
- Менять поведение ноды.
- Добавлять фичи, которых не просили.
- Удалять закомментированный код без подтверждения.
- Менять идентификаторы (class, mappings keys, `RETURN_TYPES`/`outputs`, FUNCTION name, `node_id`).
- Трогать `cosyvoice/`, `matcha/` (vendored upstream).
- «Причёсывать» формат словаря `COSYVOICE_MODEL`.

Миграция V1 → V3 — отдельный жанр работы, не часть «чистого refactor». В migration window она разрешена и идёт по [MIGRATION_PLAN.md](MIGRATION_PLAN.md): по одной ноде за коммит, с сохранением `node_id` и порядка outputs.

Формат ответа для рефакторинга — на русском, четырьмя секциями:

```markdown
### Анализ
### Риски
### Изменение
### Проверка
```

---

## 19. Self-Review Gate (перед финалом)

- [ ] Ответ на русском?
- [ ] Все возможные тесты запущены (`python -m pytest -v`, `python -m compileall .`)?
- [ ] Контрактный тест `tests/test_node_contracts.py` зелёный (7/7)?
- [ ] Незапущенные проверки явно перечислены с причинами?
- [ ] Импорты идут через `from comfy_api.v0_0_2 import IO, ComfyExtension` — **не** `comfy_api.latest`?
- [ ] `IO.Schema(node_id=...)` совпадает с тем, что записан в `tests/contracts/v1_baseline.json`?
- [ ] `display_name`, `category`, `is_output_node` совпадают с baseline?
- [ ] `inputs=[...]` сохраняют **имена**, **типы**, **порядок** и `optional=True` в нужных местах?
- [ ] `outputs=[...]` сохраняют типы и порядок из baseline?
- [ ] `search_aliases=[...]` задан (пустой список — нормально, если переименований не было)?
- [ ] Класс не использует `cls.X = ...` для кэшей (V3 классы залочены)?
- [ ] Custom тип `COSYVOICE_MODEL` импортируется из `nodes/_v3_types.py`, а не объявляется заново?
- [ ] `execute(cls, ...)` — `@classmethod`, возвращает `IO.NodeOutput(...)`?
- [ ] Формат словаря `COSYVOICE_MODEL` совместим со всеми 6 потребителями?
- [ ] Формат speaker preset совместим со старыми сохранёнными `.pt`?
- [ ] One-node-one-file сохранён (один `ts_cosyvoice_<feature>_node.py` = один публичный класс)?
- [ ] AUDIO batch dim сохранён, входной тензор не мутируется?
- [ ] Sample-rate приводится к 24000 явно?
- [ ] Нет heavy import (`torch`, `librosa`, `cosyvoice...`) на module level в нодах (loader-у `torch`/`comfy.model_management` разрешено как часть его контракта)?
- [ ] Нет авто-`pip install` / hardcoded абсолютных путей / unsafe shell?
- [ ] Vendored код (`cosyvoice/`, `matcha/`) не задет (или задет минимально с явным `# TS-PATCH:` и обоснованием)?
- [ ] Logging plain text, без секретов и без длинных raw-фрагментов пользовательского текста?
- [ ] Optional/heavy dependencies fail gracefully?
- [ ] `__init__.py` (`_V3_NODES` + `_PackExtension.get_node_list()`) обновлён при добавлении/удалении ноды?

---

## 20. Response Format

Для новых фич, отвечай на русском по структуре:

1. Что сделано.
2. Какие файлы изменены.
3. Дизайн-решения.
4. Предположения.
5. Что проверено.
6. Что не проверено и почему.
7. Риски.

Для рефакторинга — `### Анализ / ### Риски / ### Изменение / ### Проверка`.

Verification summary всегда в формате:

```text
Проверено:
- python -m compileall .
- python -m pytest -v

Не проверено:
- Реальный TTS на Fun-CosyVoice3-0.5B — модель локально не скачана.
- E2E в ComfyUI — сервер не запущен.
```

---

## 21. Definition of Done

Задача завершена только когда:

- Ответ пользователю на русском.
- Запрошенное изменение реализовано.
- Workflow contracts сохранены: `node_id`, `display_name`, `category`, имена/типы/порядок входов и выходов совпадают с `tests/contracts/v1_baseline.json`. Тест `tests/test_node_contracts.py` зелёный.
- Vendored код не задет (или задет минимально, с `# TS-PATCH:` комментарием и обоснованием).
- Все доступные релевантные проверки запущены.
- Незапущенные проверки перечислены с причинами.
- Код импортируется без ошибок (`python -m compileall .`).
- `__init__.py` (`_V3_NODES` + `_PackExtension.get_node_list()`) обновлён при добавлении/удалении ноды.
- Используется `comfy_api.v0_0_2`, а не `comfy_api.latest`.
- Logging plain text.
- Optional/heavy dependencies fail gracefully.
- Никаких unsafe API/секретов.
- README обновлён, если задеты публичные имена нод или их inputs/outputs.
- One-node-one-file сохранён (или shared-utility в `utils/` обоснован).

---

## 22. Final Priority Order

1. Preserve existing workflows.
2. Preserve compatibility (включая формат `COSYVOICE_MODEL` и speaker presets).
3. Stay close to upstream CosyVoice / Matcha.
4. Preserve user-facing behavior.
5. Improve correctness.
6. Improve testability.
7. Improve maintainability.
8. Improve performance.
9. Improve architecture.
10. Add features only when requested.

---

## 23. Полезные ссылки внутри репо

- [README.md](README.md) — пользовательская документация (русский + английский).
- [pyproject.toml](pyproject.toml) — версия, dependencies, ComfyRegistry metadata.
- [requirements.txt](requirements.txt) — runtime dependencies.
- [`__init__.py`](__init__.py) — реестр нод (правится вручную при добавлении новой).
- [`utils/ts_logging.py`](utils/ts_logging.py) — фабрика логгеров с TS-префиксом.
- `cosyvoice/`, `matcha/`, `configs/` — vendored upstream (не трогать без явного запроса).
- `workflows/` — пример .json workflows, полезны как ground truth публичных контрактов нод.
