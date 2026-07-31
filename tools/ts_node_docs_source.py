"""
Single source of truth for the pack's user-facing prose.

Everything structural — input ids, types, defaults, ranges, combo options, the
Russian tooltips — is read straight from ``define_schema()`` by
``tools/generate_node_docs.py``. This module holds only what the code cannot
provide:

- the English tooltip for every input (the in-code tooltip is Russian, and stays
  Russian: it is the fallback the pack's primary audience already sees),
- a short English/Russian description for every output,
- the long-form prose for the embedded help pages.

Nothing here is imported at runtime. The generator turns it into
``web/docs/<node_id>/{en,ru}.md`` and ``locales/{en,ru}/nodeDefs.json``, and
``tests/test_generated_docs.py`` fails if the committed artefacts drift from it.

Keys must match ``IO.Schema.node_id`` and input ids exactly; the generator raises
on any mismatch, so a renamed input cannot silently lose its documentation.
"""

from __future__ import annotations

# Reused verbatim across nodes so the wording cannot drift between them.
_MODEL_EN = "The loaded CosyVoice model, from the TS CosyVoice Model Loader node."
_SEED_EN = (
    "Random seed. Set it to -1 for a fresh take on every run: the node then "
    "re-runs instead of returning its cached audio. Any value from 0 up is "
    "deterministic — re-running with everything unchanged returns the previous "
    "result, so change the seed (or use control_after_generate: randomize) when "
    "you want a different take from a fixed seed."
)
_TEXT_NORMALIZE_EN = (
    "Run CosyVoice's text frontend (number and abbreviation expansion). Turn it "
    "off when you feed CMU phonemes or special tags that must survive verbatim."
)
_SPEED_EN = "Playback speed multiplier applied to the generated speech."
_EMOTION_PRESET_EN = (
    "Ready-made delivery preset, read from configs/ts_emotion_presets.json. "
    "Pick the custom entry to use the instruct_text field instead."
)
_INSTRUCT_TEXT_EN = (
    "Free-form instruction describing emotion and delivery. Only used when "
    "emotion_preset is set to the custom entry."
)
_REFERENCE_AUDIO_EN = (
    "Voice to clone. Trimmed to 30 seconds and converted to mono 24 kHz; "
    "3-10 seconds of clean speech gives the best timbre."
)

_REFERENCE_AUDIO_NOTE_EN = (
    "Reference audio is trimmed to the first 30 seconds and downmixed to mono "
    "24 kHz before it reaches the model. Clean speech without music or "
    "background noise clones far better than a long noisy sample."
)
_REFERENCE_AUDIO_NOTE_RU = (
    "Референс обрезается до первых 30 секунд и приводится к mono 24 кГц перед "
    "подачей в модель. Чистая речь без музыки и шума клонируется заметно лучше, "
    "чем длинный шумный фрагмент."
)
_SEED_NOTE_EN = (
    "Two ways to get a different take. seed = -1 re-runs the node on every "
    "execution, so each run is a new take. With a fixed seed the result is "
    "reproducible, and ComfyUI returns the cached audio until something changes "
    "— set control_after_generate to randomize if you want the seed itself to "
    "move between runs."
)
_SEED_NOTE_RU = (
    "Получить другой дубль можно двумя способами. Значение seed = -1 заставляет "
    "ноду выполняться заново при каждом запуске, поэтому каждый прогон даёт "
    "новый дубль. При фиксированном seed результат воспроизводим, и ComfyUI "
    "отдаёт аудио из кэша, пока ничего не изменилось — поставьте "
    "control_after_generate: randomize, если хотите, чтобы сам seed менялся "
    "между запусками."
)

NODE_DOCS: dict[str, dict] = {
    "TS_CosyVoice3_ModelLoader": {
        "description_ru": (
            "Загружает модель CosyVoice с автоматическим скачиванием и кэшированием."
        ),
        "intro_en": (
            "Downloads (once) and loads a CosyVoice model, then hands it to every "
            "other node in the pack through the COSYVOICE_MODEL output. Put one "
            "loader in the graph and fan its output out to all synthesis nodes — "
            "the model is cached in memory, so a second loader with the same "
            "settings costs nothing extra."
        ),
        "intro_ru": (
            "Один раз скачивает и загружает модель CosyVoice, а затем отдаёт её "
            "всем остальным нодам пака через выход COSYVOICE_MODEL. Держите в "
            "графе один загрузчик и разводите его выход по всем нодам синтеза — "
            "модель кэшируется в памяти, поэтому второй загрузчик с теми же "
            "настройками ничего дополнительно не стоит."
        ),
        "inputs": {
            "model_version": "Which CosyVoice model to load.",
            "download_source": (
                "Where to download the weights from on first use. If the chosen "
                "source fails, the other one is tried automatically."
            ),
            "device": (
                "Preferred compute device, best effort. The CosyVoice runtime "
                "places the model itself and uses the GPU when one is available, "
                "so this request may not be honoured literally."
            ),
            "fp16": (
                "Load in half precision on supported accelerators to cut VRAM "
                "use. Ignored without CUDA."
            ),
        },
        "outputs": {
            "model": {
                "en": "The loaded model, to be wired into the pack's synthesis nodes.",
                "ru": "Загруженная модель — подключается ко всем нодам синтеза пака.",
            },
        },
        "notes_en": [
            "Weights land in `ComfyUI/models/cosyvoice/<model_version>/` "
            "(about 1.5 GB). The first run downloads them; later runs reuse them.",
            "A downloaded model is checked for truncated or corrupted files. The "
            "result of that check is recorded next to the weights, so the "
            "expensive part runs once rather than on every ComfyUI start.",
            "The device selector is a request, not a guarantee — read the node's "
            "log line to see where the model actually ended up.",
        ],
        "notes_ru": [
            "Веса кладутся в `ComfyUI/models/cosyvoice/<model_version>/` "
            "(около 1.5 ГБ). Первый запуск скачивает их, последующие переиспользуют.",
            "Скачанная модель проверяется на обрезанные и повреждённые файлы. "
            "Результат проверки записывается рядом с весами, поэтому дорогая "
            "часть выполняется один раз, а не при каждом старте ComfyUI.",
            "Выбор устройства — это пожелание, а не гарантия: фактическое "
            "размещение модели видно в строке лога этой ноды.",
        ],
    },
    "TS_CosyVoice3_Instruct2": {
        "description_ru": (
            "Синтез речи с тембром из референса и эмоцией, заданной инструкцией."
        ),
        "intro_en": (
            "Zero-shot voice cloning: give it a reference recording and the text "
            "to read, and it speaks that text in the reference voice. The "
            "delivery — warm, dramatic, tired — is steered by an instruction, "
            "either a ready-made preset or your own sentence."
        ),
        "intro_ru": (
            "Zero-shot клонирование голоса: даёте референсную запись и текст — "
            "нода произносит этот текст голосом из референса. Манера подачи "
            "(тёплая, драматичная, усталая) задаётся инструкцией: готовым "
            "пресетом или вашей собственной фразой."
        ),
        "inputs": {
            "model": _MODEL_EN,
            "text": "The text to speak in the reference voice.",
            "instruct_text": _INSTRUCT_TEXT_EN,
            "reference_audio": _REFERENCE_AUDIO_EN,
            "speed": _SPEED_EN,
            "seed": _SEED_EN,
            "text_normalize": _TEXT_NORMALIZE_EN,
            "emotion_preset": _EMOTION_PRESET_EN,
        },
        "outputs": {
            "audio": {
                "en": "The generated speech, at the model's own sample rate (24 kHz).",
                "ru": "Сгенерированная речь с частотой дискретизации модели (24 кГц).",
            },
        },
        "notes_en": [
            _REFERENCE_AUDIO_NOTE_EN,
            "emotion_preset wins over instruct_text. instruct_text is read only "
            "when the preset is set to the custom entry.",
            "Presets are read from `configs/ts_emotion_presets.json` on every "
            "run, so you can edit that file and use the change without "
            "restarting ComfyUI. New preset *names* appear in the dropdown after "
            "a node-definition refresh.",
            _SEED_NOTE_EN,
        ],
        "notes_ru": [
            _REFERENCE_AUDIO_NOTE_RU,
            "emotion_preset важнее instruct_text: поле instruct_text читается "
            "только когда в пресете выбран пункт пользовательской инструкции.",
            "Пресеты читаются из `configs/ts_emotion_presets.json` при каждом "
            "запуске, так что файл можно править и пользоваться изменениями без "
            "перезапуска ComfyUI. Новые *названия* пресетов появляются в списке "
            "после обновления определений нод.",
            _SEED_NOTE_RU,
        ],
    },
    "TS_CosyVoice3_SpeakerInstruct2": {
        "description_ru": (
            "Синтез речи по сохранённому пресету голоса (.pt) и инструкции эмоции."
        ),
        "intro_en": (
            "The same instruction-guided synthesis as TS CosyVoice Text to Voice, "
            "but the voice comes from a preset you saved earlier with TS CosyVoice "
            "Save Speaker instead of from a reference clip on the canvas. Use it "
            "when you reuse one voice across many generations."
        ),
        "intro_ru": (
            "Тот же синтез с инструкцией, что и в TS CosyVoice Text to Voice, но "
            "голос берётся из пресета, сохранённого нодой TS CosyVoice Save "
            "Speaker, а не из референсного клипа на канвасе. Удобно, когда один "
            "голос переиспользуется во многих генерациях."
        ),
        "inputs": {
            "model": _MODEL_EN,
            "text": "The text to speak in the saved voice.",
            "instruct_text": _INSTRUCT_TEXT_EN,
            "speaker_preset": (
                "A voice preset saved earlier by TS CosyVoice Save Speaker, read "
                "from ComfyUI/models/cosyvoice/speaker/."
            ),
            "speed": _SPEED_EN,
            "seed": _SEED_EN,
            "text_normalize": _TEXT_NORMALIZE_EN,
            "emotion_preset": _EMOTION_PRESET_EN,
        },
        "outputs": {
            "audio": {
                "en": "The generated speech, at the model's own sample rate (24 kHz).",
                "ru": "Сгенерированная речь с частотой дискретизации модели (24 кГц).",
            },
        },
        "notes_en": [
            "The dropdown lists the `.pt` files in "
            "`ComfyUI/models/cosyvoice/speaker/`. With none saved yet it shows a "
            "single placeholder entry and the node asks you to run Save Speaker "
            "first.",
            "A preset re-saved under the same name is picked up: the node keys "
            "its cache on the file's timestamp, so it re-runs instead of "
            "returning the previous voice.",
            "A workflow shared with someone who does not have that preset file "
            "will fail validation on this node — ship the `.pt`, or have them "
            "recreate it with Save Speaker.",
            _SEED_NOTE_EN,
        ],
        "notes_ru": [
            "В списке — файлы `.pt` из `ComfyUI/models/cosyvoice/speaker/`. Пока "
            "не сохранён ни один пресет, там будет единственный placeholder, а "
            "нода попросит сначала выполнить Save Speaker.",
            "Пересохранённый под тем же именем пресет подхватывается: нода "
            "учитывает время изменения файла в ключе кэша и перезапускается, "
            "а не отдаёт прежний голос.",
            "У получателя workflow без этого файла пресета нода не пройдёт "
            "валидацию — передавайте `.pt` вместе с графом или попросите "
            "пересоздать его через Save Speaker.",
            _SEED_NOTE_RU,
        ],
    },
    "TS_CosyVoice3_CrossLingual": {
        "description_ru": (
            "Кросс-язычный синтез: тембр из референса сохраняется, язык меняется."
        ),
        "intro_en": (
            "Speaks text in a different language while keeping the timbre of the "
            "reference voice. The reference may be in one language and the text "
            "in another — the voice carries over, the accent follows the target "
            "language."
        ),
        "intro_ru": (
            "Произносит текст на другом языке, сохраняя тембр референсного "
            "голоса. Референс может быть на одном языке, а текст — на другом: "
            "голос переносится, а произношение следует целевому языку."
        ),
        "inputs": {
            "model": _MODEL_EN,
            "text": "The text to speak, written in the target language.",
            "reference_audio": _REFERENCE_AUDIO_EN,
            "speed": _SPEED_EN,
            "target_language": (
                "Target language of the text. auto lets the model infer it, "
                "which is usually right for unambiguous text."
            ),
            "seed": _SEED_EN,
            "text_normalize": _TEXT_NORMALIZE_EN,
        },
        "outputs": {
            "audio": {
                "en": "The generated speech, at the model's own sample rate (24 kHz).",
                "ru": "Сгенерированная речь с частотой дискретизации модели (24 кГц).",
            },
        },
        "notes_en": [
            _REFERENCE_AUDIO_NOTE_EN,
            "Write the text in the target language itself — this node does not "
            "translate. Pair it with a translation node if you need that.",
            "On CosyVoice3 the language tag is not sent explicitly; the model "
            "infers the language from the text, so target_language mostly "
            "matters for older CosyVoice checkpoints.",
            _SEED_NOTE_EN,
        ],
        "notes_ru": [
            _REFERENCE_AUDIO_NOTE_RU,
            "Текст пишите сразу на целевом языке — эта нода не переводит. Если "
            "нужен перевод, поставьте перед ней ноду-переводчик.",
            "На CosyVoice3 языковой тег явно не передаётся: модель определяет "
            "язык по самому тексту, поэтому target_language важен в основном для "
            "более старых чекпоинтов CosyVoice.",
            _SEED_NOTE_RU,
        ],
    },
    "TS_CosyVoice3_VoiceConversion": {
        "description_ru": (
            "Преобразует исходное аудио в тембр целевого голоса (voice-to-voice)."
        ),
        "intro_en": (
            "Re-voices an existing recording: keeps the timing, phrasing and "
            "delivery of the source performance, and replaces its timbre with "
            "the target voice. Nothing is re-read from text, so the original "
            "acting is preserved."
        ),
        "intro_ru": (
            "Перепевает существующую запись: сохраняет тайминг, фразировку и "
            "подачу исходного исполнения, заменяя тембр на целевой голос. Текст "
            "заново не читается, поэтому актёрская игра оригинала сохраняется."
        ),
        "inputs": {
            "model": _MODEL_EN,
            "source_audio": (
                "The performance to re-voice. Timing and delivery are kept; only "
                "the timbre changes. Long audio is processed in chunks."
            ),
            "target_audio": (
                "The voice to sound like. Trimmed to 30 seconds and converted to "
                "the model's format."
            ),
            "speed": _SPEED_EN,
            "pitch_shift_semitones": (
                "Shift the source pitch before conversion. Useful when source "
                "and target sit in different registers — try -12 or +12 when "
                "converting between a low and a high voice."
            ),
            "seed": _SEED_EN,
        },
        "outputs": {
            "audio": {
                "en": "The re-voiced audio, at the model's own sample rate (24 kHz).",
                "ru": "Переозвученное аудио с частотой дискретизации модели (24 кГц).",
            },
        },
        "notes_en": [
            "Source audio longer than 24 seconds is split into chunks. The split "
            "prefers a pause so it does not land mid-word, and the converted "
            "chunks are rejoined with a short crossfade.",
            "This is the slowest node in the pack. On CPU a ten-minute file can "
            "take hours — run it on a GPU.",
            "Cancel stops it between chunks, so a long job aborts within roughly "
            "one chunk rather than running to the end.",
            "If silence detection is unavailable in your environment, the node "
            "logs a warning and falls back to fixed-size chunks. Conversion still "
            "works; a split may land mid-word.",
        ],
        "notes_ru": [
            "Исходное аудио длиннее 24 секунд режется на чанки. Точка реза по "
            "возможности выбирается в паузе, чтобы не попасть в середину слова, "
            "а преобразованные чанки склеиваются коротким кроссфейдом.",
            "Это самая медленная нода пака. На CPU десятиминутный файл может "
            "считаться часами — запускайте на GPU.",
            "Cancel останавливает её между чанками, поэтому долгая задача "
            "прерывается примерно в пределах одного чанка, а не досчитывается до "
            "конца.",
            "Если детектор тишины в вашем окружении недоступен, нода пишет "
            "предупреждение и переходит на чанки фиксированного размера. "
            "Конвертация продолжает работать, но рез может попасть в середину слова.",
        ],
    },
    "TS_CosyVoice3_SaveSpeaker": {
        "description_ru": (
            "Извлекает признаки голоса из референса и сохраняет их в пресет .pt."
        ),
        "intro_en": (
            "Extracts the speaker features from a reference recording and saves "
            "them as a reusable `.pt` preset, so you do not have to keep the "
            "reference audio in every graph. Presets are picked up by TS "
            "CosyVoice Speaker Text To Voice."
        ),
        "intro_ru": (
            "Извлекает признаки голоса из референсной записи и сохраняет их как "
            "переиспользуемый пресет `.pt`, чтобы не таскать референсное аудио в "
            "каждый граф. Пресеты подхватывает нода TS CosyVoice Speaker Text To "
            "Voice."
        ),
        "inputs": {
            "model": _MODEL_EN,
            "reference_audio": _REFERENCE_AUDIO_EN,
            "reference_text": (
                "What is actually said in the reference audio. Leave it empty to "
                "have Whisper transcribe it automatically — an accurate "
                "transcript noticeably improves the clone."
            ),
            "speaker_name": (
                "File name for the preset, without extension. Must be a plain "
                "name: no slashes, no '..', none of < > : \" / \\ | ? *."
            ),
        },
        "outputs": {
            "saved_path": {
                "en": "Absolute path the preset was written to.",
                "ru": "Абсолютный путь, по которому записан пресет.",
            },
        },
        "notes_en": [
            _REFERENCE_AUDIO_NOTE_EN,
            "Presets are written to `ComfyUI/models/cosyvoice/speaker/`. Saving "
            "under an existing name overwrites it.",
            "Leaving reference_text empty pulls in Whisper. The first time, that "
            "downloads the Whisper 'base' model (about 140 MB) into "
            "`ComfyUI/models/whisper/`.",
            "If Whisper is missing or fails, the preset is still saved but with "
            "an empty transcript, and a warning naming the reason is written to "
            "the ComfyUI console log. Such a preset clones noticeably worse, and "
            "the node gives no on-canvas sign of it — check the log after saving "
            "a preset without a reference_text, then fill the text in by hand and "
            "save again.",
        ],
        "notes_ru": [
            _REFERENCE_AUDIO_NOTE_RU,
            "Пресеты пишутся в `ComfyUI/models/cosyvoice/speaker/`. Сохранение "
            "под существующим именем перезаписывает файл; нода Speaker Text To "
            "Voice это заметит и перегенерирует речь.",
            "Пустое поле reference_text задействует Whisper. При первом "
            "обращении скачивается модель Whisper 'base' (около 140 МБ) в "
            "`ComfyUI/models/whisper/`.",
            "Если Whisper отсутствует или падает, пресет всё равно сохраняется, "
            "но с пустой расшифровкой, а причина попадает предупреждением в лог "
            "консоли ComfyUI. Такой пресет клонирует заметно хуже, и на канвасе "
            "нода этого никак не показывает — после сохранения пресета с пустым "
            "reference_text загляните в лог, впишите текст вручную и сохраните "
            "заново.",
        ],
    },
    "TS_CosyVoice3_Dialog": {
        "description_ru": (
            "Многоголосый синтез диалога с клонированием голосов (SPEAKER A: / B: ...)."
        ),
        "intro_en": (
            "Turns a script into a multi-voice conversation. Label each line with "
            "SPEAKER A:, SPEAKER B: and so on, give each speaker a reference "
            "clip, and the node renders the whole exchange — plus one isolated "
            "track per speaker for mixing."
        ),
        "intro_ru": (
            "Превращает сценарий в многоголосый диалог. Размечайте реплики как "
            "SPEAKER A:, SPEAKER B: и так далее, дайте каждому говорящему "
            "референсный клип — нода соберёт весь диалог и, дополнительно, по "
            "отдельной дорожке на каждого говорящего для сведения."
        ),
        "inputs": {
            "model": _MODEL_EN,
            "dialog_text": (
                "The script, one line per turn, each prefixed with SPEAKER A:, "
                "SPEAKER B:, SPEAKER C: or SPEAKER D:. Lines without a "
                "recognised prefix, and lines for a speaker with no reference "
                "audio, are skipped."
            ),
            "speaker_A_Audio": (
                "Reference voice for SPEAKER A. 3-10 seconds works best; 30 "
                "seconds is the hard limit."
            ),
            "speaker_B_Audio": (
                "Reference voice for SPEAKER B. 3-10 seconds works best; 30 "
                "seconds is the hard limit."
            ),
            "speed": "Playback speed multiplier applied to every line of the dialog.",
            "speaker_C_Audio": "Optional reference voice for SPEAKER C.",
            "speaker_D_Audio": "Optional reference voice for SPEAKER D.",
            "seed": _SEED_EN,
        },
        "outputs": {
            "dialog_audio": {
                "en": "The full conversation, lines concatenated in script order.",
                "ru": "Весь диалог: реплики склеены в порядке сценария.",
            },
            "speaker_a_audio": {
                "en": "SPEAKER A alone, silent where the others speak — same "
                      "length as dialog_audio, so the tracks line up.",
                "ru": "Только SPEAKER A, тишина на чужих репликах — длина "
                      "совпадает с dialog_audio, дорожки синхронны.",
            },
            "speaker_b_audio": {
                "en": "SPEAKER B alone, silent where the others speak.",
                "ru": "Только SPEAKER B, тишина на чужих репликах.",
            },
            "speaker_c_audio": {
                "en": "SPEAKER C alone; all silence when C never speaks.",
                "ru": "Только SPEAKER C; полная тишина, если C не говорит.",
            },
            "speaker_d_audio": {
                "en": "SPEAKER D alone; all silence when D never speaks.",
                "ru": "Только SPEAKER D; полная тишина, если D не говорит.",
            },
            "message": {
                "en": "Short report: duration, number of lines rendered, and any "
                      "lines that were skipped.",
                "ru": "Короткий отчёт: длительность, сколько реплик озвучено и "
                      "какие пропущены.",
            },
        },
        "notes_en": [
            "The prefix match is case-insensitive, so `Speaker A:` works as well "
            "as `SPEAKER A:`.",
            "Every per-speaker track is the full length of the dialog, so you can "
            "drop all four straight onto a timeline without aligning them.",
            "Reference clips are transcribed with Whisper once per run and the "
            "result is reused for every line by that speaker. The first use "
            "downloads the Whisper 'base' model (about 140 MB).",
            "Cancel stops the node between lines rather than at the end of the "
            "whole dialog.",
            "Check the message output: it names the lines that were skipped for "
            "want of a matching speaker input.",
        ],
        "notes_ru": [
            "Префикс сопоставляется без учёта регистра, так что `Speaker A:` "
            "работает так же, как `SPEAKER A:`.",
            "Каждая дорожка говорящего имеет полную длину диалога, поэтому все "
            "четыре можно класть на таймлайн без дополнительного выравнивания.",
            "Референсные клипы расшифровываются Whisper один раз за запуск, и "
            "результат переиспользуется для всех реплик этого говорящего. При "
            "первом обращении скачивается модель Whisper 'base' (около 140 МБ).",
            "Cancel останавливает ноду между репликами, а не по завершении всего "
            "диалога.",
            "Смотрите выход message: там перечислены реплики, пропущенные "
            "из-за отсутствия соответствующего входа говорящего.",
        ],
    },
}


__all__ = ["NODE_DOCS"]
