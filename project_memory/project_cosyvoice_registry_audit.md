---
name: comfyui-ts-cosyvoice — Comfy registry security scanner audit
description: Snapshot 2026-05-13 of bandit-trigger audit for the pack and where vendored CosyVoice has dormant triggers
type: project
originSessionId: bd354924-8658-494e-9c85-eb3b37e6fe93
---
Audit performed 2026-05-13 after publishing version 0.8.0 (with new `.comfyignore`).

**Registry state at audit time:** все три опубликованные версии — `NodeVersionStatusActive`:
- 0.6.0 (2026-04-13) — Active
- 0.7.0 (2026-05-07) — Active
- 0.8.0 (2026-05-13) — Active (`latest_version`)

API endpoint для повторной проверки: `https://api.comfy.org/nodes/comfyui-ts-cosyvoice` и `/versions`.

**Scanner-relevant findings (информационно, не баги):**

Наш код (nodes/, utils/, tests/, __init__.py) — **0 bandit-триггеров**.

Vendored CosyVoice — реальные триггеры есть, но scanner их сейчас принимает:
- `cosyvoice/utils/file_utils.py:121` — `os.system('sed -i ... {}/config.json'.format(...))` внутри `export_cosyvoice2_vllm()`. Вызывается только при `load_vllm=True`; наши ноды этого не делают (ни одна нода не передаёт load_vllm/load_trt/load_jit).
- `cosyvoice/bin/average_model.py:55` — `yaml.load(BaseLoader)`. BaseLoader безопасен (не выполняет код), но bandit B506 формально ругается. Файл — training-only, runtime пути не задеты.

**Training-only код, не используется в runtime ни одной ноды** (потенциально удаляемое если scanner станет строже):
- `cosyvoice/bin/` (4 файла: average_model.py, export_jit.py, export_onnx.py, train.py)
- `cosyvoice/utils/executor.py`
- `cosyvoice/utils/train_utils.py`
- `cosyvoice/dataset/` (2 файла)
- Функция `export_cosyvoice2_vllm` в `cosyvoice/utils/file_utils.py:98-124` (но сам файл целиком удалить нельзя — runtime импортирует оттуда `load_wav`/`read_lists`/`logging`)

**False positives (PyTorch `.eval()` метод, не Python builtin `eval(`):**
`cosyvoice/cli/model.py:66,68,72`, `hifigan/generator.py:738`, `flow/flow.py:415`, `utils/executor.py:50,151`, `bin/export_onnx.py:65` — все безопасны.

**Why:** pack уже Active в registry, scanner пропустил всё перечисленное; срочной правки vendored нет. Записано как карта для случая, когда scanner станет строже или появится Flagged-версия.

**How to apply:** при будущем re-audit-е сначала проверь `https://api.comfy.org/nodes/comfyui-ts-cosyvoice/versions` — если статусы изменились, открой эту памятку и решай что чистить. Если статусы не изменились — vendored не трогать (см. CLAUDE.md §4.5).

**.comfyignore note:** scanner анализирует ВЕСЬ git checkout, а не только zip. `.comfyignore` уменьшает только размер публикуемого архива и убирает визуальный шум, но НЕ защищает от scanner. Подтверждено мейнтейнером Comfy-Org в Discord.
