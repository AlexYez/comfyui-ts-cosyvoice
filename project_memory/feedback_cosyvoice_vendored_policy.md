---
name: comfyui-ts-cosyvoice — vendored cleanup policy
description: User decision 2026-05-13 — не трогать vendored CosyVoice/Matcha ради профилактики Comfy scanner, пока пак Active
type: feedback
originSessionId: bd354924-8658-494e-9c85-eb3b37e6fe93
---
Пользователь сознательно отказался от профилактической чистки vendored (`cosyvoice/bin/`, `cosyvoice/utils/executor.py`, `cosyvoice/dataset/`, `# TS-PATCH:` правка `os.system` в `cosyvoice/utils/file_utils.py:121`), даже когда был предложен вариант с минимальным риском (удаление training-only файлов, которые ни одна нода не импортирует).

**Why:** На момент решения 2026-05-13 все опубликованные версии (0.6.0, 0.7.0, 0.8.0) имели статус `NodeVersionStatusActive` в Comfy registry. Реактивная позиция: чинить только когда появится реальный `NodeVersionStatusFlagged`. Это согласуется с CLAUDE.md §4.5 («По умолчанию — не править» vendored) и принципом «Primum non nocere».

**How to apply:**
- Не предлагай профилактическую правку vendored сама по себе («давай уберём bin/ для безопасности», «давай заменим os.system на subprocess для bandit»). Это уже разобрано и отклонено.
- Поднимай вопрос только если: (а) появилась реальная `Flagged` версия в `https://api.comfy.org/nodes/comfyui-ts-cosyvoice/versions`, ИЛИ (б) пользователь сам спросил.
- Перед любой правкой vendored всё равно требуется явное подтверждение и `# TS-PATCH:` комментарий.
- Аудит-карта что и где можно удалить — в memory `project_cosyvoice_registry_audit.md`.
