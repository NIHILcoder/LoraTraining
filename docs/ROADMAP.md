# LoRA Studio — Roadmap (функциональные пробелы + фиксы кода)

Документ составлен по результатам аудита всего кода (Electron + FastAPI + trainer + React).
Каждый пункт проверен по фактическому исходнику (адверсариальная верификация), ссылки — на реальные `file:line`.

Легенда:
- Severity: `core` = table-stakes, без этого продукт неполноценен · `important` = ждут серьёзные пользователи · `nice` = приятно иметь.
- Effort: `S` (<1 дня) · `M` (1–3 дня) · `L` (неделя+) · `XL` (крупная фича).
- 🐞 = фикс существующего бага · ✨ = новая функциональность.
- Ссылки `P0-xx/P1-xx` — соответствие пунктам `technical-debt-plan.md`.

---

## Статус реализации (сессия 2026-07-05)

Легенда: ✅ сделано и проверено (type-check + py_compile + адверсариальное ревью) · 🔶 код готов, нужна проверка на GPU · ⏸ отложено (нужен запуск/UI/железо/сеть).

- **M0 — ✅ полностью.** SD2.1 закрыт для обучения + v-pred target, auto-caption без потери подписей, поля Pydantic (`enableBucketing/captionDropout/noiseOffset`), убран двойной scale LoRA, подключён `_inference_lock`, реальные размеры изображений, escape SVG + коды ошибок BLIP, относительный URL галереи + приём `?token=` в middleware.
- **M1 — 🔶 частично.** Сэмплинг без повторов + per-bucket очереди (M1-4), LR-schedule по optimizer-steps (M1-5), сиды `random`/`torch` (M1-12), v-prediction target (M1-8 частично), `time_ids` под переменный батч. Требуют прогона на GPU. ⏸ Осталось: sample-превью, чекпоинты/resume, on-the-fly латенты, train_text_encoder, min-SNR, EMA/validation/TensorBoard.
- **M2 — 🔶 частично.** ✅ Per-image delete (M2-6); bulk caption tools — prepend/trigger, append, find-replace (M2-3); серверный batch-caption endpoint + одно сохранение (M2-1); валидационные подсказки датасета. ⏸ Осталось: авто-trigger в конфиге, captioner’ы (WD14/Florence), кроп-редактор.
- **M4 — ✅ / ⏸.** ✅ Resumable/segmented download (Range + докачка; `.part` сохраняется при сетевой ошибке) (M4-1), SHA256-механизм (M4-2), предпроверка диска (M4-3). ⏸ Осталось: импорт локального файла (нужен file-picker IPC + UI), CivitAI.
- **M5 — 🔶 / ⏸.** ✅ Пины зависимостей + лог версий + запрет pickle-расширений (M5-2), ограничение роста lossHistory/logs (M5-4), сохраняемые пресеты конфига (localStorage, M5-5 часть), **тесты (vitest + reducer)** (M5-1 часть), **автообновление через electron-updater + GitHub Releases** (M5-7). ⏸ Осталось: lint/CI, устойчивость (single-instance/логи/рестарт backend), кроссплатформенность. Экран Settings — понижен: настройки (models/output dir, HF-токен) уже имеют UI в ModelsPage/GalleryPage. ⚠️ **M5-3 НЕ выполнен намеренно** — в `ConfigPage/DatasetPage/TrainingPage` лежат незакоммиченные правки; удаление затрёт их (решение за автором). ⚠️ Авто-обновление заработает только после публикации релиза с установщиком + `latest.yml` через `npm run electron:publish` (см. README/ниже).
- **M3 — 🔶 частично.** ✅ URL вместо base64 + PNG-info метаданные (M3-5), batch-генерация N картинок с разными сидами (M3-2, клиентский цикл), показ ошибок генерации пользователю (было — только в консоль). ⏸ Осталось (нужен GPU/крупнее): live-прогресс + отмена, img2img, hires-fix, multi-LoRA, инференс Flux/SD3.

---

## Что уже работает (не переписывать)

Реальный training loop SD1.5/SDXL на diffusers+peft; aspect-ratio bucketing; кеш латентов и эмбеддингов; VRAM-профили и ETA; cosine/constant + warmup; AdamW/AdamW8bit/SGD; noise offset, caption dropout, clip-skip, hflip; SDXL dual text-encoder + time_ids; WebSocket-прогресс; кооперативный stop; каталог/загрузка/кастомные модели; BLIP-captioning; датасеты CRUD; Playground t2i с инъекцией LoRA; галерея. Security-boundary Electron (contextIsolation + preload allowlist + per-session token + динамический порт) — сделано и на уровне выше среднего.

---

## M0 — Корректность и потеря данных (BLOCKER, делать первым)

Эти баги дают **молча испорченный результат** или **теряют пользовательские данные**. Фичи поверх них бессмысленны.

| # | 🐞 | Severity | Файл | Проблема → действие |
|---|----|----------|------|---------------------|
| M0-1 | 🐞 | core | `backend/trainer.py:585`, `:837` | **SD 2.1 обучается сломанным.** Грузится ViT-L вместо OpenCLIP ViT-H, а loss всегда epsilon-MSE (SD2.1 — v-prediction). Модель тихо деградирует. → Либо **запретить sd21** в `start_training` до реализации, либо ветка sd21: `stable-diffusion-2-1` text_encoder + `target = get_velocity(...)` при `prediction_type=='v_prediction'`. |
| M0-2 | 🐞 | core | `src/components/workspace/DatasetSection.tsx:95` | **Auto-caption-all теряет подписи.** Stale-closure по `state.currentDataset` в цикле → каждая итерация перезатирает предыдущие. → Аккумулировать локально и один PUT, либо серверный batch-endpoint (см. M2-1). |
| M0-3 | 🐞 | important | `backend/main.py:201` | **Advanced-настройки не доходят до тренера.** Pydantic `TrainingConfig` не объявляет `enableBucketing/captionDropout/noiseOffset` → Pydantic их выбрасывает → `config.get(...)` в тренере всегда берёт дефолт. Пользовательские тумблеры — no-op. → Добавить поля в модель. |
| M0-4 | 🐞 | important | `backend/main.py:819` + `:865` | **Вес LoRA применяется дважды** (set_adapters + cross_attention scale) → нелинейный эффект слайдера. → Оставить один механизм (убрать `cross_attention_kwargs`). |
| M0-5 | 🐞 | important | `backend/main.py:737` / `:501` | **Гонка на кеше инференса.** `_inference_lock` создан, но не используется; `_inference_cache.clear()` может снести pipe, который читает параллельный запрос. → `async with _inference_lock:` вокруг load+inference. |
| M0-6 | 🐞 | important | `backend/main.py:341` | **width/height захардкожены в 1024** при загрузке картинки → неверные метаданные/бакеты. → Читать `PIL.Image.open(...).size`. |
| M0-7 | 🐞 | important | `backend/main.py:888`, `:922`/`:980` | **Ошибки маскируются:** непроэкранированный prompt ломает mock-SVG (инъекция); BLIP-фейлы возвращают строку-подпись (`"error"`) вместо ошибки. → `xml.escape`; возвращать 4xx/5xx. |
| M0-8 | 🐞 | minor | `backend/main.py:1640` | Захардкоженный `http://localhost:8000` в URL картинок галереи ломается при динамическом порте. → Отдавать относительный `/api/generated/...`. |
| M0-9 | 🐞 | minor | `backend/main.py:626` | Реальные фейлы инференса «тонут» в mock-успехе (совпадает с **P1-07**). → Явный `simulationMode`, иначе HTTP-ошибка. |

**Быстрые победы (S) из M0:** M0-3, M0-4, M0-6, M0-7, M0-8. Один день — заметно чище поведение.

---

## M1 — Обучение до профессионального уровня

Без этого нельзя судить о сходимости и получать конкурентное качество LoRA.

| # | тип | Severity | Effort | Пункт |
|---|-----|----------|--------|-------|
| M1-1 | ✨ | core | L | **Sample-превью во время/после обучения** (`sample_every_n_steps`, промпты) — главный инструмент оценки сходимости. Сейчас нет вообще. |
| M1-2 | ✨ | core | M | **Промежуточные чекпоинты** (save-every-N / по эпохам) + понятие эпох. Сейчас сохраняется только финал. |
| M1-3 | ✨ | important | L | **Resume-from-checkpoint.** |
| M1-4 | 🐞→✨ | important | M | **DataLoader вместо `random.choices` с повторами** (`trainer.py:807`): epoch-shuffle без повторов, per-bucket sampler. Сейчас на малом датасете одна картинка попадает в батч несколько раз → переобучение. |
| M1-5 | 🐞 | important | M | **grad-accum: рассинхрон шагов/шедулера** (`trainer.py:849`) — определить, `trainingSteps` это optimizer- или micro-steps, и привести цикл/шедулер/прогресс к одному. |
| M1-6 | 🐞→✨ | important | L | **train_text_encoder реально включить** (`trainer.py:193` — флаг вычисляется, но нигде не применяется) либо убрать из профиля. |
| M1-7 | ✨ | important | M | **on-the-fly латенты** (`cache_latents=False` путь) — сейчас его нет, большой датасет упал бы на `IndexError`. |
| M1-8 | ✨ | important | M | **min-SNR gamma / v-prediction / zero-terminal-SNR** — стандартные множители качества. |
| M1-9 | ✨ | important | L | **Regularization / class images** (prior preservation). |
| M1-10 | ✨ | nice | M | Validation split + val-loss; EMA; TensorBoard/CSV-лог; **kohya-совместимый формат ключей** (чтобы LoRA грузилась в A1111/ComfyUI). |
| M1-11 | ✨ | nice | S | Довести UI до реальных возможностей: экспонировать AdamW8bit/SGD (сейчас в UI только AdamW), добавить linear/poly/cosine_restarts. |
| M1-12 | 🐞 | minor | S | Сид не покрывает crop/hflip RNG (`trainer.py:782`) → нерепродуцируемо; VRAM-guard на 1536 для SD1.5 (`ConfigSection.tsx:205`). |

---

## M2 — Датасет и подготовка данных

| # | тип | Severity | Effort | Пункт |
|---|-----|----------|--------|-------|
| M2-1 | ✨ | important | M | **Серверный batch-caption endpoint** (сейчас клиентский цикл, см. M0-2). |
| M2-2 | ✨ | core | M | **Trigger word / activation token** — управление и авто-подстановка в подписи. Для персонажных/стилевых LoRA — обязательное. |
| M2-3 | ✨ | important | M | **Шаблоны подписей** (prepend/append) + **find-and-replace** по всему датасету. |
| M2-4 | ✨ | important | L | **Лучшие captioner’ы**: WD14-tagger (аниме/booru), BLIP2/Florence-2. BLIP-base слаб. |
| M2-5 | ✨ | important | L | Кроп/поворот/редактирование картинок до обучения; drag-drop импорт папки; import/export датасета. |
| M2-6 | ✨/🐞 | important | M | **Валидация датасета** (кол-во, разрешение, формат — сейчас только пустой-датасет). Пер-image delete endpoint (`api.ts:172` переписывает весь датасет → потеря правок); caption-sync fire-and-forget (`DatasetSection.tsx:52`). |
| M2-7 | ✨ | nice | M | Частоты тегов + автокомплит; превью бакетов; dedup; per-image enable/repeats. |

---

## M3 — Playground / инференс до полезного уровня

| # | тип | Severity | Effort | Пункт |
|---|-----|----------|--------|-------|
| M3-1 | ✨ | important | L | **Live-прогресс шагов + отмена генерации** (сейчас только спиннер, отмены нет) — с сериализацией через `_inference_lock` (M0-5). |
| M3-2 | ✨ | important | M | **Batch (N картинок) + очередь генераций.** |
| M3-3 | ✨ | important | M | **img2img**; hires-fix/апскейл. |
| M3-4 | ✨ | important | M | **Стек нескольких LoRA**; VAE override; больше шедулеров. |
| M3-5 | ✨ | important | S | **PNG-info / встроенные метаданные** (совместимость с A1111/Civitai). Отдавать URL вместо base64 data-URL (payload 2–5 МБ). |
| M3-6 | ✨ | important | M | **Prompt weighting / длинные промпты** (compel). |
| M3-7 | ✨ | important | L | **Инференс Flux/SD3/Cascade** — сейчас в UI выбираются, но backend бросает «not implemented». Реализовать или явно скрыть (**P1-01/P1-07**). |
| M3-8 | ✨ | nice | L/XL | Inpainting; ControlNet; XY-plot/prompt-matrix. |

---

## M4 — Модели и загрузки

| # | тип | Severity | Effort | Пункт |
|---|-----|----------|--------|-------|
| M4-1 | ✨ | core | L | **Resumable/segmented downloads** — сейчас при обрыве качается заново (файлы по 4–24 ГБ). |
| M4-2 | ✨ | important | M | **SHA256-верификация** (поле `sha256` в типе есть, но не используется). |
| M4-3 | ✨ | important | S | **Предпроверка места на диске** до старта загрузки. |
| M4-4 | ✨ | important | M | **Импорт локального файла модели** через file-picker (не только по URL). |
| M4-5 | ✨ | nice | M | CivitAI-интеграция + API-ключ; управление VAE; retry/pause очереди. |

---

## M5 — Инфраструктура и продакшн-готовность

| # | тип | Severity | Effort | Пункт |
|---|-----|----------|--------|-------|
| M5-1 | ✨ | core | M | **Тесты + lint + CI.** Сейчас скриптов `test/lint` нет вообще (**P2-01**) — самый большой риск при таком числе хрупких стыков. Начать с pytest на backend (settings/paths/error-codes) и reducer/api-хелперов. |
| M5-2 | 🐞 | core | M | **Запинить версии `torch/transformers/diffusers/peft/accelerate`, централизовать monkey-patch, убрать bypass `check_torch_load_is_safe`** (**P1-02**). Сейчас незапиненные deps + отключённая pickle-защита + произвольные URL моделей = RCE-риск. |
| M5-3 | 🐞 | important | S | **Мёртвый код:** `ConfigPage/DatasetPage/TrainingPage` (~1200 строк) нигде не роутятся (App.tsx монтирует только Workspace/Models/Playground/Gallery). Удалить или консолидировать (**P2-02**). |
| M5-4 | 🐞 | important | S | **Безлимитный рост** `lossHistory` и `logs` (`AppContext.tsx:122/134`) — тормозит UI на долгом обучении. → Кольцевой буфер (последние N). |
| M5-5 | ✨ | important | M | **Экран Settings** (models/output dir, HF-токен, simulation mode) + **сохраняемые пользовательские пресеты** (сейчас пресеты хардкод apply-only, персиста нет). |
| M5-6 | ✨ | important | M | **Устойчивость рантайма:** health-check + авто-рестарт backend при падении; file-logging (electron-log); crashReporter; single-instance lock; graceful shutdown. Ничего из этого нет. |
| M5-7 | ✨ | nice | L | Auto-update (electron-updater) + code signing. |
| M5-8 | ✨ | nice | L | Кросс-платформенность (сейчас Windows-only: `uv-*-windows`, `python.exe`, `taskkill`, `Expand-Archive`). |

---

## Рекомендуемая последовательность

1. **M0** целиком (корректность/данные) — 3–5 дней, много S-фиксов.
2. **M5-1 + M5-2** (тесты + безопасность deps) — чтобы дальше не ломать вслепую.
3. **M1-1, M1-2, M1-4, M1-8** (sample-превью, чекпоинты, DataLoader, min-SNR) — это то, что отличает «работает» от «делает хорошие LoRA».
4. **M2-1..M2-3** (batch-caption, trigger word, шаблоны) — реальный workflow подготовки.
5. **M3-1, M3-2, M3-5, M3-7** (прогресс+отмена, batch, PNG-info, честный Flux/SD3).
6. **M4-1..M4-3** (resumable/checksum/disk-check).
7. Остальное по мере приоритетов.

## Definition of Done для «1.0 (не beta)»
M0 полностью · M5-1/M5-2 · M1-1/M1-2/M1-4 · M3-7 (Flux/SD3 честно реализован или скрыт) · M5-3/M5-4 · один smoke-test запакованного приложения.
