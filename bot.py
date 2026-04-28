import os
import logging
import tempfile
import time
import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Message
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    CommandHandler,
    CallbackQueryHandler,
    filters,
)
from openai import OpenAI
from google import genai

# ── Логирование ───────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Настройки ─────────────────────────────────────────────────────────────────
TELEGRAM_MAX_DOWNLOAD_MB = 20
TELEGRAM_MAX_DOWNLOAD_BYTES = TELEGRAM_MAX_DOWNLOAD_MB * 1024 * 1024
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.0-flash")
GEMINI_GENERATE_RETRIES = int(os.getenv("GEMINI_GENERATE_RETRIES", "4"))

# ── Клиенты ───────────────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# ── Промпты ───────────────────────────────────────────────────────────────────
GEMINI_VIDEO_PROMPT = """Ты анализируешь рекламный видео-креатив для AI-production. Важно: опиши всё, что реально видно и слышно в видео. Не будь чрезмерно осторожным: если объект/язык/формат очевиден визуально или на слух — укажи его. Если параметр совсем невозможно определить — пиши «не определено». Без вводных фраз, без смайлов.

Проанализируй видео и выдай строго в этом формате:

АНАЛИЗ ВИДЕО:
Хронометраж: [общая длительность видео]
Структура:
- [0:00-0:XX] [что происходит в кадре]
- [продолжай по каждому смысловому блоку]

Визуал: [локация/фон, цветовая гамма, стиль, композиция, где находится стример/персонаж, как показан слот/игра, есть ли рамка/оверлей]
Темп монтажа: [медленный / средний / быстрый, примерная частота смены кадров]

Озвучка:
- Есть / Нет
- Пол: [мужской / женский / не определено]
- Возраст голоса: [молодой / средний / зрелый / не определено]
- Язык: [язык, если слышно или очевидно]
- Акцент: [есть / нет / не определено, если есть — какой]
- Эмоция: [возбуждение / удивление / спокойная подача / агрессия / другое]
- Тип: [живой голос / TTS / не определено]
- Основные фразы дословно: [если слышно; если речь неразборчива — кратко опиши смысл]

Субтитры:
- Есть / Нет
- Если есть:
  - Текст: [дословно, если читается]
  - Количество строк: [1 строка / 2 строки / динамически меняется]
  - Позиция: [низ / центр / верх / другое]
  - Стиль: [жирный / обычный / капс / смешанный]
  - Цвет: [основной цвет текста]
  - Обводка / тень: [есть / нет, описание]
  - Подложка: [есть / нет, описание]
  - Анимация: [статично / попап / подсветка / появление по словам / другое]
- Если нет: субтитры отсутствуют

Игровой процесс / слот:
- Есть / Нет
- Игра / слот: [название, если видно; если нет — не определено]
- Источник геймплея: [реальная запись / сгенерированная имитация / UI-мокап / готовая запись / не определено]
- Что происходит в игре: [спины, бонус, колесо, выигрыш, множители, фриспины, реакции и т.д.]

Музыка: [есть/нет, характер]
CTA: [есть/нет, текст, тайминг появления, оформление]
Формат: [соотношение сторон, например 9:16 / 1:1 / 16:9]
Технические особенности: [оверлеи, рамка стримера, эффекты, UI, store badges, логотипы, переходы, всё важное для сборки]
"""

GEMINI_VIDEO_FALLBACK_PROMPT = """Предыдущий анализ мог быть неполным. Проанализируй видео заново максимально практично для сборки рекламного креатива. Обязательно определи: длительность, язык/голос, есть ли субтитры, что видно в игровом процессе, есть ли стример/рамка/CTA/store badges, соотношение сторон. Не пиши «не определено», если это можно понять визуально или на слух. Ответь по структуре АНАЛИЗ ВИДЕО с теми же пунктами: Хронометраж, Структура, Визуал, Темп монтажа, Озвучка, Субтитры, Игровой процесс / слот, Музыка, CTA, Формат, Технические особенности.
"""

GPT_STRUCTURE_PROMPT = """Ты — менеджер по рекламным задачам и AI-креатор для performance creative production. Ты получаешь:
1. Анализ референс-видео от Gemini
2. Текст ТЗ от заказчика

Твоя задача — сопоставить их, структурировать ТЗ и составить план в работу.

СТРОГИЕ ПРАВИЛА:
- Никаких смайлов, emoji, звёздочек, украшений.
- Только факты — никаких воды и неподтвержденных утверждений.
- Тексты озвучки/субтитры/CTA — дословно как у заказчика или как видно/слышно в референсе.
- Если поле не указано — пишешь: не указано.
- Если анализ видео от Gemini содержит данные по видео, обязательно переноси их в блок «АНАЛИЗ РЕФЕРЕНСА». Не заменяй их на «не указано».
- Писать «Референс видео не удалось проанализировать» можно только если в анализе Gemini прямо есть ошибка обработки видео или пустой ответ.
- Никаких вводных фраз.
- В "НЕ УКАЗАНО / УТОЧНИТЬ" — только реально отсутствующие параметры, которые влияют на продакшн.

AI-FIRST ЛОГИКА ПЛАНИРОВАНИЯ:
- По умолчанию планируй задачу как AI-production: генерация персонажей/стримеров, готовые или сгенерированные ассеты, запись/генерация геймплея, AI voice/TTS, липсинк, монтаж, адаптация.
- Реальную съемку предлагай только если заказчик прямо указал: съемка, живой актер, реальный стример, UGC-съемка, офлайн-продакшн или запись реального человека.
- Если в задаче есть стример/блогер/актер/персонаж, но конкретный человек не указан: НЕ писать «организовать съемку», а писать «подобрать AI-стримера / подходящий типаж».
- Если бот не определил конкретного стримера: в плане нужно предложить 2–3 подходящих AI-типажа, согласовать с заказчиком, затем делать генерацию, озвучку и липсинк.
- Если в задаче есть игровой процесс: НЕ писать «организовать запись игрового процесса». Нужно определить, нужен ли реальный записанный геймплей, сборка слота с нуля, генерация/имитация слота или готовая запись.
- Если источник геймплея не указан: добавить в "НЕ УКАЗАНО / УТОЧНИТЬ" вопрос, нужно ли использовать реальную запись геймплея или можно собрать/сгенерировать слот.

ОБЯЗАТЕЛЬНЫЕ ПРОВЕРКИ:
- Субтитры: если в видео есть субтитры — описать их подробнее в анализе референса. Если субтитров нет и в ТЗ не указано, нужны ли они — добавить в "НЕ УКАЗАНО / УТОЧНИТЬ": "Нужны ли субтитры".
- Озвучка: указывать пол, язык, эмоцию, акцент, тип голоса. Если что-то не определено — так и писать.
- Валюта: если есть выигрыши/бонус/оффер без валюты — добавить валюту в уточнения.
- Слот/игра: если название слота не определено — добавить в уточнения.
- Стример: если нужен стример, но конкретный типаж не указан — добавить подбор AI-типажей в план, а при необходимости уточнить у заказчика.

ФОРМАТ ОТВЕТА:

──────────────────────────
GEO: [страна / регион]
LANGUAGE: [язык креативов]
CREATIVES: [число]
FORMAT: [статика / видео / карусель]
SIZE: [размеры]
CURRENCY: [валюта]
BONUS: [условия или «нет»]
──────────────────────────
КОММЕНТАРИЙ ЗАКАЗЧИКА:
[общие требования дословно, без текстов озвучки]
──────────────────────────
ДЕТАЛИ ЗАКАЗА:
Оффер: [продукт / оффер]
Площадка: [площадка]
Дополнительно: [логика крео, структура, особенности — кратко]
──────────────────────────
[только если есть тексты озвучки или субтитров]
ТЕКСТЫ / ОЗВУЧКА:
[дословно с нумерацией как у заказчика]
──────────────────────────
АНАЛИЗ РЕФЕРЕНСА:
Хронометраж: [из анализа видео]
Структура: [тайминги и смысловые блоки]
Визуал: [ключевые визуальные особенности]
Озвучка: [пол, язык, эмоция, акцент, тип, основные фразы]
Субтитры: [есть/нет; если есть — позиция, строки, стиль, цвет, тень/обводка, подложка, анимация]
Игровой процесс / слот: [название слота, если видно; источник геймплея; что происходит]
CTA: [текст, тайминг, оформление]
Формат: [соотношение сторон]
Технические особенности: [важное для сборки]
──────────────────────────
СОПОСТАВЛЕНИЕ:
[3-5 строк — что в видео совпадает с ТЗ, что расходится, на что обратить внимание]
──────────────────────────
ПЛАН В РАБОТУ:
1. Определить способ производства: AI-генерация / монтаж / готовые ассеты / съемка. По умолчанию использовать AI-production, если съемка не указана прямо.
2. Определить стримера / персонажа: если конкретный человек не указан, подобрать AI-типаж и предложить 2–3 варианта заказчику.
3. Утвердить выбранный типаж стримера / персонажа с заказчиком.
4. Определить источник игрового процесса: реальная запись геймплея / сборка слота с нуля / генерация или имитация слота / готовая запись.
5. После утверждения подготовить озвучку: язык, пол, эмоция, акцент, TTS или живой голос.
6. Сгенерировать / подготовить стримера и сделать липсинк, если лицо говорит в кадре.
7. Собрать сцену: стример / слот / UI / эффекты / CTA / store badges.
8. Добавить субтитры, если они нужны по ТЗ или по согласованию.
9. Смонтировать видео по структуре референса и адаптировать под нужный формат.
10. Подготовить финальные версии в нужном количестве с разными стримерами / типажами / вариациями.
──────────────────────────

Если не хватает технических параметров:
НЕ УКАЗАНО / УТОЧНИТЬ:
- [валюта, если не указана]
- [нужны ли субтитры, если не указано]
- [источник геймплея: реальная запись / сборка слота / генерация / готовая запись, если не указано]
- [название слота/игры, если не определено]
- [типаж стримера, если нужен, но не указан]
- [тип голоса: TTS или живой, если важно и не указано]
- [платформа размещения, если не указана]
"""

# ── Состояния ─────────────────────────────────────────────────────────────────
user_states: dict[int, str] = {}
user_text_buffers: dict[int, list[str]] = {}
user_media_buffers: dict[int, list[tuple[int, int]]] = {}
# file_id, mime_type, file_size
user_video_buffers: dict[int, list[tuple[str, str, int | None]]] = {}

CB_LOAD = "cb_load"
CB_STRUCTURE = "cb_structure"
CB_CLEAR = "cb_clear"
CB_STATUS = "cb_status"
CB_HELP = "cb_help"
CB_BACK = "cb_back"


def get_state(uid: int) -> str:
    return user_states.get(uid, "idle")


def set_state(uid: int, state: str) -> None:
    user_states[uid] = state


def clear_buffers(uid: int) -> None:
    user_text_buffers[uid] = []
    user_media_buffers[uid] = []
    user_video_buffers[uid] = []


def buf_summary(uid: int) -> str:
    t = len(user_text_buffers.get(uid, []))
    v = len(user_video_buffers.get(uid, []))
    m = len(user_media_buffers.get(uid, []))
    if t == 0 and v == 0 and m == 0:
        return "Буфер пустой"
    parts = []
    if t:
        parts.append(f"текст: {t} шт.")
    if v:
        parts.append(f"видео: {v} шт.")
    if m:
        parts.append(f"файлов/фото: {m} шт.")
    return "В буфере: " + ", ".join(parts)


# ── Клавиатуры ────────────────────────────────────────────────────────────────
def kb_main() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Загрузить ТЗ", callback_data=CB_LOAD)],
        [InlineKeyboardButton("Помощь", callback_data=CB_HELP)],
    ])


def kb_loading(uid: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Структурировать", callback_data=CB_STRUCTURE)],
        [
            InlineKeyboardButton("Статус буфера", callback_data=CB_STATUS),
            InlineKeyboardButton("Очистить буфер", callback_data=CB_CLEAR),
        ],
        [InlineKeyboardButton("Отмена", callback_data=CB_BACK)],
    ])


def kb_after_result() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Новое ТЗ", callback_data=CB_LOAD)],
    ])


def split_message(text: str, max_len: int = 4000) -> list[str]:
    if len(text) <= max_len:
        return [text]
    parts = []
    while text:
        parts.append(text[:max_len])
        text = text[max_len:]
    return parts


def count_label(n: int) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return f"{n} сообщение"
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return f"{n} сообщения"
    return f"{n} сообщений"


# ── Gemini: анализ видео ──────────────────────────────────────────────────────
def _gemini_file_state_name(uploaded_file) -> str:
    """Возвращает состояние Gemini File API в виде строки: PROCESSING / ACTIVE / FAILED."""
    state = getattr(uploaded_file, "state", None)
    if state is None:
        return ""
    return getattr(state, "name", str(state)).upper()


def _looks_like_weak_video_analysis(text: str) -> bool:
    """Проверяет, не вернул ли Gemini формально-пустой анализ видео."""
    if not text or len(text.strip()) < 250:
        return True
    low = text.lower()
    failure_markers = [
        "не удалось проанализировать",
        "пустой ответ",
        "i cannot",
        "i can't",
        "unable to",
    ]
    if any(m in low for m in failure_markers):
        return True
    unknown_count = low.count("не определено") + low.count("не указано")
    has_core = any(k in low for k in ["субтитры", "озвучка", "cta", "игровой процесс", "хронометраж"])
    return unknown_count >= 8 or not has_core




def _is_gemini_overloaded_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "503" in text
        or "unavailable" in text
        or "high demand" in text
        or "overloaded" in text
        or "temporarily" in text
    )


def _gemini_generate_content_with_retries(prompt: str, uploaded_file) -> str:
    """Генерирует анализ видео с ретраями и fallback-моделью при 503/high demand."""
    models_to_try = []
    for model_name in (GEMINI_MODEL, GEMINI_FALLBACK_MODEL):
        if model_name and model_name not in models_to_try:
            models_to_try.append(model_name)

    last_error: Exception | None = None

    for model_name in models_to_try:
        for attempt in range(1, GEMINI_GENERATE_RETRIES + 1):
            try:
                logger.info(
                    "Gemini video analysis request: model=%s attempt=%s/%s",
                    model_name, attempt, GEMINI_GENERATE_RETRIES,
                )
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=[prompt, uploaded_file],
                )
                return (getattr(response, "text", None) or "").strip()
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Gemini generate_content failed: model=%s attempt=%s/%s error=%s",
                    model_name, attempt, GEMINI_GENERATE_RETRIES, exc,
                    exc_info=True,
                )

                if not _is_gemini_overloaded_error(exc):
                    break

                delay = min(2 ** attempt, 20)
                time.sleep(delay)

    raise last_error or RuntimeError("Gemini вернул неизвестную ошибку генерации.")

def _analyze_video_gemini_sync(video_bytes: bytes, mime_type: str) -> str:
    tmp_path = None
    uploaded = None
    try:
        suffix = ".mp4"
        if "quicktime" in mime_type or "mov" in mime_type:
            suffix = ".mov"
        elif "webm" in mime_type:
            suffix = ".webm"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        uploaded = gemini_client.files.upload(file=tmp_path)

        # Видео в Gemini File API не всегда доступно сразу.
        # Ждём именно ACTIVE, а не просто окончание PROCESSING.
        wait_started = time.time()
        while True:
            state_name = _gemini_file_state_name(uploaded)

            if state_name == "ACTIVE":
                break

            if state_name == "FAILED":
                return "Видео не удалось проанализировать: Gemini не смог обработать файл."

            if time.time() - wait_started > 300:
                return "Видео не удалось проанализировать: Gemini обрабатывал файл дольше 5 минут."

            time.sleep(2)
            uploaded = gemini_client.files.get(name=uploaded.name)

        text = _gemini_generate_content_with_retries(GEMINI_VIDEO_PROMPT, uploaded)

        if _looks_like_weak_video_analysis(text):
            logger.warning("Gemini returned weak video analysis, retrying with fallback prompt")
            fallback_text = _gemini_generate_content_with_retries(GEMINI_VIDEO_FALLBACK_PROMPT, uploaded)
            if fallback_text and not _looks_like_weak_video_analysis(fallback_text):
                return fallback_text
            if fallback_text and len(fallback_text) > len(text):
                return fallback_text

        if text:
            return text

        return "Видео проанализировано, но Gemini вернул пустой ответ."
    except Exception as e:
        logger.error("Gemini video error: %s", e, exc_info=True)
        if _is_gemini_overloaded_error(e):
            return (
                "Видео не удалось проанализировать: Gemini временно недоступен из-за высокой нагрузки "
                "(503 UNAVAILABLE / high demand). Повтори структурирование позже или переключи GEMINI_MODEL/GEMINI_FALLBACK_MODEL."
            )
        return f"Видео не удалось проанализировать: {e}"
    finally:
        if uploaded is not None:
            try:
                gemini_client.files.delete(name=uploaded.name)
            except Exception as e:
                logger.warning("Не удалось удалить файл Gemini: %s", e)
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning("Не удалось удалить временный файл: %s", e)


async def analyze_video_gemini(video_bytes: bytes, mime_type: str) -> str:
    return await asyncio.to_thread(_analyze_video_gemini_sync, video_bytes, mime_type)


# ── GPT: структуризация ───────────────────────────────────────────────────────
def _structure_with_gpt_sync(tz_text: str, video_analysis: str | None) -> str:
    if video_analysis:
        user_content = (
            f"АНАЛИЗ ВИДЕО (от Gemini):\n{video_analysis}\n\n"
            f"ТЗ ОТ ЗАКАЗЧИКА:\n{tz_text}"
        )
    else:
        user_content = f"ТЗ ОТ ЗАКАЗЧИКА:\n{tz_text}"

    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": GPT_STRUCTURE_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content or "GPT вернул пустой ответ."


async def structure_with_gpt(tz_text: str, video_analysis: str | None) -> str:
    return await asyncio.to_thread(_structure_with_gpt_sync, tz_text, video_analysis)


# ── Хэндлеры ─────────────────────────────────────────────────────────────────
async def list_models(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        models = gemini_client.models.list()
        names = [m.name for m in models]
        text = "Доступные модели Gemini:\n" + "\n".join(names)
        for chunk in split_message(text):
            await update.message.reply_text(chunk)
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    set_state(uid, "idle")
    clear_buffers(uid)
    await update.message.reply_text(
        "ТЗ Бот готов к работе.\n\n"
        "Нажми «Загрузить ТЗ» — пересылай сообщения от заказчика в любом порядке "
        "(текст, видео-референсы, фото, файлы), затем нажми «Структурировать».",
        reply_markup=kb_main(),
    )


async def cb_load(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    uid = query.from_user.id
    set_state(uid, "loading")
    clear_buffers(uid)
    await query.message.reply_text(
        "Буфер открыт. Пересылай всё от заказчика — текст, видео, фото, файлы.\n"
        "Порядок не важен. Когда всё загружено — нажми «Структурировать».",
        reply_markup=kb_loading(uid),
    )


async def cb_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    uid = query.from_user.id
    await query.answer(buf_summary(uid), show_alert=True)


async def cb_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    uid = query.from_user.id
    clear_buffers(uid)
    await query.message.reply_text(
        "Буфер очищен. Можешь загружать заново.",
        reply_markup=kb_loading(uid),
    )


async def cb_back(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    uid = query.from_user.id
    set_state(uid, "idle")
    clear_buffers(uid)
    await query.message.reply_text(
        "Загрузка отменена.",
        reply_markup=kb_main(),
    )


async def cb_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    help_text = (
        "Как пользоваться ботом:\n\n"
        "1. Нажми «Загрузить ТЗ»\n"
        "2. Пересылай всё от заказчика — текст, видео, фото, файлы в любом порядке\n"
        "3. Нажми «Структурировать» — бот проанализирует всё и выдаст готовое ТЗ\n\n"
        "Кнопки во время загрузки:\n"
        "- Статус буфера — покажет сколько всего загружено\n"
        "- Очистить буфер — удалить всё и начать заново\n"
        "- Отмена — выйти без сохранения\n\n"
        "Если есть видео-референс — бот сначала проанализирует его через Gemini, "
        "потом сопоставит с текстом ТЗ и составит план в работу.\n\n"
        f"Видео: максимум {TELEGRAM_MAX_DOWNLOAD_MB} МБ для скачивания через обычный Telegram Bot API. "
        "Если файл больше — сожми видео или подключи локальный Telegram Bot API Server."
    )
    await query.message.reply_text(help_text, reply_markup=kb_main())


async def cb_structure(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    uid = query.from_user.id

    text_buf = user_text_buffers.get(uid, [])
    media_buf = user_media_buffers.get(uid, [])
    video_buf = user_video_buffers.get(uid, [])

    if not text_buf and not media_buf and not video_buf:
        await query.message.reply_text(
            "Буфер пустой — нечего структурировать.",
            reply_markup=kb_loading(uid),
        )
        return

    if not text_buf:
        await query.message.reply_text(
            "Нет текста. Добавь хотя бы одно текстовое сообщение с описанием задачи.",
            reply_markup=kb_loading(uid),
        )
        return

    set_state(uid, "processing")
    media_snap = list(media_buf)
    video_snap = list(video_buf)
    full_text = "\n\n---\n\n".join(text_buf)

    await context.bot.send_chat_action(chat_id=query.message.chat_id, action="typing")

    video_analysis = None
    if video_snap:
        count = len(video_snap)
        await query.message.reply_text(
            f"Анализирую видео через Gemini: {count} шт."
        )
        analyses = []
        for idx, (file_id, mime_type, file_size) in enumerate(video_snap, 1):
            try:
                if file_size and file_size > TELEGRAM_MAX_DOWNLOAD_BYTES:
                    analyses.append(
                        f"Видео {idx}: не удалось обработать — файл больше {TELEGRAM_MAX_DOWNLOAD_MB} МБ. "
                        "Сожми видео или подключи локальный Telegram Bot API Server."
                    )
                    continue

                await context.bot.send_chat_action(chat_id=query.message.chat_id, action="typing")
                tg_file = await context.bot.get_file(file_id)
                video_bytes = await tg_file.download_as_bytearray(read_timeout=120, write_timeout=120)
                result = await analyze_video_gemini(bytes(video_bytes), mime_type)
                label = f"Видео {idx}:\n" if count > 1 else ""
                analyses.append(f"{label}{result}".strip())
            except Exception as e:
                logger.error("Video error: %s", e, exc_info=True)
                analyses.append(f"Видео {idx}: не удалось обработать — {e}")

        video_analysis = "\n\n".join(analyses)

        if "503 UNAVAILABLE" in video_analysis or "high demand" in video_analysis:
            set_state(uid, "loading")
            # Сохраняем буфер: пользователь может нажать «Структурировать» повторно,
            # когда Gemini перестанет отдавать 503.
            user_text_buffers[uid] = text_buf
            user_media_buffers[uid] = media_snap
            user_video_buffers[uid] = video_snap
            await query.message.reply_text(
                "Gemini сейчас перегружен и не смог проанализировать видео. "
                "Файл в буфере сохранён — нажми «Структурировать» повторно позже.",
                reply_markup=kb_loading(uid),
            )
            return

        await query.message.reply_text("Видео обработано. Структурирую ТЗ...")

    await context.bot.send_chat_action(chat_id=query.message.chat_id, action="typing")
    try:
        reply = await structure_with_gpt(full_text, video_analysis)
    except Exception as e:
        logger.error("GPT error: %s", e, exc_info=True)
        set_state(uid, "idle")
        await query.message.reply_text(
            f"Ошибка при обращении к GPT: {e}",
            reply_markup=kb_main(),
        )
        return

    set_state(uid, "idle")
    clear_buffers(uid)

    for chunk in split_message(reply):
        try:
            await query.message.reply_text(chunk)
        except Exception as send_err:
            logger.error("Send error: %s", send_err, exc_info=True)
            await query.message.reply_text(
                chunk.encode("utf-8", errors="ignore").decode("utf-8")
            )

    if media_snap:
        await query.message.reply_text("Материалы от заказчика:")
        for chat_id, msg_id in media_snap:
            try:
                await context.bot.forward_message(
                    chat_id=query.message.chat_id,
                    from_chat_id=chat_id,
                    message_id=msg_id,
                )
            except Exception as e:
                logger.warning("Не удалось переслать %s: %s", msg_id, e)

    await query.message.reply_text("Готово.", reply_markup=kb_after_result())


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    msg: Message = update.message
    state = get_state(uid)

    if state == "processing":
        await msg.reply_text("Идёт обработка, подожди немного...")
        return

    if state != "loading":
        await msg.reply_text(
            "Нажми «Загрузить ТЗ» чтобы начать.",
            reply_markup=kb_main(),
        )
        return

    text_buf = user_text_buffers.setdefault(uid, [])
    media_buf = user_media_buffers.setdefault(uid, [])
    video_buf = user_video_buffers.setdefault(uid, [])

    is_video_document = bool(
        msg.document
        and msg.document.mime_type
        and msg.document.mime_type.startswith("video/")
    )
    is_media = bool(
        msg.photo or msg.audio or msg.voice or msg.sticker or msg.animation or
        (msg.document and not is_video_document)
    )

    if msg.text:
        text_buf.append(msg.text.strip())
    elif msg.caption:
        text_buf.append(msg.caption.strip())

    if msg.video:
        video_buf.append((
            msg.video.file_id,
            msg.video.mime_type or "video/mp4",
            msg.video.file_size,
        ))
    elif is_video_document:
        video_buf.append((
            msg.document.file_id,
            msg.document.mime_type or "video/mp4",
            msg.document.file_size,
        ))
    elif is_media:
        media_buf.append((msg.chat_id, msg.message_id))

    parts = []
    if text_buf:
        parts.append(f"текст: {count_label(len(text_buf))}")
    if video_buf:
        parts.append(f"видео: {len(video_buf)} шт.")
    if media_buf:
        parts.append(f"файлов/фото: {len(media_buf)} шт.")
    status = "Принято. " + ", ".join(parts) + "."

    await msg.reply_text(status, reply_markup=kb_loading(uid))


# ── Запуск ────────────────────────────────────────────────────────────────────
def main() -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("models", list_models))
    app.add_handler(CallbackQueryHandler(cb_load, pattern=CB_LOAD))
    app.add_handler(CallbackQueryHandler(cb_structure, pattern=CB_STRUCTURE))
    app.add_handler(CallbackQueryHandler(cb_clear, pattern=CB_CLEAR))
    app.add_handler(CallbackQueryHandler(cb_status, pattern=CB_STATUS))
    app.add_handler(CallbackQueryHandler(cb_help, pattern=CB_HELP))
    app.add_handler(CallbackQueryHandler(cb_back, pattern=CB_BACK))
    app.add_handler(MessageHandler(
        (filters.TEXT | filters.PHOTO | filters.VIDEO |
         filters.Document.ALL | filters.AUDIO | filters.VOICE) & ~filters.COMMAND,
        handle_message,
    ))

    logger.info("Бот запущен.")
    app.run_polling()


if __name__ == "__main__":
    main()
