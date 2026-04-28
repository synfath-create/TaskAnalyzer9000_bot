import os
import logging
import tempfile
import time
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
import google.generativeai as genai

# ── Логирование ───────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Клиенты ───────────────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-1.5-pro-latest")

# ── Промпты ───────────────────────────────────────────────────────────────────
GEMINI_VIDEO_PROMPT = """Ты анализируешь рекламный видео-креатив. Без вводных фраз, без смайлов.

Проанализируй видео и выдай строго в этом формате:

АНАЛИЗ ВИДЕО:
Хронометраж: [общая длительность]
Структура:
- [0:00-0:XX] [что происходит]
- [продолжай по каждому смысловому блоку]

Визуал: [локация, освещение, цветовая гамма, стиль съёмки]
Темп монтажа: [медленный / средний / быстрый, cuts каждые N секунд]
Озвучка: [есть/нет, мужская/женская, тон, дословно основные фразы если слышно]
Субтитры: [есть/нет, цвет, позиция]
Музыка: [есть/нет, характер]
CTA: [есть/нет, текст, тайминг появления]
Формат: [соотношение сторон]
Технические особенности: [всё остальное важное]"""

GPT_STRUCTURE_PROMPT = """Ты — менеджер по рекламным задачам. Ты получаешь:
1. Анализ референс-видео от Gemini
2. Текст ТЗ от заказчика

Твоя задача — сопоставить их, структурировать ТЗ и составить план в работу.

СТРОГИЕ ПРАВИЛА:
- Никаких смайлов, emoji, звёздочек, украшений.
- Только факты — никаких интерпретаций и воды.
- Тексты озвучки/субтитры/CTA — дословно как у заказчика.
- Если поле не указано — пишешь: не указано.
- Никаких вводных фраз.
- В "НЕ УКАЗАНО / УТОЧНИТЬ" — только реально отсутствующие технические параметры.

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
[анализ видео от Gemini — структуру, тайминги, визуал, озвучку, CTA]
──────────────────────────
СОПОСТАВЛЕНИЕ:
[3-5 строк — что в видео совпадает с ТЗ, что расходится, на что обратить внимание]
──────────────────────────
ПЛАН В РАБОТУ:
[нумерованный чеклист конкретных шагов для реализации этого крео — от съёмки до финального файла]
──────────────────────────

Если не хватает технических параметров:
НЕ УКАЗАНО / УТОЧНИТЬ:
- [список]"""

# ── Состояния ─────────────────────────────────────────────────────────────────
user_states: dict[int, str] = {}
user_text_buffers: dict[int, list[str]] = {}
user_media_buffers: dict[int, list[tuple[int, int]]] = {}
user_video_buffers: dict[int, list[tuple[str, str]]] = {}

CB_LOAD      = "cb_load"
CB_STRUCTURE = "cb_structure"
CB_CLEAR     = "cb_clear"
CB_STATUS    = "cb_status"
CB_HELP      = "cb_help"
CB_BACK      = "cb_back"


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
    if t: parts.append(f"текст: {t} шт.")
    if v: parts.append(f"видео: {v} шт.")
    if m: parts.append(f"файлов/фото: {m} шт.")
    return "В буфере: " + ", ".join(parts)

# ── Клавиатуры ────────────────────────────────────────────────────────────────
def kb_main() -> InlineKeyboardMarkup:
    """Главное меню — режим ожидания"""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Загрузить ТЗ", callback_data=CB_LOAD)],
        [InlineKeyboardButton("Помощь", callback_data=CB_HELP)],
    ])

def kb_loading(uid: int) -> InlineKeyboardMarkup:
    """Меню во время загрузки"""
    summary = buf_summary(uid)
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Структурировать", callback_data=CB_STRUCTURE)],
        [
            InlineKeyboardButton("Статус буфера", callback_data=CB_STATUS),
            InlineKeyboardButton("Очистить буфер", callback_data=CB_CLEAR),
        ],
        [InlineKeyboardButton("Отмена", callback_data=CB_BACK)],
    ])

def kb_after_result() -> InlineKeyboardMarkup:
    """После выдачи результата"""
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
async def analyze_video_gemini(video_bytes: bytes, mime_type: str) -> str:
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        uploaded = genai.upload_file(path=tmp_path, mime_type=mime_type)

        while uploaded.state.name == "PROCESSING":
            time.sleep(2)
            uploaded = genai.get_file(uploaded.name)

        response = gemini_model.generate_content([GEMINI_VIDEO_PROMPT, uploaded])
        genai.delete_file(uploaded.name)
        os.unlink(tmp_path)
        return response.text
    except Exception as e:
        logger.error("Gemini video error: %s", e, exc_info=True)
        return f"Видео не удалось проанализировать: {e}"


# ── GPT: структуризация ───────────────────────────────────────────────────────
async def structure_with_gpt(tz_text: str, video_analysis: str | None) -> str:
    if video_analysis:
        user_content = (
            f"АНАЛИЗ ВИДЕО (от Gemini):\n{video_analysis}\n\n"
            f"ТЗ ОТ ЗАКАЗЧИКА:\n{tz_text}"
        )
    else:
        user_content = f"ТЗ ОТ ЗАКАЗЧИКА:\n{tz_text}"

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": GPT_STRUCTURE_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


# ── Хэндлеры ─────────────────────────────────────────────────────────────────
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
        "Видео: максимум 20 МБ. Если больше — отправляй как файл (скрепка -> файл)."
    )
    await query.message.reply_text(help_text, reply_markup=kb_main())


async def cb_structure(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    uid = query.from_user.id

    text_buf  = user_text_buffers.get(uid, [])
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

    # Снимаем снепшот и блокируем повторное нажатие
    set_state(uid, "processing")
    media_snap = list(media_buf)
    video_snap = list(video_buf)
    full_text  = "\n\n---\n\n".join(text_buf)
    clear_buffers(uid)

    await context.bot.send_chat_action(chat_id=query.message.chat_id, action="typing")

    # 1. Анализ видео через Gemini
    video_analysis = None
    if video_snap:
        count = len(video_snap)
        await query.message.reply_text(
            f"Анализирую {count} видео через Gemini... Это может занять 20-40 секунд."
        )
        analyses = []
        for idx, (file_id, mime_type) in enumerate(video_snap, 1):
            try:
                await context.bot.send_chat_action(chat_id=query.message.chat_id, action="typing")
                tg_file = await context.bot.get_file(file_id)
                video_bytes = await tg_file.download_as_bytearray()
                result = await analyze_video_gemini(bytes(video_bytes), mime_type)
                label = f"Видео {idx}:\n" if count > 1 else ""
                analyses.append(f"{label}{result}".strip())
            except Exception as e:
                logger.error("Video error: %s", e, exc_info=True)
                analyses.append(f"Видео {idx}: не удалось обработать — {e}")
        video_analysis = "\n\n".join(analyses)
        await query.message.reply_text("Видео проанализировано. Структурирую ТЗ...")

    # 2. Структуризация через GPT
    await context.bot.send_chat_action(chat_id=query.message.chat_id, action="typing")
    try:
        reply = await structure_with_gpt(full_text, video_analysis)
    except Exception as e:
        logger.error("GPT error: %s", e, exc_info=True)
        set_state(uid, "idle")
        await query.message.reply_text(
            "Ошибка при обращении к GPT. Попробуй ещё раз.",
            reply_markup=kb_main(),
        )
        return

    set_state(uid, "idle")

    # 3. Результат
    for chunk in split_message(reply):
        try:
            await query.message.reply_text(chunk)
        except Exception as send_err:
            logger.error("Send error: %s", send_err, exc_info=True)
            await query.message.reply_text(
                chunk.encode("utf-8", errors="ignore").decode("utf-8")
            )

    # 4. Медиа-материалы
    if media_snap:
        await query.message.reply_text("Материалы от заказчика:")
        for (chat_id, msg_id) in media_snap:
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

    text_buf  = user_text_buffers.setdefault(uid, [])
    media_buf = user_media_buffers.setdefault(uid, [])
    video_buf = user_video_buffers.setdefault(uid, [])

    is_video = bool(
        msg.video or
        (msg.document and msg.document.mime_type and msg.document.mime_type.startswith("video/"))
    )
    is_media = bool(
        msg.photo or msg.audio or msg.voice or msg.sticker or msg.animation or
        (msg.document and not (msg.document.mime_type and msg.document.mime_type.startswith("video/")))
    )

    if msg.text:
        text_buf.append(msg.text.strip())
    elif msg.caption:
        text_buf.append(msg.caption.strip())

    if msg.video:
        video_buf.append((msg.video.file_id, msg.video.mime_type or "video/mp4"))
    elif msg.document and msg.document.mime_type and msg.document.mime_type.startswith("video/"):
        video_buf.append((msg.document.file_id, msg.document.mime_type))
    elif is_media:
        media_buf.append((msg.chat_id, msg.message_id))

    parts = []
    if text_buf:  parts.append(f"текст: {count_label(len(text_buf))}")
    if video_buf: parts.append(f"видео: {len(video_buf)} шт.")
    if media_buf: parts.append(f"файлов/фото: {len(media_buf)} шт.")
    status = "Принято. " + ", ".join(parts) + "."

    await msg.reply_text(status, reply_markup=kb_loading(uid))


# ── Запуск ────────────────────────────────────────────────────────────────────
def main() -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(cb_load,      pattern=CB_LOAD))
    app.add_handler(CallbackQueryHandler(cb_structure, pattern=CB_STRUCTURE))
    app.add_handler(CallbackQueryHandler(cb_clear,     pattern=CB_CLEAR))
    app.add_handler(CallbackQueryHandler(cb_status,    pattern=CB_STATUS))
    app.add_handler(CallbackQueryHandler(cb_help,      pattern=CB_HELP))
    app.add_handler(CallbackQueryHandler(cb_back,      pattern=CB_BACK))
    app.add_handler(MessageHandler(
        (filters.TEXT | filters.PHOTO | filters.VIDEO |
         filters.Document.ALL | filters.AUDIO | filters.VOICE) & ~filters.COMMAND,
        handle_message,
    ))

    logger.info("Бот запущен.")
    app.run_polling()


if __name__ == "__main__":
    main()
