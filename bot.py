import os
import logging
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

# ── Логирование ───────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Клиент OpenAI ─────────────────────────────────────────────────────────────
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ── Системный промпт ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Ты — менеджер по рекламным задачам. Твоя работа — аккуратно и точно структурировать входящий бриф или задачу.

СТРОГИЕ ПРАВИЛА:
- Никаких смайлов, emoji, звёздочек, украшений.
- Никаких пояснений, интерпретаций, «простых слов» — только факты из задачи.
- Комментарий заказчика — копируешь ДОСЛОВНО, без единого изменения.
- Если поле не указано в задаче — пишешь: не указано.
- Никаких вводных фраз типа «Конечно!», «Вот структура», «Отлично» и т.п.

ФОРМАТ ОТВЕТА — строго такой, без отступлений:

──────────────────────────
GEO: [страна / регион]
LANGUAGE: [язык креативов]
CREATIVES: [число]
FORMAT: [формат — например: статика, видео, карусель]
SIZE: [размеры — например: 1080x1080, 1080x1920]
CURRENCY: [валюта]
BONUS: [условия бонуса или «нет»]
──────────────────────────
КОММЕНТАРИЙ ЗАКАЗЧИКА:
[дословный текст от заказчика — без изменений, без сокращений]
──────────────────────────
ДЕТАЛИ ЗАКАЗА:
Оффер: [название продукта / оффера]
Площадка: [где крутится реклама]
ЦА: [целевая аудитория, если указана]
Дедлайн: [если указан]
Дополнительно: [всё остальное важное из задачи — требования, ограничения, пожелания]
──────────────────────────

Если каких-то данных нет в задаче — в самом конце, отдельным блоком:
НЕ УКАЗАНО / УТОЧНИТЬ:
- [список полей, которые нужно уточнить у заказчика]
"""

# ── Состояния ─────────────────────────────────────────────────────────────────
# "idle"    — ждём нажатия «Загрузить ТЗ»
# "loading" — принимаем сообщения в буфер

user_states: dict[int, str] = {}
user_text_buffers: dict[int, list[str]] = {}
# (chat_id, message_id) для пересылки медиа
user_media_buffers: dict[int, list[tuple[int, int]]] = {}

CB_LOAD      = "cb_load"
CB_STRUCTURE = "cb_structure"
CB_CANCEL    = "cb_cancel"


def get_state(uid: int) -> str:
    return user_states.get(uid, "idle")

def set_state(uid: int, state: str) -> None:
    user_states[uid] = state

def clear_buffers(uid: int) -> None:
    user_text_buffers[uid] = []
    user_media_buffers[uid] = []

def btn_load() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("Загрузить ТЗ", callback_data=CB_LOAD)
    ]])

def btn_working() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("Структурировать", callback_data=CB_STRUCTURE),
        InlineKeyboardButton("Отмена", callback_data=CB_CANCEL),
    ]])

def split_message(text: str, max_len: int = 4000) -> list[str]:
    if len(text) <= max_len:
        return [text]
    parts = []
    while text:
        parts.append(text[:max_len])
        text = text[max_len:]
    return parts

async def ask_gpt(full_text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": full_text},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content

def count_label(n: int) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return f"{n} сообщение"
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return f"{n} сообщения"
    return f"{n} сообщений"


# ── Хэндлеры ─────────────────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    set_state(uid, "idle")
    clear_buffers(uid)
    await update.message.reply_text(
        "Нажми кнопку чтобы начать загрузку ТЗ.",
        reply_markup=btn_load(),
    )


async def cb_load(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    uid = query.from_user.id
    set_state(uid, "loading")
    clear_buffers(uid)
    await query.message.reply_text(
        "Готов принимать. Пересылай сообщения с задачей — текст, фото, файлы, всё что есть.\n"
        "Когда закончишь — нажми «Структурировать».",
        reply_markup=btn_working(),
    )


async def cb_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    uid = query.from_user.id
    set_state(uid, "idle")
    clear_buffers(uid)
    await query.message.reply_text(
        "Отменено. Буфер очищен.",
        reply_markup=btn_load(),
    )


async def cb_structure(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    uid = query.from_user.id

    text_buf  = user_text_buffers.get(uid, [])
    media_buf = user_media_buffers.get(uid, [])

    if not text_buf and not media_buf:
        await query.message.reply_text(
            "Буфер пустой — сначала загрузи сообщения.",
            reply_markup=btn_load(),
        )
        return

    if not text_buf:
        await query.message.reply_text(
            "Нет текста для анализа. Добавь хотя бы одно текстовое сообщение с описанием задачи.",
            reply_markup=btn_working(),
        )
        return

    await context.bot.send_chat_action(chat_id=query.message.chat_id, action="typing")

    full_text = "\n\n---\n\n".join(text_buf)

    try:
        reply = await ask_gpt(full_text)
    except Exception as e:
        logger.error("GPT error: %s", e)
        await query.message.reply_text("Ошибка при обращении к GPT. Попробуй ещё раз.")
        return

    # Снимаем снепшот и сбрасываем
    media_snap = list(media_buf)
    set_state(uid, "idle")
    clear_buffers(uid)

# 1. Структурированное ТЗ
    for chunk in split_message(reply):
        try:
            await query.message.reply_text(chunk)
        except Exception as send_err:
            logger.error("Send error: %s", send_err, exc_info=True)
            await query.message.reply_text(chunk.encode("utf-8", errors="ignore").decode("utf-8"))

    # 2. Материалы — пересылаем оригиналы
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

    # 3. Готово
    await query.message.reply_text(
        "Готово. Для следующей задачи нажми кнопку.",
        reply_markup=btn_load(),
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    msg: Message = update.message

    if get_state(uid) != "loading":
        await msg.reply_text(
            "Нажми «Загрузить ТЗ» чтобы начать.",
            reply_markup=btn_load(),
        )
        return

    text_buf  = user_text_buffers.setdefault(uid, [])
    media_buf = user_media_buffers.setdefault(uid, [])

    has_media = bool(
        msg.photo or msg.video or msg.document or
        msg.audio or msg.voice or msg.sticker or msg.animation
    )

    # Текст или подпись к медиа
    if msg.text:
        text_buf.append(msg.text.strip())
    elif msg.caption:
        text_buf.append(msg.caption.strip())

    # Медиа — сохраняем для пересылки
    if has_media:
        media_buf.append((msg.chat_id, msg.message_id))

    # Статус
    parts = []
    if text_buf:
        parts.append(f"текст: {count_label(len(text_buf))}")
    if media_buf:
        parts.append(f"файлов/фото: {len(media_buf)}")
    status = "Принято — " + ", ".join(parts) + "."

    await msg.reply_text(status, reply_markup=btn_working())


# ── Запуск ────────────────────────────────────────────────────────────────────
def main() -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(cb_load,      pattern=CB_LOAD))
    app.add_handler(CallbackQueryHandler(cb_cancel,    pattern=CB_CANCEL))
    app.add_handler(CallbackQueryHandler(cb_structure, pattern=CB_STRUCTURE))
    app.add_handler(MessageHandler(
        (filters.TEXT | filters.PHOTO | filters.VIDEO |
         filters.Document.ALL | filters.AUDIO | filters.VOICE) & ~filters.COMMAND,
        handle_message,
    ))

    logger.info("Бот запущен.")
    app.run_polling()


if __name__ == "__main__":
    main()
