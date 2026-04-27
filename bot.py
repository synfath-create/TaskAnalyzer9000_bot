import os
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    CommandHandler,
    CallbackQueryHandler,
    filters,
)
from openai import OpenAI

# ── Логирование ──────────────────────────────────────────────────────────────
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

# ── Буфер сообщений (накапливаем до нажатия кнопки) ─────────────────────────
# user_id -> список строк
user_buffers: dict[int, list[str]] = {}

BUTTON_LABEL = "Структурировать ТЗ"
CALLBACK_STRUCTURE = "do_structure"


def get_buffer(user_id: int) -> list[str]:
    if user_id not in user_buffers:
        user_buffers[user_id] = []
    return user_buffers[user_id]


async def ask_gpt(full_text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": full_text},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


def split_message(text: str, max_len: int = 4000) -> list[str]:
    if len(text) <= max_len:
        return [text]
    parts = []
    while text:
        parts.append(text[:max_len])
        text = text[max_len:]
    return parts


def structure_button() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(BUTTON_LABEL, callback_data=CALLBACK_STRUCTURE)]]
    )


# ── Хэндлеры ─────────────────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Отправляй сообщения с задачей — можно несколько подряд.\n"
        "Когда всё скинешь, нажми кнопку «Структурировать ТЗ»."
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_buffers[user_id] = []
    await update.message.reply_text("Буфер очищен. Жду новую задачу.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    text = (update.message.text or "").strip()

    if not text:
        return

    buf = get_buffer(user_id)
    buf.append(text)

    # Показываем кнопку после каждого сообщения
    count = len(buf)
    status = f"Принято ({count} {'сообщение' if count == 1 else 'сообщения' if count < 5 else 'сообщений'}). Добавь ещё или нажми кнопку."
    await update.message.reply_text(status, reply_markup=structure_button())


async def handle_structure(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    buf = get_buffer(user_id)

    if not buf:
        await query.message.reply_text("Нет сообщений для обработки. Сначала отправь задачу.")
        return

    await context.bot.send_chat_action(chat_id=query.message.chat_id, action="typing")

    full_text = "\n\n---\n\n".join(buf)

    try:
        reply = await ask_gpt(full_text)
        # Очищаем буфер после успешной обработки
        user_buffers[user_id] = []
        for chunk in split_message(reply):
            await query.message.reply_text(chunk, parse_mode=None)
        await query.message.reply_text("Буфер очищен. Жду следующую задачу.")
    except Exception as e:
        logger.error("GPT error: %s", e)
        await query.message.reply_text("Ошибка при обращении к GPT. Попробуй ещё раз.")


# ── Запуск ────────────────────────────────────────────────────────────────────
def main() -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(handle_structure, pattern=CALLBACK_STRUCTURE))

    logger.info("Бот запущен. Ожидаю сообщения...")
    app.run_polling()


if __name__ == "__main__":
    main()
