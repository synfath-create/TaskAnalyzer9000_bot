import os
import logging
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    CommandHandler,
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

# ── История диалогов (in-memory, per user) ────────────────────────────────────
user_histories: dict[int, list[dict]] = {}


def get_history(user_id: int) -> list[dict]:
    if user_id not in user_histories:
        user_histories[user_id] = []
    return user_histories[user_id]


async def ask_gpt(user_id: int, user_message: str) -> str:
    history = get_history(user_id)
    history.append({"role": "user", "content": user_message})

    # Держим не более 20 сообщений, чтобы не переполнять контекст
    trimmed = history[-20:]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + trimmed,
        temperature=0.7,
    )
    assistant_msg = response.choices[0].message.content
    history.append({"role": "assistant", "content": assistant_msg})
    return assistant_msg


# ── Хэндлеры ─────────────────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 Привет! Я твой ассистент по задачам.\n\n"
        "Отправь мне бриф, задачу или любые материалы — "
        "я структурирую их, определю количество креативов "
        "и напишу комментарий от лица заказчика.\n\n"
        "Просто напиши что нужно сделать 👇"
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_histories[user_id] = []
    await update.message.reply_text("🔄 История очищена. Начинаем заново!")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_text = update.message.text or ""

    if not user_text.strip():
        await update.message.reply_text("Пришли текст задачи или бриф 📝")
        return

    # Показываем индикатор печатания
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        reply = await ask_gpt(user_id, user_text)
        # Telegram ограничивает сообщение 4096 символами — делим при необходимости
        for chunk in split_message(reply):
            await update.message.reply_text(chunk, parse_mode=None)
    except Exception as e:
        logger.error("GPT error: %s", e)
        await update.message.reply_text(
            "⚠️ Произошла ошибка при обращении к GPT. Попробуй ещё раз."
        )


def split_message(text: str, max_len: int = 4000) -> list[str]:
    """Разбивает длинный текст на части по max_len символов."""
    if len(text) <= max_len:
        return [text]
    parts = []
    while text:
        parts.append(text[:max_len])
        text = text[max_len:]
    return parts


# ── Запуск ────────────────────────────────────────────────────────────────────
def main() -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Бот запущен. Ожидаю сообщения...")
    app.run_polling()


if __name__ == "__main__":
    main()
