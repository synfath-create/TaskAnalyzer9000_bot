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
        await query.message.reply_text(chunk)

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
Что на экране: Открыто банковское приложение PhonePe.
Крупная сумма: ₹ 8,42,150.00 (8.4 лакха рупий).
Анимация: Зеленая галочка «Transfer Successful».
Диалог:
Оператор: Ого! 8 лакхов?! Брат, ты что, банк хакнул? Откуда такие бабки?
Парень: Какой банк? Это всё Coin Volcano. Пока все смотрят на самолетики, я просто ловлю извержения золота.
Текст на экране: «Заработал за неделю: ₹ 8,42,150 💸»

Сцена 3: Демонстрация игры (0:10 – 0:25)
Действие в кадре: Парень открывает игру Coin Volcano. Камера снимает экран крупным планом.
На экране - Геймплей Coin Volcano:
Видно сетку 3х3, заполненную золотыми монетами.
Процесс: Парень нажимает на спин. Сверху начинают трястись вулканы.
Один вулкан извергается и добавляет на монеты множители: x10, x20, x50.
Сумма выигрыша внизу экрана стремительно растет: 2000... 8000... 15000...
Диалог:
Парень: Смотри, всё элементарно. Главное - дождаться, когда проснутся вулканы. Видишь? Один выстрел - и мой баланс летит в космос! В этой игре монеты буквально липнут к рукам. Забрал бонус - и на вывод!
Текст на экране: «Trigger the Volcano - Catch the Gold!»

Сцена 4: Призыв к действию (0:25 – 0:30)
Действие в кадре: Парень довольно улыбается, показывает большой палец в камеру. На фоне ярко блестит мотоцикл.
Диалог:
Парень: Хватит терять время на ерунду. Ссылка внизу. Заставляй свой вулкан извергать золото прямо сейчас!

Финальный экран:
Логотип Coin Volcano.
Крупная пульсирующая кнопка «PLAY NOW».
Текст: «JOIN THE WINNERS! 🏆»    material_message_ids: list[int] = field(default_factory=list)


# user_id -> UserTaskBuffer
user_buffers: dict[int, UserTaskBuffer] = {}

BUTTON_UPLOAD = "Загрузить ТЗ"
BUTTON_STRUCTURE = "Структурировать"
CALLBACK_UPLOAD = "upload_tz"
CALLBACK_STRUCTURE = "do_structure"


def get_buffer(user_id: int) -> UserTaskBuffer:
    if user_id not in user_buffers:
        user_buffers[user_id] = UserTaskBuffer()
    return user_buffers[user_id]


def main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(BUTTON_UPLOAD, callback_data=CALLBACK_UPLOAD)]]
    )


def collection_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(BUTTON_STRUCTURE, callback_data=CALLBACK_STRUCTURE)]]
    )


async def ask_gpt(full_text: str) -> str:
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": full_text},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def split_message(text: str, max_len: int = 3900) -> list[str]:
    if len(text) <= max_len:
        return [text]

    parts = []
    current = text
    while len(current) > max_len:
        split_at = current.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len
        parts.append(current[:split_at].strip())
        current = current[split_at:].strip()

    if current:
        parts.append(current)
    return parts


def plural_messages(count: int) -> str:
    if count % 10 == 1 and count % 100 != 11:
        return "сообщение"
    if 2 <= count % 10 <= 4 and not 12 <= count % 100 <= 14:
        return "сообщения"
    return "сообщений"


# ── Хэндлеры ─────────────────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Нажми «Загрузить ТЗ», затем отправь все сообщения и материалы по задаче. "
        "Когда всё будет загружено, нажми «Структурировать».",
        reply_markup=main_menu(),
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_buffers[user_id] = UserTaskBuffer()
    await update.message.reply_text(
        "Буфер очищен. Для новой задачи нажми «Загрузить ТЗ».",
        reply_markup=main_menu(),
    )


async def handle_upload_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    user_buffers[user_id] = UserTaskBuffer(is_collecting=True)

    await query.message.reply_text(
        "Режим загрузки ТЗ включён. Отправляй текст, фото, видео, документы и другие материалы. "
        "После последнего сообщения нажми «Структурировать».",
        reply_markup=collection_menu(),
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    buf = get_buffer(user_id)

    if not buf.is_collecting:
        await update.message.reply_text(
            "Сначала нажми «Загрузить ТЗ», чтобы бот начал принимать задачу.",
            reply_markup=main_menu(),
        )
        return

    text = (update.message.text or update.message.caption or "").strip()

    if text:
        buf.text_messages.append(text)

    # Всё, что не является чистым текстовым сообщением, считаем материалом.
    # Потом бот скопирует это ниже структурированного ТЗ.
    is_plain_text = bool(update.message.text) and not any([
        update.message.photo,
        update.message.video,
        update.message.document,
        update.message.animation,
        update.message.audio,
        update.message.voice,
        update.message.video_note,
        update.message.sticker,
    ])

    if not is_plain_text:
        buf.material_message_ids.append(update.message.message_id)

    text_count = len(buf.text_messages)
    material_count = len(buf.material_message_ids)

    await update.message.reply_text(
        f"Принято: {text_count} {plural_messages(text_count)} с текстом, материалов: {material_count}. "
        "Добавь ещё или нажми «Структурировать».",
        reply_markup=collection_menu(),
    )


async def handle_structure(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    buf = get_buffer(user_id)

    if not buf.is_collecting:
        await query.message.reply_text(
            "Сначала нажми «Загрузить ТЗ» и отправь материалы по задаче.",
            reply_markup=main_menu(),
        )
        return

    if not buf.text_messages and not buf.material_message_ids:
        await query.message.reply_text("Буфер пустой. Отправь текст или материалы задачи.")
        return

    await context.bot.send_chat_action(chat_id=query.message.chat_id, action="typing")

    full_text = "\n\n--- СЛЕДУЮЩЕЕ СООБЩЕНИЕ ---\n\n".join(buf.text_messages)
    if not full_text:
        full_text = "Текстовые комментарии не указаны. Пользователь отправил только материалы."

    if buf.material_message_ids:
        full_text += f"\n\n--- МАТЕРИАЛЫ ---\nПользователь отправил вложения: {len(buf.material_message_ids)} шт. Они будут пересланы ниже структурированного ТЗ."

    try:
        structured_reply = await ask_gpt(full_text)

        for chunk in split_message(structured_reply):
            await query.message.reply_text(chunk, parse_mode=None)

        if buf.material_message_ids:
            await query.message.reply_text("Пересланные материалы:")
            for message_id in buf.material_message_ids:
                try:
                    await context.bot.copy_message(
                        chat_id=query.message.chat_id,
                        from_chat_id=query.message.chat_id,
                        message_id=message_id,
                    )
                except Exception as copy_error:
                    logger.error("Material copy error: %s", copy_error)
                    await query.message.reply_text(
                        "Не удалось переслать один из материалов. Возможно, Telegram ограничил копирование этого типа сообщения."
                    )

        user_buffers[user_id] = UserTaskBuffer()
        await query.message.reply_text(
            "Готово. Буфер очищен. Для новой задачи нажми «Загрузить ТЗ».",
            reply_markup=main_menu(),
        )
    except Exception as e:
        logger.error("GPT error: %s", e)
        await query.message.reply_text("Ошибка при обращении к GPT. Буфер не очищен, можно попробовать ещё раз.")


# ── Запуск ────────────────────────────────────────────────────────────────────
def main() -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CallbackQueryHandler(handle_upload_button, pattern=f"^{CALLBACK_UPLOAD}$"))
    app.add_handler(CallbackQueryHandler(handle_structure, pattern=f"^{CALLBACK_STRUCTURE}$"))
    app.add_handler(MessageHandler(~filters.COMMAND, handle_message))

    logger.info("Бот запущен. Ожидаю сообщения...")
    app.run_polling()


if __name__ == "__main__":
    main()
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
