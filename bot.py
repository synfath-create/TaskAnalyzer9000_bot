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
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

# ── Промпты ───────────────────────────────────────────────────────────────────
GEMINI_VIDEO_PROMPT = """Ты — эксперт по анализу рекламных видео-креативов для iGaming / apps / UGC / streamer ads.

Твоя задача — НЕ кратко пересказать видео, а максимально подробно извлечь из него факты для постановки задачи дизайнеру/моушн-дизайнеру/монтажеру.

СТРОГИЕ ПРАВИЛА:
- Не пиши вводные фразы.
- Не используй emoji, markdown-звездочки, декоративные заголовки.
- Если не можешь определить факт точно — пиши "предположительно".
- Если текст на экране виден — выписывай дословно.
- Если бонус/оффер/валюта/CTA есть на видео — обязательно выпиши.
- Если слот/игра не узнается по названию — опиши механику, сетку, символы, UI и бонусные элементы.
- Не ограничивайся общими словами вроде "игровой процесс". Нужно описывать, как именно работает визуал.

ВЫДАЙ СТРОГО В ФОРМАТЕ:

АНАЛИЗ ВИДЕО:
Хронометраж: [общая длительность]
Формат: [9:16 / 1:1 / 16:9 / другое]
Тип креатива: [UGC / Streamer PIP / Gameplay / Cinematic / Mixed / другое]

Структура по секундам:
- [0:00-0:XX] [что происходит в кадре, какие тексты/события/эффекты]
- [продолжай по каждому смысловому блоку]

Слот / игра:
Название: [если можно определить]
Если название не определено: [описание игры]
Сетка / поле: [например 5x5, 6 барабанов, линии, кластер и т.д.]
Ключевые элементы: [символы, множители, wheel, jackpot, free spins, buy bonus и т.д.]

Механика игрового процесса:
- Тип вращения: [непрерывное / отдельные спины / cascade / cluster / другое]
- Как появляются выигрыши: [описание]
- Как появляются множители: [описание]
- Как триггерится бонус: [описание]
- Как двигается колесо/бонусный элемент, если он есть: [описание]

BIG WIN / WIN-сцена:
- Есть ли BIG WIN: [да/нет]
- Текст на экране: [дословно]
- Счетчик выигрыша: [есть/нет, count-up/фиксированное значение]
- Визуальные эффекты: [монеты, вспышки, glow, shake и т.д.]

Бонус / оффер из видео:
- Есть ли бонус: [да/нет]
- Текст бонуса: [дословно]
- Валюта на видео: [если видна]
- Сумма/FS/проценты: [если видны]

UI / финальный экран:
- Кнопки: [дословно]
- Store badges: [Google Play / App Store / другое]
- CTA: [дословно + тайминг]
- Дополнительный текст: [дословно]

Стример / персонаж:
- Есть ли стример: [да/нет]
- Формат: [picture-in-picture / full screen / split screen / другое]
- Позиция на экране: [верх/низ/угол/центр]
- Внешность и образ: [кратко]
- Поведение: [комментирует, кричит, удивляется, смотрит на слот и т.д.]
- Эмоциональная динамика: [как меняется реакция]
- Попытка идентификации: [если можно определить кто это; иначе "не идентифицирован"]

Озвучка:
- Есть ли голос: [да/нет]
- Пол: [мужской/женский/не определить]
- Язык: [язык]
- Акцент: [нативный/иностранный/не определить]
- Стиль подачи: [стримерский/рекламный/спокойный/эмоциональный/агрессивный]
- Дословные фразы, если слышно: [список]

Субтитры:
- Есть ли: [да/нет]
- Текст: [дословно, если виден]
- Цвет/позиция/стиль: [описание]

Визуал:
- Локация/фон:
- Цветовая гамма:
- Свет:
- Эффекты:
- Общий стиль:

Монтаж и звук:
- Темп монтажа:
- Переходы:
- Музыка:
- SFX:

Что важно перенести в новую задачу:
- [список конкретных элементов]

Что нужно уточнить по видео:
- [список вопросов, если есть неоднозначности]
"""

GPT_DIALOGUE_ANALYSIS_PROMPT = """Ты — ассистент, который разбирает хаотичную переписку по задачам на рекламные креативы.

На входе — сообщения, пересланные пользователем из диалога с заказчиком/менеджером/командой. Сообщения могут быть без имен, с именами, с датами, с цитатами, кусками ТЗ, комментариями и правками.

Твоя задача:
1. Определить роли участников по контексту.
2. Отделить факты задачи от комментариев, вопросов, правок и служебных сообщений.
3. Сохранить дословные тексты для озвучки/субтитров/CTA/офферов.
4. Не терять контекст диалога.
5. Не придумывать факты.

СТРОГИЕ ПРАВИЛА:
- Если автор сообщения не определен — пиши "автор не определен".
- Если понятно, что пишет заказчик — помечай как "заказчик".
- Если понятно, что пишет менеджер/байер/дизайнер — помечай соответствующую роль.
- Не меняй дословные тексты креатива.
- Не исправляй язык озвучки/субтитров, если это текст для креатива.
- Не удаляй важные ограничения.

ВЫДАЙ СТРОГО В ФОРМАТЕ:

АНАЛИЗ ПЕРЕПИСКИ:

Участники и роли:
- [роль]: [как определено / по каким признакам]

Контекст диалога:
[кратко: что за задача, что обсуждают, что требуется]

Факты задачи:
- GEO: [если есть]
- Language: [если есть]
- Количество креативов: [если есть]
- Формат: [если есть]
- Размер: [если есть]
- Площадка: [если есть]
- Оффер: [если есть]
- Бонус: [если есть]
- Валюта: [если есть]
- Слот/игра/продукт: [если есть]
- Референсы: [если есть]

Дословные тексты для креатива:
[все фразы, озвучка, субтитры, CTA, офферы — дословно]

Комментарии заказчика:
[только требования и пожелания заказчика, дословно или максимально близко]

Правки / уточнения из диалога:
- [список]

Неоднозначности:
- [что непонятно или конфликтует]

Что важно учесть в финальном ТЗ:
- [список]
"""

GPT_FINAL_STRUCTURE_PROMPT = """Ты — senior creative task analyst для команды, которая делает рекламные креативы в iGaming / apps / UGC / motion.

Ты получаешь:
1. Анализ переписки и ролей
2. Анализ видео-референсов от Gemini
3. Исходный текст сообщений

Твоя задача — собрать итоговое рабочее ТЗ. Не просто переписать, а:
- извлечь факты;
- сопоставить текст и видео;
- определить недостающие данные;
- сделать логические выводы;
- дать рекомендации менеджеру;
- составить реалистичный план работы.

СТРОГИЕ ПРАВИЛА:
- Никаких emoji, звездочек, вводных фраз.
- Тексты озвучки/субтитров/CTA/офферов копируй дословно.
- Если данных нет — пиши "не указано".
- Если делаешь вывод по контексту — пиши "предположительно".
- Если GEO и язык позволяют определить валюту — обязательно предложи валюту в рекомендациях.
- Если в видео есть бонус, но в ТЗ бонус не указан — обязательно вынеси это в "НЕ УКАЗАНО / УТОЧНИТЬ".
- Если в видео есть слот, но название неизвестно — опиши механику и укажи, что название нужно уточнить.
- Если есть видео-референс со стримером — НЕ пиши "организовать съемку" по умолчанию.
- План работы должен быть под генерацию/адаптацию/монтаж, если заказчик прямо не просил съемку.
- Не пропускай: тип креатива, слот, валюту, бонус, озвучку, UI, CTA, стримера, механику слота.

ЛОГИКА ВАЛЮТЫ:
Если валюта не указана, но есть GEO:
- Турция → предположительно TRY / турецкая лира
- Польша → PLN
- Чехия → CZK
- Германия/Франция/Италия/Испания/Австрия → EUR
- Великобритания → GBP
- Канада → CAD
- Австралия → AUD
- Бразилия → BRL
- Аргентина → ARS
- Украина → UAH
Если уверенности нет — "не указано, нужно уточнить".

ФОРМАТ ОТВЕТА:

──────────────────────────
GEO: [страна / регион]
LANGUAGE: [язык]
CREATIVES: [число]
FORMAT: [video / img / pwa / carousel / другое]
SIZE: [размер]
CURRENCY: [указанная / предположительная / не указано]
BONUS: [указанный / найденный в видео / не указано]
──────────────────────────
ТИП КРЕАТИВА:
[например: Streamer PIP, UGC gameplay, cinematic gameplay, static banner]
──────────────────────────
КОММЕНТАРИЙ ЗАКАЗЧИКА:
[требования заказчика дословно или максимально близко; без текстов озвучки]
──────────────────────────
ДЕТАЛИ ЗАКАЗА:
Оффер: [продукт / приложение / слот]
Площадка: [Google Play / App Store / Meta / TikTok / не указано]
Подход: [генерация / адаптация / монтаж / съемка только если явно указана]
Дополнительно: [важные детали]
──────────────────────────
ТЕКСТЫ / ОЗВУЧКА:
[дословно, с нумерацией]
──────────────────────────
АНАЛИЗ РЕФЕРЕНСА:

Хронометраж:
Формат:
Тип креатива:

Структура:
- [тайминг] [событие]

Слот / игра:
- Название:
- Сетка / поле:
- Механика:
- Верхние/бонусные элементы:

Механика анимации слота:
- Вращение:
- Множители:
- Бонус/колесо:
- Выигрыш:

Стример:
- Формат:
- Позиция:
- Поведение:
- Эмоции:
- Идентификация:

BIG WIN:
- Текст:
- Счетчик:
- Визуальные эффекты:

Финальный экран / UI:
- Бонус:
- CTA:
- Кнопки:
- Store badges:

Озвучка:
- Пол:
- Язык:
- Акцент:
- Стиль:

Визуал:
Темп:
Музыка/SFX:
──────────────────────────
СОПОСТАВЛЕНИЕ ТЗ И РЕФЕРЕНСА:
- [что совпадает]
- [что есть в ТЗ, но не видно в видео]
- [что есть в видео, но не указано в ТЗ]
- [риски/важные замечания]
──────────────────────────
ПЛАН В РАБОТУ:
1. [шаг]
2. [шаг]
3. [шаг]

Важно: план должен описывать реализацию через генерацию/адаптацию/монтаж, если съемка не указана явно.
──────────────────────────
РЕКОМЕНДАЦИИ:
- Валюта: [что добавить]
- Бонус: [что делать]
- Слот: [что уточнить/сохранить]
- Стримеры: [3 вариации / разные типажи / AI или реальные]
- Озвучка: [что использовать]
──────────────────────────
НЕ УКАЗАНО / УТОЧНИТЬ:
- [конкретный вопрос]
- [конкретный вопрос]
"""

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
async def analyze_video_gemini(video_bytes: bytes, mime_type: str) -> str:
    tmp_path = None
    uploaded = None
    try:
        suffix = ".mp4" if "mp4" in mime_type else ".video"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        uploaded = genai.upload_file(path=tmp_path, mime_type=mime_type)

        while uploaded.state.name == "PROCESSING":
            time.sleep(2)
            uploaded = genai.get_file(uploaded.name)

        if uploaded.state.name == "FAILED":
            return "Видео не удалось проанализировать: Gemini не смог обработать файл."

        response = gemini_model.generate_content([GEMINI_VIDEO_PROMPT, uploaded])
        return response.text or "Видео проанализировано, но Gemini вернул пустой ответ."
    except Exception as e:
        logger.error("Gemini video error: %s", e, exc_info=True)
        return f"Видео не удалось проанализировать: {e}"
    finally:
        try:
            if uploaded:
                genai.delete_file(uploaded.name)
        except Exception:
            pass
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass


# ── GPT: анализ переписки ─────────────────────────────────────────────────────
async def analyze_dialogue_with_gpt(tz_text: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": GPT_DIALOGUE_ANALYSIS_PROMPT},
            {"role": "user", "content": f"ИСХОДНЫЕ СООБЩЕНИЯ:\n{tz_text}"},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or "Анализ переписки не получен."


# ── GPT: финальная структуризация ─────────────────────────────────────────────
async def structure_with_gpt(tz_text: str, dialogue_analysis: str, video_analysis: str | None) -> str:
    user_content = (
        f"АНАЛИЗ ПЕРЕПИСКИ И РОЛЕЙ:\n{dialogue_analysis}\n\n"
        f"АНАЛИЗ ВИДЕО ОТ GEMINI:\n{video_analysis or 'Видео не предоставлено.'}\n\n"
        f"ИСХОДНЫЕ СООБЩЕНИЯ:\n{tz_text}"
    )

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": GPT_FINAL_STRUCTURE_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.25,
    )
    return response.choices[0].message.content or "GPT вернул пустой ответ."


# ── Хэндлеры ─────────────────────────────────────────────────────────────────
async def list_models(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        models = genai.list_models()
        names = [m.name for m in models if "generateContent" in m.supported_generation_methods]
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
        "Нажми «Загрузить ТЗ» — пересылай переписку, комментарии заказчика, тексты, видео-референсы, фото и файлы. "
        "Затем нажми «Структурировать». Бот отдельно разберет переписку, видео и соберет финальное ТЗ.",
        reply_markup=kb_main(),
    )


async def cb_load(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    uid = query.from_user.id
    set_state(uid, "loading")
    clear_buffers(uid)
    await query.message.reply_text(
        "Буфер открыт. Пересылай всё от заказчика — переписку, текст ТЗ, видео, фото, файлы.\n"
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
        "2. Пересылай всё от заказчика — переписку, текст, видео, фото, файлы\n"
        "3. Нажми «Структурировать»\n\n"
        "Что делает бот:\n"
        "- отдельно разбирает переписку и роли участников\n"
        "- отдельно анализирует видео через Gemini\n"
        "- сопоставляет текст и референсы\n"
        "- достает бонусы, валюту, CTA, слот, механику, озвучку и UI\n"
        "- формирует рабочее ТЗ и список уточнений\n\n"
        "Видео: максимум зависит от лимитов Telegram/Gemini. Если обычным видео не проходит — отправляй как файл."
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
            "Нет текста. Добавь хотя бы одно текстовое сообщение с описанием задачи или перепиской.",
            reply_markup=kb_loading(uid),
        )
        return

    set_state(uid, "processing")
    media_snap = list(media_buf)
    video_snap = list(video_buf)
    full_text = "\n\n---\n\n".join(text_buf)
    clear_buffers(uid)

    await context.bot.send_chat_action(chat_id=query.message.chat_id, action="typing")

    # 1. Анализ видео через Gemini
    video_analysis = None
    if video_snap:
        count = len(video_snap)
        await query.message.reply_text(f"Анализирую видео через Gemini: {count} шт.")
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
        await query.message.reply_text("Видео проанализировано. Разбираю переписку и роли...")
    else:
        await query.message.reply_text("Разбираю переписку и роли...")

    # 2. Анализ переписки через GPT
    await context.bot.send_chat_action(chat_id=query.message.chat_id, action="typing")
    try:
        dialogue_analysis = await analyze_dialogue_with_gpt(full_text)
    except Exception as e:
        logger.error("Dialogue GPT error: %s", e, exc_info=True)
        set_state(uid, "idle")
        await query.message.reply_text(
            "Ошибка при анализе переписки. Попробуй ещё раз.",
            reply_markup=kb_main(),
        )
        return

    # 3. Финальная структуризация через GPT
    await query.message.reply_text("Собираю финальное ТЗ...")
    await context.bot.send_chat_action(chat_id=query.message.chat_id, action="typing")
    try:
        reply = await structure_with_gpt(full_text, dialogue_analysis, video_analysis)
    except Exception as e:
        logger.error("Final GPT error: %s", e, exc_info=True)
        set_state(uid, "idle")
        await query.message.reply_text(
            "Ошибка при финальной структуризации. Попробуй ещё раз.",
            reply_markup=kb_main(),
        )
        return

    set_state(uid, "idle")

    # 4. Результат
    for chunk in split_message(reply):
        try:
            await query.message.reply_text(chunk)
        except Exception as send_err:
            logger.error("Send error: %s", send_err, exc_info=True)
            await query.message.reply_text(
                chunk.encode("utf-8", errors="ignore").decode("utf-8")
            )

    # 5. Медиа-материалы
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
        await msg.reply_text("Идёт обработка. Новые сообщения пока не добавлены в буфер.")
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
