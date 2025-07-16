# import joblib
# from dotenv import load_dotenv
# import os
# from telegram import Update
# from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
#
# import torch
# from transformers import pipeline
#
# # Load environment variables from .env file
# load_dotenv()
# TOKEN = os.getenv("API_TOKEN")
#
# async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     user_text = update.message.text     # Take text from user
#     chat_id = update.message.chat_id    # Take chat_id with the user
#     print(f"Message from {chat_id}: {user_text}")
#
#     # Загружаем модель
#     model = joblib.load('review_model.pkl')
#
#     # Пример использования
#     text = user_text
#     prediction = model.predict([text])[0]
#
#     if prediction == 1:
#         await update.message.reply_text(f"You said: ✅ Positive review")
#     else:
#         await update.message.reply_text(f"You said: ❌ Negative review")
#
#     # PART WITH BIG MODEL
#     # # Create the pipeline
#     # sentiment_pipe = pipeline("sentiment-analysis")
#     #
#     # # Use the pipeline
#     # result = sentiment_pipe(user_text)
#     #
#     # # Example: reply back
#     # await update.message.reply_text(f"You said: {result[0]['label']}")
#
# async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     await update.message.reply_text("Hi! Send me a message 😎")
#
# # Main bot setup
# if __name__ == "__main__":
#     app = ApplicationBuilder().token(TOKEN).build()
#
#     app.add_handler(CommandHandler("start", start))
#     app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
#
#     print("Bot is running...")
#     app.run_polling()
#

######################################################################
######################################################################
######################################################################

import torch
from transformers import pipeline

import os
from dotenv import load_dotenv
import joblib
import pandas as pd
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, MessageHandler, ContextTypes, filters
)
import datetime

# Загрузка токенов
load_dotenv()
TOKEN = os.getenv("API_TOKEN")
ID_ILIA = os.getenv("ID_ILIA")

# Загрузка модели
model = joblib.load("review_model_1.pkl") # Our model
sentiment_pipe = pipeline("sentiment-analysis") # Our spare model



# ID чата модератора (можно временно использовать свой ID)
MODERATOR_CHAT_ID = ID_ILIA  # ← замени на свой Telegram ID

# Путь к CSV для логирования
LOG_FILE = "flagged_comments.csv"

# Создание пустого CSV если он не существует
try:
    pd.read_csv(LOG_FILE)
except FileNotFoundError:
    pd.DataFrame(columns=["datetime", "user", "text", "negative_score"]).to_csv(LOG_FILE, index=False)

# Обработка входящих сообщений в группах
async def moderate_comment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    user = update.message.from_user.full_name
    print(text)

    # Получаем вероятности
    probas = model.predict_proba([text])[0] # Our model
    negative_score = probas[0]
    positive_score = probas[1]
    print(probas)

    result = sentiment_pipe(text) # Our spare model
    # print(result[0]['score'])
    print(result[0]['label'])

    # Порог отсечения
    # if result[0]['label'] == 'NEGATIVE': # Our spare model
    if negative_score > 0.6 and result[0]['label'] == 'NEGATIVE': # Our  model
        # Удаляем комментарий
        await update.message.delete()

        # Отправляем сообщение модератору
        await context.bot.send_message(
            chat_id=MODERATOR_CHAT_ID,
            text=f"🚨 *Flagged comment removed!*\n\n"
                 f"*User:* {user}\n"
                 f"*Negative score:* {round(negative_score * 100, 2)}%\n"
                 f"*Text:* `{text}`",
            parse_mode='Markdown'
        )

        # Логируем
        new_entry = {
            "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": user,
            "text": text,
            "negative_score": round(negative_score * 100, 2)
        }
        log_df = pd.read_csv(LOG_FILE)
        log_df = pd.concat([log_df, pd.DataFrame([new_entry])], ignore_index=True)
        log_df.to_csv(LOG_FILE, index=False)

# Инициализация бота
application = ApplicationBuilder().token(TOKEN).build()

# Обработка текстовых сообщений в группах
application.add_handler(MessageHandler(filters.TEXT & filters.ChatType.GROUPS, moderate_comment))

# Запуск бота
application.run_polling()










    # # PART WITH BIG MODEL
    # # Create the pipeline
    # sentiment_pipe = pipeline("sentiment-analysis")
    #
    # # Use the pipeline
    # result = sentiment_pipe(user_text)
    #
    # # Example: reply back
    # await update.message.reply_text(f"You said: {result[0]['label']}")