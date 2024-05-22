import telebot
import config
from telebot import types
from datetime import date
from datetime import timedelta
import pandas as pd

from scripts.train import train_model

bot = telebot.TeleBot(config.token)

@bot.message_handler(commands=['start'])
def start(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton('Узнать F1 score лучшей модели')
    btn2 = types.KeyboardButton('Получить оценку размещения банкомата')
    markup.add(btn1, btn2)

    bot.send_message(
        message.chat.id,
        'Бот умеет производить оценку потенциального места для '
        'размещения банкомата по 5-ти балльной шкале',
        reply_markup=markup
    )


@bot.message_handler(content_types=['text'])
def bot_answer(message):

    if message.chat.type == 'private':
        if message.text == 'Назад':

            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            btn1 = types.KeyboardButton('Узнать текущую МАЕ модели')
            btn2 = types.KeyboardButton('Получить оценку размещения банкомата')
            markup.add(btn1, btn2)

            bot.send_message(
                message.chat.id,
                'Бот умеет производить оценку потенциального места для '
                'размещения банкомата',
                reply_markup=markup
            )

        elif message.text == 'Узнать текущую МАЕ модели':

            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            back = types.KeyboardButton('Назад')
            markup.add(back)

            df_mae = pd.read_csv('./outputs/mae.csv')
            mae = df_mae.loc[0]['MAE']

            msg = bot.send_message(message.chat.id, f'МАЕ для актуальной модели: {mae}',
                                   reply_markup=markup)

        elif message.text == 'Получить оценку размещения банкомата':

            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            back = types.KeyboardButton('Назад')
            markup.add(back)

            msg = bot.send_message(message.chat.id, 'Тут пока ничего нет ... Нужно осмыслить формат ввода',
                                   reply_markup=markup)


if __name__ == "__main__":
    bot.infinity_polling()