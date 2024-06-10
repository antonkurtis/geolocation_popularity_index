import telebot
import config
from telebot import types
import pandas as pd
from datetime import datetime

from scripts.scr import *
from scripts.train import train_model
from scripts.infer import inference_model
from scripts.get_data import get_all_futures_dataset

from airflow.api.client.local_client import Client

bot = telebot.TeleBot(config.token)


@bot.message_handler(commands=['start'])
def start(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton('Узнать F1 score лучшей модели')
    btn2 = types.KeyboardButton('Получить оценку размещения банкомата')
    btn3 = types.KeyboardButton('Обучить модель')
    btn4 = types.KeyboardButton('Обновить геоданные')

    markup.add(btn1, btn2, btn3, btn4)

    bot.send_message(
        message.chat.id,
        'Бот умеет производить оценку потенциального места для '
        'размещения банкомата по 5-ти балльной шкале',
        reply_markup=markup
    )


@bot.message_handler(content_types=['text'])
def bot_answer(message):

    if message.chat.type == 'private':
        if message.text == 'В начало':

            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            btn1 = types.KeyboardButton('Узнать F1 score лучшей модели')
            btn2 = types.KeyboardButton('Получить оценку размещения банкомата')
            btn3 = types.KeyboardButton('Обучить модель')
            btn4 = types.KeyboardButton('Обновить геоданные')

            markup.add(btn1, btn2, btn3, btn4)

            bot.send_message(
                message.chat.id,
                'Бот умеет производить оценку потенциального места для '
                'размещения банкомата по 5-ти балльной шкале',
                reply_markup=markup
            )

        elif message.text == 'Узнать F1 score лучшей модели':

            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            back = types.KeyboardButton('В начало')
            markup.add(back)

            filename, f1_sc = get_f1_score()

            bot.send_message(message.chat.id, f'F1 score для лучшей модели: {f1_sc}',
                             reply_markup=markup)
            bot.send_message(message.chat.id, f'Название лучшей модели: {filename}',
                    reply_markup=markup)

        elif message.text == 'Получить оценку размещения банкомата':
            
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            back = types.KeyboardButton('В начало')
            markup.add(back)

            msg = bot.send_message(message.chat.id, 'Введите координаты точки через пробел.',
                                   reply_markup=markup)
            bot.register_next_step_handler(msg, coords_get)

        elif message.text == 'Обучить модель':
            
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            btn1 = types.KeyboardButton('Задать гиперпараметры')
            btn2 = types.KeyboardButton('Использовать гиперпараметры по умолчинию')
            btn3 = types.KeyboardButton('Обновить геоданные и переобучить модель')
            back = types.KeyboardButton('В начало')
            markup.add(btn1, btn2, btn3, back)

            msg = bot.send_message(message.chat.id, 'Выберите вариант обучения модели:',
                                   reply_markup=markup)
            bot.register_next_step_handler(msg, model_train)
        
        elif message.text == 'Обновить геоданные':
            bot.send_message(message.chat.id, 'ВНИМАНИЕ!!!')
            bot.send_message(message.chat.id, 'Обновление геоданных занимает около 40 часов.')

            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            btn1 = types.KeyboardButton('Подтвердить')
            back = types.KeyboardButton('В начало')
            markup.add(btn1, back)

            msg = bot.send_message(message.chat.id, 'Подтвердите ваш выбор.',
                                   reply_markup=markup)
            bot.register_next_step_handler(msg, get_data)


def get_data(message):
    if message.text == 'В начало':

        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn1 = types.KeyboardButton('Узнать F1 score лучшей модели')
        btn2 = types.KeyboardButton('Получить оценку размещения банкомата')
        btn3 = types.KeyboardButton('Обучить модель')
        btn4 = types.KeyboardButton('Обновить геоданные')

        markup.add(btn1, btn2, btn3, btn4)

        bot.send_message(
            message.chat.id,
            'Бот умеет производить оценку потенциального места для '
            'размещения банкомата по 5-ти балльной шкале',
            reply_markup=markup
        )
    elif message.text == 'Подтвердить':
        bot.send_message(message.chat.id, 'Запускается сбор геоданых.')
        bot.send_message(message.chat.id, 'Процесс займет около 40 часов...')
        
        NOW = datetime.now().strftime("%Y-%m-%d_%H:%M")

        c = Client(None, None)
        c.trigger_dag(dag_id='get_geodata', run_id=f'from_telebot_{NOW}', conf={})
        
        get_all_futures_dataset()
        
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        back = types.KeyboardButton('В начало')
        markup.add(back)

        bot.send_message(message.chat.id, 'Сбор данных завершен.', reply_markup=markup)


def model_train(message):
    if message.text == 'В начало':

        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn1 = types.KeyboardButton('Узнать F1 score лучшей модели')
        btn2 = types.KeyboardButton('Получить оценку размещения банкомата')
        btn3 = types.KeyboardButton('Обучить модель')
        btn4 = types.KeyboardButton('Обновить геоданные')

        markup.add(btn1, btn2, btn3, btn4)

        bot.send_message(
            message.chat.id,
            'Бот умеет производить оценку потенциального места для '
            'размещения банкомата по 5-ти балльной шкале',
            reply_markup=markup
        )
    elif message.text == 'Задать гиперпараметры':
        
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        back = types.KeyboardButton('В начало')
        markup.add(back)

        msg = bot.send_message(message.chat.id, 'Введите значение Learning Rate (float от 0 до 1)',
                                reply_markup=markup)
        bot.register_next_step_handler(msg, model_train_hyper)
    
    elif message.text == 'Использовать гиперпараметры по умолчинию':
        bot.send_message(message.chat.id, 'Начинается обучение модели. Ориентировочное время обучения - 15 минут.')

        NOW = datetime.now().strftime("%Y-%m-%d_%H:%M")

        c = Client(None, None)
        c.trigger_dag(dag_id='catboost_model', run_id=f'from_telebot_{NOW}', conf={})
        
        err, model_name = train_model()
        
        bot.send_message(message.chat.id, f'F1 score текущей модели: {err}')
        bot.send_message(message.chat.id, f'Модель сохранена с названием {model_name}')

        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        back = types.KeyboardButton('В начало')
        markup.add(back)

        bot.send_message(message.chat.id, 'Вернуться в начало',
                                    reply_markup=markup)
        
    elif message.text == 'Обновить геоданные и переобучить модель':
        bot.send_message(message.chat.id, 'ВНИМАНИЕ!!!')
        bot.send_message(message.chat.id, 'Обновление геоданных занимает около 40 часов.')

        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn1 = types.KeyboardButton('Подтвердить')
        back = types.KeyboardButton('В начало')
        markup.add(btn1, back)

        msg = bot.send_message(message.chat.id, 'Подтвердите ваш выбор.',
                                reply_markup=markup)
        bot.register_next_step_handler(msg, get_data_train)


def get_data_train(message):
    if message.text == 'В начало':

        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn1 = types.KeyboardButton('Узнать F1 score лучшей модели')
        btn2 = types.KeyboardButton('Получить оценку размещения банкомата')
        btn3 = types.KeyboardButton('Обучить модель')
        btn4 = types.KeyboardButton('Обновить геоданные')

        markup.add(btn1, btn2, btn3, btn4)

        bot.send_message(
            message.chat.id,
            'Бот умеет производить оценку потенциального места для '
            'размещения банкомата по 5-ти балльной шкале',
            reply_markup=markup
        )
    elif message.text == 'Подтвердить':
        bot.send_message(message.chat.id, 'Запускается сбор геоданых для последующего обучения модели')
        bot.send_message(message.chat.id, 'Процесс займет около 40 часов...')

        NOW = datetime.now().strftime("%Y-%m-%d_%H:%M")

        c = Client(None, None)
        c.trigger_dag(dag_id='get_geodata', run_id=f'from_telebot_{NOW}', conf={})
        
        get_all_futures_dataset()
        bot.send_message(message.chat.id, 'Сбор данных завершен.')

        bot.send_message(message.chat.id, 'Начинается обучение модели. Ориентировочное время обучения - 15 минут.')

        c = Client(None, None)
        c.trigger_dag(dag_id='catboost_model', run_id=f'from_telebot_{NOW}', conf={})
        
        err, model_name = train_model()
        
        bot.send_message(message.chat.id, f'F1 score текущей модели: {err}')
        
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        back = types.KeyboardButton('В начало')
        markup.add(back)

        bot.send_message(message.chat.id, 
                         f'Модель сохранена с названием {model_name}',
                         reply_markup=markup)


def model_train_hyper(message):
    lr = message.text

    if message.text == 'В начало':

        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn1 = types.KeyboardButton('Узнать F1 score лучшей модели')
        btn2 = types.KeyboardButton('Получить оценку размещения банкомата')
        btn3 = types.KeyboardButton('Обучить модель')
        btn4 = types.KeyboardButton('Обновить геоданные')

        markup.add(btn1, btn2, btn3, btn4)

        bot.send_message(
            message.chat.id,
            'Бот умеет производить оценку потенциального места для '
            'размещения банкомата по 5-ти балльной шкале',
            reply_markup=markup
        )
    msg = bot.send_message(message.chat.id, 'Введите значение Iterations (int от 0 до 100000)')
    bot.register_next_step_handler(msg, model_train_hyper_next, lr)


def model_train_hyper_next(message, lr):
    iterations = message.text

    if message.text == 'В начало':

        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn1 = types.KeyboardButton('Узнать F1 score лучшей модели')
        btn2 = types.KeyboardButton('Получить оценку размещения банкомата')
        btn3 = types.KeyboardButton('Обучить модель')
        btn4 = types.KeyboardButton('Обновить геоданные')

        markup.add(btn1, btn2, btn3, btn4)

        bot.send_message(
            message.chat.id,
            'Бот умеет производить оценку потенциального места для '
            'размещения банкомата по 5-ти балльной шкале',
            reply_markup=markup
        )
    msg = bot.send_message(message.chat.id, 'Введите значение early stopping rounds (int от 0 до 30000)')
    bot.register_next_step_handler(msg, model_train_hyper_type, lr, iterations)


def model_train_hyper_type(message, lr, iterations):
    early_stopping_rounds = message.text
    
    if message.text == 'В начало':

        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn1 = types.KeyboardButton('Узнать F1 score лучшей модели')
        btn2 = types.KeyboardButton('Получить оценку размещения банкомата')
        btn3 = types.KeyboardButton('Обучить модель')
        btn4 = types.KeyboardButton('Обновить геоданные')

        markup.add(btn1, btn2, btn3, btn4)

        bot.send_message(
            message.chat.id,
            'Бот умеет производить оценку потенциального места для '
            'размещения банкомата по 5-ти балльной шкале',
            reply_markup=markup
        )

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = 'CPU'
    btn2 = 'GPU'
    back = types.KeyboardButton('В начало')
    markup.add(btn1, btn2, back)

    msg = bot.send_message(message.chat.id, 'Выберите тип ядра для обучения модели',
                           reply_markup=markup)
    bot.register_next_step_handler(msg, model_calc, lr, iterations, early_stopping_rounds)


def model_calc(message, lr, iterations, early_stopping_rounds):
    task_type = message.text
    
    bot.send_message(message.chat.id, 'Начинается обучение модели. Ориентировочное время обучения - 15 минут.')
    
    NOW = datetime.now().strftime("%Y-%m-%d_%H:%M")

    c = Client(None, None)
    c.trigger_dag(dag_id='catboost_model', run_id=f'from_telebot_{NOW}', conf={})
    
    err, model_name = train_model(
        learning_rate=float(lr),
        iterations=int(iterations),
        early_stopping_rounds=int(early_stopping_rounds),
        task_type=task_type
        )
    

    
    bot.send_message(message.chat.id, f'F1 score текущей модели: {err}')
    bot.send_message(message.chat.id, f'Модель сохранена с названием {model_name}')

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    back = types.KeyboardButton('В начало')
    markup.add(back)

    bot.send_message(message.chat.id, 'Вернуться в начало',
                                reply_markup=markup)


def coords_get(message):
    coords = message.text

    if message.text == 'В начало':

        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn1 = types.KeyboardButton('Узнать F1 score лучшей модели')
        btn2 = types.KeyboardButton('Получить оценку размещения банкомата')
        btn3 = types.KeyboardButton('Обучить модель')
        btn4 = types.KeyboardButton('Обновить геоданные')

        markup.add(btn1, btn2, btn3, btn4)

        bot.send_message(
            message.chat.id,
            'Бот умеет производить оценку потенциального места для '
            'размещения банкомата по 5-ти балльной шкале',
            reply_markup=markup
        )

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = 'АК БАРС'
    btn2 = 'ВТБ'
    btn3 = 'АЛЬФА БАНК'
    btn4 = 'ГАЗПРОМБАНК'
    btn5 = 'РОСБАНК'
    btn6 = 'РОССЕЛЬХОЗБАНК'
    btn7 = 'ТКБ'
    back = types.KeyboardButton('В начало')
    markup.add(btn1, btn2, btn3, btn4, btn5, btn6, btn7, back)

    msg = bot.send_message(message.chat.id, 'Выберите банк-владельца банкомата',
                           reply_markup=markup)
    bot.register_next_step_handler(msg, atm_get, coords)


def atm_get(message, coords):

    if message.text == 'В начало':

        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn1 = types.KeyboardButton('Узнать F1 score лучшей модели')
        btn2 = types.KeyboardButton('Получить оценку размещения банкомата')
        btn3 = types.KeyboardButton('Обучить модель')
        btn4 = types.KeyboardButton('Обновить геоданные')

        markup.add(btn1, btn2, btn3, btn4)

        bot.send_message(
            message.chat.id,
            'Бот умеет производить оценку потенциального места для '
            'размещения банкомата по 5-ти балльной шкале',
            reply_markup=markup
        )
    elif message.text == 'АК БАРС':
        atm_group = 1022.0
    elif message.text == 'ВТБ':
        atm_group = 5478.0
    elif message.text == 'АЛЬФА БАНК':
        atm_group = 1942.0
    elif message.text == 'ГАЗПРОМБАНК':
        atm_group = 3185.5
    elif message.text == 'РОСБАНК':
        atm_group = 8083.0
    elif message.text == 'РОССЕЛЬХОЗБАНК':
        atm_group = 496.5
    elif message.text == 'ТКБ':
        atm_group = 32.0

    coords = [coord.strip() for coord in coords.split(',')]
    
    lats, longs, atms = [], [], []

    for coord in coords:
        try:
            lats.append(float(coord.split(' ')[0].strip()))
            longs.append(float(coord.split(' ')[1].strip()))
            atms.append(atm_group)
        except:
            bot.send_message(message.chat.id, 'Введите корректный формат координат')

    bot.send_message(message.chat.id, 'Производится оценка, подождите ...')
    
    predicts = inference_model(lats=lats,longs=longs,atm=atms)

    bot.send_message(message.chat.id, 'Оценка по 5-ти балльной шкале:')

    for pred in predicts:
        bot.send_message(message.chat.id, f'{pred}')
    
  
if __name__ == "__main__":
    bot.infinity_polling()