
import numpy as np
from fastapi import FastAPI
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from PIL import Image
import io
from enum import Enum

from api.api import router

app = FastAPI()
app.include_router(router)

class Files(Enum):
    text = 'text'
    image = 'image'



#Тестовые функции для изображений

#
# @app.post('/test/image/save/')
# async def test_image_parsing(file_name: str = 'foto'):
#     #image = Image.new('RGB', (100, 100), color='red')
#     # image = requests.get('https://tatarstan.ru/file/news/1221_n2075376_big.jpg')
#     # image = BeautifulSoup(image.content)
#     image = download_random_price_image()
#     image_byte_array = io.BytesIO()
#     image.save(image_byte_array, format='PNG')
#     image_byte_array.seek(0)
#     await save_content(image_byte_array.getvalue(), Files.image, file_name)
#     return {'detail': {'code': 'OK', 'message': 'Фото сохранено в files/'}}
#
#
# def download_random_price_image():
#     # URL для получения случайного изображения с ценником
#     image_url = "https://dummyimage.com/400x300/000/fff&text=%2410.00"
#     image_url = 'https://tatarstan.ru/file/news/1221_n2075376_big.jpg'
#     # Пример генератора изображений
#     try:
#         # Выполняем GET-запрос к изображению
#         response = requests.get(image_url)
#         response.raise_for_status()  # Проверка успешности запроса
#
#         # Открываем изображение
#         image = Image.open(io.BytesIO(response.content))
#
#         # Сохраняем изображение
#         image.save("random_price_image.png")
#         print("Изображение ценника успешно загружено и сохранено как random_price_image.png")
#         return image
#
#     except Exception as e:
#         print(f"Произошла ошибка при загрузке изображения: {e}")

