from datetime import datetime
from enum import Enum
import os
import requests
from bs4 import BeautifulSoup
from PIL import Image
import io


class Files(Enum):
    text = 'text'
    image = 'image'


#Функция, которая парсит текст по ссылке
async def link_parsing(url: str):
    try:
        response = requests.get(url)
        assert response.status_code == 200
    except:
        raise Exception("Failed to load page")

    soup = BeautifulSoup(response.content, 'html.parser')

    paragraphs = soup.find_all('p')
    article_text = '\n'.join([para.get_text() for para in paragraphs])

    status = await save_content(article_text, Files.text, 'article')
    return status

#Общая функция для сохранения файлов
async def save_content(data, data_type: Files, file_name: str = 'file'):

    if data_type == Files.text:
        await save_text(data, file_name)
        return True

    if data_type == Files.image:
        await save_image(data, file_name)
        return True


#Функция, которая сохраняет текст
async def save_text(text: str, file_name: str):

    file_name = datetime.now().strftime('%y.%m.%d.%f_') + file_name + '.txt'

    if not os.path.exists("files/"):
        os.makedirs("files/")

    with open(f'files/{file_name}', 'w', encoding='utf-8') as file:
        file.write(text)


#ФФункция, которая сохраняет изображение
async def save_image(image, file_name):
    image = Image.open(io.BytesIO(image))

    file_name = datetime.now().strftime('%y.%m.%d.%f_') + file_name + '.png'

    if not os.path.exists("files/"):
        os.makedirs("files/")

    image.save(f'files/{file_name}')

