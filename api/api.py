from datetime import datetime
from enum import Enum

from fastapi.routing import APIRouter
import requests
from bs4 import BeautifulSoup



from consillium.etl import link_parsing, save_text, save_image
from vectorize import vectorize

router = APIRouter()


#Этот эндпоинт парсит текст по ссылке и сохраняет его в директорию
@router.post('/parsing/link/')
async def parse_article(url: str):

    status = await link_parsing(url)

    if not status:
        raise ProcessLookupError

    return {'detail': {'code': 'OK', 'message': 'Текст сохранён в files/'}}


class Files(Enum):
    text = 'text'
    image = 'image'


# #Общий эндпоинт для сохранения текста и изображений
# @router.post('/content/save/')
# async def save_content(data, data_type: Files, file_name: str = 'file'):
#
#     if data_type == Files.text:
#         await save_text(data, file_name)
#         return True
#
#     if data_type == Files.image:
#         await save_image(data, file_name)
#         return True


#Эндпоинт для векторизации
@router.post("/api/vectorize/")
async def text_vectorize(file: str):
    file = "files/" + file
    return vectorize(file)