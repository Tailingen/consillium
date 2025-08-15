from datetime import datetime
from enum import Enum
import time
from fastapi import Body
from fastapi.routing import APIRouter
import requests
from bs4 import BeautifulSoup
from googletrans import Translator
import textwrap


from consillium.etl import link_parsing, save_text, save_image
from consillium.generator import MedicalTextGenerator
from data_base.db_work import biobert_tokenizer, biobert_model, llm
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
@router.post("/parsing/vectorize/")
async def text_vectorize(file: str):
    file = "files/" + file
    return await vectorize(file)

@router.post("/request/")
async def request_post(req: str = Body()):
    request_eng = await consillium_translate(req)
    with MedicalTextGenerator(biobert_tokenizer, biobert_model, llm) as generator:
        print(f"Запрос: {request_eng}")
        start = time.time()
        result = generator.generate_medical_text(request_eng)
        print(f"Затрачено: {time.time() - start:.2f} сек")
        print(f"Результат:\n")

        # for line in textwrap.wrap(result, width=150):
        #     print(line)
        # print("\n" + "-" * 50 + "\n")
    result = await consillium_translate(result)
    return result

async def consillium_translate(text):
    translator = Translator()
    try:
        translated = await translator.translate(text, src='ru', dest='en')
        return translated.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text