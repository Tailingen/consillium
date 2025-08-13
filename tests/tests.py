import pytest
from unittest import mock

from consillium.etl import save_content, Files


@pytest.mark.asyncio
async def test_save_text(mocker):
    # Мокируем функцию save_text
    mock_save_text = mocker.patch('your_module.save_text', return_value=None)

    result = await save_content("Sample text", Files.text, "test.txt")

    mock_save_text.assert_called_once_with("Sample text", "test.txt")
    assert result is True

@pytest.mark.asyncio
async def test_save_image(mocker):
    # Мокируем функцию save_image
    mock_save_image = mocker.patch('your_module.save_image', return_value=None)

    result = await save_content(b"fake_image_data", Files.image, "test_image.png")

    mock_save_image.assert_called_once_with(b"fake_image_data", "test_image.png")
    assert result is True

@pytest.mark.asyncio
async def test_save_unknown_type():
    with pytest.raises(ValueError, match="Unknown data type"):
        await save_content("data", "unknown_type")  # Замените "unknown_type" на тип, который у вас не определён