# -*- coding: utf-8 -*-
"""
Filename: dalle.py
Author: Iliya Vereshchagin
Copyright (c) 2023. All rights reserved.

Created: 25.08.2023
Last Modified: 10.11.2023

Description:
This file contains implementation for DALL-E2.
"""

import asyncio
import json
import logging
import os
import tempfile
import uuid
from io import BytesIO
from typing import Optional, Literal

import aiohttp
from PIL import Image
from openai import AsyncOpenAI

from .logger_config import setup_logger
from .models import DALLE_MODELS


# pylint: disable=too-many-instance-attributes,too-many-public-methods
class DALLE:
    """The DALLE class is for managing an instance of the DALLE model."""

    def __init__(
        self,
        auth_token: Optional[str] = None,
        organization: Optional[str] = None,
        default_count: int = 1,
        default_size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"] = "1024x1024",
        default_file_format: str = "PNG",
        user: str = "",
        model: str = DALLE_MODELS[0],
        quality: Literal["hd", "standard"] = "hd",
        style: Literal["vivid", "natural"] = "natural",
        logger: Optional[logging.Logger] = None,
    ):
        """
        General init

        :param auth_token (str): Authentication bearer token. If None, will be taken from env OPENAI_API_KEY.
        :param organization (str): Organization uses auth token. If None, will be taken from env OPENAI_ORGANIZATION.
        :param default_count:  Default count of images to produce. Optional. Default is 1.
        :param default_size:  Default dimensions for output images. Optional. Default is "512x512".
        :param default_file_format:  Default file format. Optional. Optional. Default is 'PNG'.
        :param user: The user ID. Optional. Default is ''.
        :param model: The model ID. Optional. Default is 'dall-e-3'.
        :param quality: The quality of the image. Optional. Default is 'hd'. Possible values: 'hd', 'standard'.
        :param style: The style of the image. Optional. Default is 'natural'. Possible values: 'vivid', 'natural'.
        :param logger: default logger. Optional. Default is None.
        :raises: ValueError: If invalid settings provided.

        Raises:
            ValueError: If invalid settings provided.
        """
        self.___logger = logger if logger is not None else setup_logger("DALLE", "dalle.log", logging.DEBUG)
        self.___logger.debug("DALLE init")
        self.___default_count = default_count
        self.___default_size = default_size
        self.___default_file_format = default_file_format
        self.___default_model = model
        self.___default_quality = quality
        self.___default_style = style
        self.___user = user
        auth_token = auth_token if auth_token is not None else os.environ.get("OPENAI_API_KEY")
        organization = organization if organization is not None else os.environ.get("OPENAI_ORGANIZATION")
        self.___engine = AsyncOpenAI(api_key=auth_token, organization=organization)
        # Validate settings
        if not self.___validate_image_size() or not self.___validate_model():
            raise ValueError("Invalid settings provided!")

    @property
    def default_count(self):
        """
        Getter for default_count.

        :return: Returns default_count value.
        """
        self.___logger.debug("Getting default_count %s", self.___default_count)
        return self.___default_count

    @default_count.setter
    def default_count(self, value):
        """
        Setter for default_count.

        :param value: The new value of default_count.
        """
        self.___logger.debug("Setting default_count %s", value)
        self.___default_count = value

    @property
    def default_size(self):
        """
        Getter for default_size.

        :return: Returns default_size value.
        """
        self.___logger.debug("Getting default_size %s", self.___default_size)
        return self.___default_size

    @default_size.setter
    def default_size(self, value):
        """
        Setter for default_size.

        :param value: The new value of  default_size.
        """
        self.___logger.debug("Setting default_size %s", value)
        if self.___validate_image_size():
            self.___default_size = value

    @property
    def default_file_format(self):
        """
        Getter for default_file_format.

        :return: Returns default_file_format value.
        """
        self.___logger.debug("Getting default_file_format %s", self.___default_file_format)
        return self.___default_file_format

    @default_file_format.setter
    def default_file_format(self, value):
        """
        Setter for default_size.

        :param value: The new value of default_file_format.
        """
        self.___logger.debug("Setting default_file_format %s", value)
        self.___default_file_format = value

    @property
    def default_style(self):
        """
        Getter for default_style.

        :return: Returns default_style value.
        """
        self.___logger.debug("Getting default_style %s", self.___default_style)
        return self.___default_style

    @default_style.setter
    def default_style(self, value):
        """
        Setter for default_size.

        :param value: The new value of default_file_format.
        """
        self.___logger.debug("Setting default_style %s", value)
        self.___default_style = value

    @property
    def user(self):
        """
        Getter for user.

        :return: The user.
        """
        self.___logger.debug("Getting user %s", self.___user)
        return self.___user

    @user.setter
    def user(self, value):
        """
        Setter for user.

        :param value: The user.
        """
        self.___logger.debug("Setting user %s", value)
        self.___user = value

    @property
    def default_quality(self):
        """
        Getter for default quality.

        :return: The default quality.
        """
        self.___logger.debug("Getting default quality %s", self.___default_quality)
        return self.___default_quality

    @default_quality.setter
    def default_quality(self, value):
        """
        Setter for default quality.

        :param value: The user.
        """
        self.___logger.debug("Setting default quality %s", value)
        self.___default_quality = value

    @property
    def default_model(self):
        """
        Getter for default model.

        :return: The default model.
        """
        self.___logger.debug("Getting default model %s", self.___default_model)
        return self.___default_model

    @default_model.setter
    def default_model(self, value):
        """
        Setter for default model.

        :param value: The user.
        """
        self.___logger.debug("Setting default model %s", value)
        if self.___validate_model():
            self.___default_model = value

    @property
    def logger(self):
        """
        Getter for logger.

        :return: The logger object.
        """
        self.___logger.debug("Getting logger...")
        return self.___logger

    @logger.setter
    def logger(self, value):
        """
        Setter for logger.

        :param value: The new logger object.
        """
        self.___logger.debug("Setting logger...")
        self.___logger = value

    async def create_image(self, prompt):
        """
        Creates an image using DALL-E Image API.

        :param prompt: The prompt to be used for image creation.

        :return: A data dict created from the image data received from the API.
        """
        self.___logger.debug("Creating image using prompt: '%s'", prompt)
        response = await self.___engine.images.generate(
            prompt=prompt,
            n=self.default_count,
            size=self.default_size,
            user=self.user,
            model=self.default_model,
            quality=self.default_quality,
            style=self.default_style,
        )
        try:
            data = json.loads(response.model_dump_json())["data"]
            self.___logger.debug("Image created, response: %s", data)
        except Exception as error:  # pylint: disable=W0718
            self.___logger.error("Can't parse response: %s", error)
            return None
        return data

    async def create_image_url(self, prompt):
        """
        Creates an image using DALL-E Image API, returns list of URLs with images.

        :param prompt: The prompt to be used for image creation.

        :return: list of URLs
        """
        self.___logger.debug("Creating image URLs using prompt %s", prompt)
        image_urls = []
        for items in await self.create_image(prompt):
            image_urls.append(items["url"])
        self.___logger.debug("Image URLs created, response: %s", image_urls)
        return image_urls

    async def convert_image_from_url_to_bytes(self, url):
        """
        Converts image from URL to bytes format.

        :param url: URL of image.

        :return: URL
        """
        self.___logger.debug("Converting image from URL %s to bytes format", url)
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                image_data = await resp.read()
                image_data = self.___convert_to_rgba(image_data)
        self.___logger.debug("Image converted from URL %s to bytes format", url)
        return Image.open(BytesIO(await image_data))

    async def create_image_data(self, prompt):
        """
        Creates an image using DALL-E Image API, returns list of images (bytes format).

        :param prompt: The prompt to be used for image creation.

        :return: list of images (bytes format).
        """
        self.___logger.debug("Creating image data using prompt %s", prompt)
        tasks = []
        for items in await self.create_image(prompt):
            task = asyncio.ensure_future(self.convert_image_from_url_to_bytes(items["url"]))
            tasks.append(task)
        return_value = await asyncio.gather(*tasks)
        self.___logger.debug("Images created, total: %s", len(return_value))
        return return_value

    async def save_image(self, image, filename=None, file_format=None):
        """Saves an image to a file.

        :param image: A PIL.Image object to be saved.
        :param filename: The name of the file where the image will be saved.
                         If None, a random filename in the system's temporary directory will be used.
        :param file_format: The format of the file. This is optional and defaults to 'PNG'.

        :return: The full path of the file where the image was saved, or None if the image could not be saved.
        """
        self.___logger.debug("Saving image %s", image)
        if file_format is None:
            file_format = self.default_file_format
        if filename is None:
            filename = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.{file_format.lower()}")
        try:
            image.save(filename)
        except Exception as error:  # pylint: disable=W0718
            self.___logger.error("Can't save image: %s", error)
            return None
        self.___logger.debug("Image saved to %s", filename)
        return filename

    async def create_variation_from_file(self, file):
        """
        Creates an image variation from file using DALL-E Image API.

        :param file: file of the image (bytes).

        :return: A data dict object created from the image data received from the API.
        """
        self.___logger.debug("Creating image variation from file")
        response = await self.___engine.images.create_variation(
            image=file, n=self.default_count, size=self.default_size, user=self.user, model=self.default_model
        )
        try:
            data = json.loads(response.model_dump_json())["data"]
            self.___logger.debug("Image variation created from file, response: %s", data)
        except Exception as error:  # pylint: disable=W0718
            self.___logger.error("Can't parse response: %s", error)
            return None
        return data

    async def create_variation_from_file_and_get_url(self, file):
        """
        Creates an image variation from file using DALL-E Image API, returns list of URLs with images.

        :param file: file of the image (bytes).

        :return: list of URLs
        """
        self.___logger.debug("Creating image variation from file and getting URLs")
        image_urls = []
        for items in await self.create_variation_from_file(file):
            image_urls.append(items["url"])
        self.___logger.debug("Image variation created from file and got URLs, response: %s", image_urls)
        return image_urls

    async def create_variation_from_file_and_get_data(self, file):
        """
        Creates an image variation from file using DALL-E Image API, returns list of images (bytes format).

        :param file: file of the image (bytes).

        :return: list of images (bytes format).
        """
        self.___logger.debug("Creating image variation from file and getting data")
        tasks = []
        for items in await self.create_variation_from_file(file):
            task = asyncio.ensure_future(self.convert_image_from_url_to_bytes(items["url"]))
            tasks.append(task)
        return_value = await asyncio.gather(*tasks)
        self.___logger.debug("Image variation created from file and got data, total: %s", len(return_value))
        return return_value

    async def create_variation_from_url(self, url):
        """
        Creates an image variation from URL using DALL-E Image API.

        :param url: URL of the image.

        :return: A data dict object created from the image data received from the API.
        """
        self.___logger.debug("Creating image variation from URL %s", url)
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                image_data = await resp.read()
                image_data = await self.___convert_to_rgba(image_data)

        response = await self.___engine.images.create_variation(
            image=BytesIO(image_data),
            n=self.default_count,
            size=self.default_size,
            user=self.user,
            model=self.default_model,
        )
        try:
            data = json.loads(response.model_dump_json())["data"]
            self.___logger.debug("Image variation created from URL %s, response: %s", url, data)
        except Exception as error:  # pylint: disable=W0718
            self.___logger.error("Can't parse response: %s", error)
            return None
        return data

    async def create_variation_from_url_and_get_url(self, url):
        """
        Creates an image variation from URL using DALL-E Image API, returns list of URLs with images.

        :param url: URL of the image.

        :return: list of URLs
        """
        self.___logger.debug("Creating image variation from URL %s and getting URLs", url)
        image_urls = []
        for items in await self.create_variation_from_url(url):
            image_urls.append(items["url"])
        self.___logger.debug("Image variation created from URL %s and got URLs, response: %s", url, image_urls)
        return image_urls

    async def create_variation_from_url_and_get_data(self, url):
        """
        Creates an image variation from URL using DALL-E Image API, returns list of images (bytes format).

        :param url: URL of the image.

        :return: list of images (bytes format).
        """
        self.___logger.debug("Creating image variation from URL %s and getting data", url)
        tasks = []
        for items in await self.create_variation_from_url(url):
            task = asyncio.ensure_future(self.convert_image_from_url_to_bytes(items["url"]))
            tasks.append(task)
        return_value = await asyncio.gather(*tasks)
        self.___logger.debug("Image variation created from URL %s and got data, total: %s", url, len(return_value))
        return return_value

    async def edit_image_from_file(self, file, prompt, mask=None):
        """
        Edits an image using OpenAI's Image API.

        :param file: A file-like object opened in binary mode containing the image to be edited.
        :param prompt: The prompt to be used for image editing.
        :param mask: An optional file-like object opened in binary mode containing the mask image.
                     If provided, the mask will be applied to the image.
        :return: A data dict created from the image data received from the API.
        """
        self.___logger.debug("Editing image from file using mask and prompt '%s'", prompt)
        response = await self.___engine.images.edit(
            image=file,
            prompt=prompt,
            mask=mask,
            n=self.default_count,
            size=self.default_size,
            user=self.user,
            model=self.default_model,
        )
        try:
            data = json.loads(response.model_dump_json())["data"]
            self.___logger.debug("Image edited from file, mask and prompt '%s', response: %s", prompt, data)
        except Exception as error:  # pylint: disable=W0718
            self.___logger.error("Can't parse response: %s", error)
            return None
        return data

    async def edit_image_from_url(self, url, prompt, mask_url=None):
        """
        Edits an image using OpenAI's Image API.

        :param url: A url of image to be edited.
        :param prompt: The prompt to be used for image editing.
        :param mask_url: Url containing mask image. If provided, the mask will be applied to the image.
        :return: A data dict created from the image data received from the API.
        """
        self.___logger.debug("Editing image from URL %s", url)
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                image_data = await resp.read()
                image_data = await self.___convert_to_rgba(image_data)

        async with aiohttp.ClientSession() as mark_session:
            async with mark_session.get(mask_url) as mark_resp:
                mask_data = await mark_resp.read()
                mask_data = BytesIO(mask_data)
                mask_data = await self.___convert_to_rgba(mask_data)

        response = await self.___engine.images.edit(
            image=BytesIO(image_data),
            prompt=prompt,
            mask=mask_data,
            n=self.default_count,
            size=self.default_size,
            user=self.user,
            model=self.default_model,
        )
        try:
            data = json.loads(response.model_dump_json())["data"]
            self.___logger.debug(
                "Image edited from URL %s using mask %s and prompt '%s', response: %s",
                url,
                mask_url,
                prompt,
                data,
            )
        except Exception as error:  # pylint: disable=W0718
            self.___logger.error("Can't parse response: %s", error)
            return None
        return data

    async def edit_image_from_url_and_get_url(self, url, prompt, mask_url=None):
        """
        Edits an image using OpenAI's Image API, returns list of URLs with images.

        :param url: A url of image to be edited.
        :param prompt: The prompt to be used for image editing.
        :param mask_url: Url containing mask image. If provided, the mask will be applied to the image.
        :return: list of URLs
        """
        self.___logger.debug("Editing image from URL %s and getting URLs", url)
        image_urls = []
        for items in await self.edit_image_from_url(url, prompt, mask_url):
            image_urls.append(items["url"])
        self.___logger.debug(
            "Image edited from URL %s using mask %s and prompt '%s', response: %s", url, mask_url, prompt, image_urls
        )
        return image_urls

    async def edit_image_from_url_and_get_data(self, url, prompt, mask_url=None):
        """
        Edits an image using OpenAI's Image API, returns list of images (bytes format).

        :param url: A url of image to be edited.
        :param prompt: The prompt to be used for image editing.
        :param mask_url: Url containing mask image. If provided, the mask will be applied to the image.
        :return: list of images (bytes format).
        """
        self.___logger.debug("Editing image from URL %s and getting data", url)
        tasks = []
        for items in await self.edit_image_from_url(url, prompt, mask_url):
            task = asyncio.ensure_future(self.convert_image_from_url_to_bytes(items["url"]))
            tasks.append(task)
        return_value = await asyncio.gather(*tasks)
        self.___logger.debug(
            "Image edited from URL %s using mask %s and prompt '%s', response: %s",
            url,
            mask_url,
            prompt,
            len(return_value),
        )
        return return_value

    async def ___convert_to_rgba(self, image_data):
        """
        Converts image to RGBA format.

        :param image_data: image to convert.

        :return: image in RGBA format.
        """
        self.___logger.debug("Converting image to RGBA format")
        image = Image.open(BytesIO(image_data))
        image = image.convert("RGBA")
        image_data = BytesIO()
        image.save(image_data, format="PNG")
        return image_data.getvalue()

    def ___validate_image_size(self):
        """
        Validate image size.

        :return: True if image size is valid, False otherwise.
        """
        self.___logger.debug("Validating image size...")
        if (self.default_size not in {"256x256", "512x512", "1024x1024"} and self.default_model == DALLE_MODELS[1]) or (
            self.___default_size not in {"1024x1024", "1792x1024", "1024x1792"}
            and self.default_model == DALLE_MODELS[0]
        ):
            self.___logger.error("Image size is invalid!")
            return False
        return True

    def ___validate_model(self):
        """
        Validate model.

        :return: True if model is valid, False otherwise.
        """
        self.___logger.debug("Validating model...")
        if self.default_model not in DALLE_MODELS:
            self.___logger.error("Model is invalid!")
            return False
        return True
