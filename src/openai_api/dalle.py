# -*- coding: utf-8 -*-
"""
Filename: dalle.py
Author: Iliya Vereshchagin
Copyright (c) 2023. All rights reserved.

Created: 25.08.2023
Last Modified: 14.10.2023

Description:
This file contains implementation for DALL-E2.
"""

import asyncio
import logging
import os
import tempfile
import uuid
from io import BytesIO
from typing import Optional

import aiohttp
import openai
from PIL import Image

from .logger_config import setup_logger


class DALLE:
    """
    The DALLE class is for managing an instance of the DALLE model.

    Parameters:
    auth_token (str): Authentication bearer token. Required.
    organization (str): Organization uses auth toke. Required.
    default_count (int): Default count of images to produce. Default is 1.
    default_size (str): Default dimensions for output images. Default is "512x512". "256x256" and "1024x1024" as option.
    default_file_format (str): Default file format. Optional. Default is 'PNG'.
    user (str, optional): The user ID. Default is ''.
    logger (logging.Logger, optional): default logger. Default is None.
    """

    def __init__(
        self,
        auth_token: str,
        organization: str,
        default_count: int = 1,
        default_size: str = "512x512",
        default_file_format: str = "PNG",
        user: str = "",
        logger: Optional[logging.Logger] = None,
    ):
        """
        General init

        :param auth_token (str): Authentication bearer token. Required.
        :param organization (str): Organization uses auth toke. Required.
        :param default_count:  Default count of images to produce. Optional. Default is 1.
        :param default_size:  Default dimensions for output images. Optional. Default is "512x512".
        :param default_file_format:  Default file format. Optional. Optional. Default is 'PNG'.
        :param user: The user ID. Optional. Default is ''.
        :param logger: default logger. Optional. Default is None.
        """
        self.___logger = logger if logger is not None else setup_logger("DALLE", "dalle.log", logging.DEBUG)
        self.___logger.debug("DALLE init")
        self.___default_count = default_count
        self.___default_size = default_size
        self.___default_file_format = default_file_format
        self.___user = user
        self.___set_auth(auth_token, organization)

    def ___set_auth(self, token, organization):
        """
        Method to set auth bearer.

        :param token: authentication bearer token.
        :param organization: organization, which drives the chat.
        """
        self.___logger.debug("Setting auth bearer")
        openai.api_key = token
        openai.organization = organization

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

        :return: A PIL.Image object created from the image data received from the API.
        """
        self.___logger.debug("Creating image using prompt %s", prompt)
        response = await openai.Image.acreate(
            prompt=prompt, n=self.default_count, size=self.default_size, user=self.user
        )
        self.___logger.debug("Image created, response: %s", response["data"])
        return response["data"]

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
        self.___logger.debug("Image converted from URL %s to bytes format", url)
        return Image.open(BytesIO(image_data))

    async def create_image_data(self, prompt):
        """
        Creates an image using DALL-E Image API, returns list of images (bytes format).

        :param prompt: The prompt to be used for image creation.

        :return: list of images (bytes format).
        """
        self.___logger.debug("Creating image data using prompt %s", prompt)
        tasks = []
        async for items in await self.create_image(prompt):
            task = asyncio.ensure_future(self.convert_image_from_url_to_bytes(items["url"]))
            tasks.append(task)
        return_value = await asyncio.gather(*tasks)
        self.___logger.debug("Images created, total: %s", len(return_value))
        return return_value

    def save_image(self, image, filename=None, file_format=None):
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
            print(f"Can't save image: {error}")
            return None
        self.___logger.debug("Image saved to %s", filename)
        return filename

    async def create_variation_from_file(self, file):
        """
        Creates an image variation from file using DALL-E Image API.

        :param file: file of the image (bytes).

        :return: A PIL.Image object created from the image data received from the API.
        """
        self.___logger.debug("Creating image variation from file")
        response = await openai.Image.acreate_variation(
            image=file, n=self.default_count, size=self.default_size, user=self.user
        )
        self.___logger.debug("Image variation created from file, response: %s", response["data"])
        return response["data"]

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
        async for items in await self.create_variation_from_file(file):
            task = asyncio.ensure_future(self.convert_image_from_url_to_bytes(items["url"]))
            tasks.append(task)
        return_value = await asyncio.gather(*tasks)
        self.___logger.debug("Image variation created from file and got data, total: %s", len(return_value))
        return return_value

    async def create_variation_from_url(self, url):
        """
        Creates an image variation from URL using DALL-E Image API.

        :param url: URL of the image.

        :return: A PIL.Image object created from the image data received from the API.
        """
        self.___logger.debug("Creating image variation from URL %s", url)
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                image_data = await resp.read()

        response = await openai.Image.acreate_variation(
            image=BytesIO(image_data), n=self.default_count, size=self.default_size, user=self.user
        )
        self.___logger.debug("Image variation created from URL %s, response: %s", url, response["data"])
        return response["data"]

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
        async for items in await self.create_variation_from_url(url):
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
        :return: A PIL.Image object created from the image data received from the API.
        """
        self.___logger.debug("Editing image from file using mask and prompt '%s'", prompt)
        response = await openai.Image.acreate_edit(
            image=file, prompt=prompt, mask=mask, n=self.default_count, size=self.default_size, user=self.user
        )
        self.___logger.debug("Image edited from file, mask and prompt '%s', response: %s", prompt, response["data"])
        return response["data"]

    async def edit_image_from_url(self, url, prompt, mask_url=None):
        """
        Edits an image using OpenAI's Image API.

        :param url: A url of image to be edited.
        :param prompt: The prompt to be used for image editing.
        :param mask_url: Url containing mask image. If provided, the mask will be applied to the image.
        :return: A PIL.Image object created from the image data received from the API.
        """
        self.___logger.debug("Editing image from URL %s", url)
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                image_data = await resp.read()

        async with aiohttp.ClientSession() as session:
            async with session.get(mask_url) as resp:
                mask_data = await resp.read()
        response = await openai.Image.acreate_edit(
            image=BytesIO(image_data),
            prompt=prompt,
            mask=BytesIO(mask_data),
            n=self.default_count,
            size=self.default_size,
            user=self.user,
        )
        self.___logger.debug(
            "Image edited from URL %s using mask %s and prompt '%s', response: %s",
            url,
            mask_url,
            prompt,
            response["data"],
        )
        return response["data"]

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
        async for items in await self.edit_image_from_url(url, prompt, mask_url):
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
