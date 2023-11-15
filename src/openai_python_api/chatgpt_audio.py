# -*- coding: utf-8 -*-
"""
Filename: chatgpt_audio.py
Author: Iliya Vereshchagin
Copyright (c) 2023. All rights reserved.

Created: 15.11.2023
Last Modified: 15.11.2023

Description:
This file contains implementation for ChatGPT audio components.
"""

from typing import Literal

from openai import AsyncOpenAI

from .models import TRANSCRIPTIONS, TRANSLATIONS


class ChatGPTAudioMixin:
    """Mixin for ChatGPT class with audio components."""

    async def transcript(
        self,
        file,
        prompt=None,
        language="en",
        response_format: Literal["text", "json", "srt", "verbose_json", "vtt"] = "text",
    ):
        """
        Wrapper for the transcribe function. Returns only the content of the message.

        :param file: Path with filename to transcript.
        :param prompt: Previous prompt. Default is None.
        :param language: Language on which audio is. Default is 'en'.
        :param response_format: default response format, by default is 'text'.
                                Possible values are: json, text, srt, verbose_json, or vtt.


        :return: transcription (text, json, srt, verbose_json or vtt)
        """
        self.___logger.debug("Transcribing file in %s with prompt '%s' to %s format", language, prompt, response_format)
        kwargs = {}
        if prompt is not None:
            kwargs["prompt"] = prompt
        response = await AsyncOpenAI.audio.transcriptions.create(
            model=TRANSCRIPTIONS[0],
            file=file,
            language=language,
            response_format=response_format,
            temperature=self.temperature,
            **kwargs,
        )
        self.___logger.debug("Transcription response: %s", response)
        return response

    async def translate(self, file, prompt=None, response_format="text"):
        """
        Wrapper for the 'translate' function. Returns only the content of the message.

        :param file: Path with filename to transcript.
        :param prompt: Previous prompt. Default is None.
        :param response_format: default response format, by default is 'text'.
                               Possible values are: json, text, srt, verbose_json, or vtt.


        :return: transcription (text, json, srt, verbose_json or vtt)
        """
        self.___logger.debug("Translating file with prompt '%s' to %s format", prompt, response_format)
        kwargs = {}
        if prompt is not None:
            kwargs["prompt"] = prompt
        response = await AsyncOpenAI.audio.translations.create(
            model=TRANSLATIONS[0],
            file=file,
            response_format=response_format,
            temperature=self.temperature,
            **kwargs,
        )
        self.___logger.debug("Translation response: %s", response)
        return response
