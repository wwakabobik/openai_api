# -*- coding: utf-8 -*-
"""
Filename: models.py
Author: Iliya Vereshchagin
Copyright (c) 2023. All rights reserved.

Created: 25.08.2023
Last Modified: 10.11.2023

Description:
This file contains OpenAI constants
"""

COMPLETIONS = (
    # GPT-4
    "gpt-4",
    "gpt-4-1106-preview",
    "gpt-4-vision-preview",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0613",
    # GPT-3
    "gpt-3.5-turbo-1106" "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    # DEPRECATION WARNING: GPT-3
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k-0613",
    "gpt-3.5-turbo-0301",
    # Other
    "gpt-3.5-turbo-instruct",
    "text-davinci-003",
    "text-davinci-002",
    "code-davinci-002",
)
DALLE_MODELS = ("dall-e-3", "dall-e-2")
TRANSCRIPTIONS = ("whisper-1",)
TRANSLATIONS = ("whisper-1",)
FINE_TUNES = ("davinci", "curie", "babbage", "ada")
EMBEDDINGS = ("text-embedding-ada-002", "text-similarity-*-001", "text-search-*-*-001", "code-search-*-*-001")
MODERATIONS = ("text-moderation-stable", "text-moderation-latest")
