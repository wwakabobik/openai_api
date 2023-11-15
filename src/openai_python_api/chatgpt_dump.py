# -*- coding: utf-8 -*-
"""
Filename: chatgpt_dump.py
Author: Iliya Vereshchagin
Copyright (c) 2023. All rights reserved.

Created: 15.11.2023
Last Modified: 15.11.2023

Description:
This file contains implementation for ChatGPT of dump functions.
"""

import json


class ChatGPTDumpMixin:
    """Mixin for ChatGPT class with dump functions."""

    async def dump_settings(self):
        """
        Dumps settings to JSON.

        :return: JSON with settings.
        """
        self.___logger.debug("Dumping settings")
        dump = json.dumps(
            {
                "model": self.model,
                "choices": self.choices,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "stream": self.stream,
                "stop": self.stop,
                "max_tokens": self.max_tokens,
                "presence_penalty": self.presence_penalty,
                "frequency_penalty": self.frequency_penalty,
                "logit_bias": self.logit_bias,
                "user": self.user,
                "functions": self.functions,
                "function_call": self.function_call,
                "function_dict": self.function_dict,
                "history_length": self.history_length,
                "prompt_method": self.prompt_method,
                "system_settings": self.system_settings,
            }
        )
        self.___logger.debug("Settings dumped: %s", dump)
        return dump

    async def dump_chats(self):
        """
        Dumps chats to JSON.

        :return: JSON with chats.
        """
        if not self.chats:
            self.___logger.error("No chats found to dump")
            return json.dumps({})
        self.___logger.debug("Dumping %s chats", len(self.chats))
        return json.dumps(self.chats)

    async def dump_chat(self, chat_name):
        """
        Dumps chat to JSON.

        :param chat_name: Name of the chat.

        :return: JSON with chat.
        """
        if chat_name not in self.chats:
            self.___logger.error("Chat %s not found to dump", chat_name)
            return json.dumps({})
        self.___logger.debug("Dumped %s records in chat %s", len(self.chats[chat_name]), chat_name)
        return json.dumps(self.chats[chat_name])
