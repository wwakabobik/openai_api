# -*- coding: utf-8 -*-
"""
Filename: chatgpt_completions.py
Author: Iliya Vereshchagin
Copyright (c) 2023. All rights reserved.

Created: 15.11.2023
Last Modified: 15.11.2023

Description:
This file contains implementation for ChatGPT completions.
"""

import json
from uuid import uuid4


class ChatGPTCompletionsMixin:
    """Mixin for ChatGPT completions."""

    async def process_chat(self, prompt, default_choice=0, chat_name=None):
        """
        Creates a new chat completion for the provided messages and parameters.

        :param prompt: The prompt to pass to the model.
        :param default_choice: Default number of choice to monitor for stream end. By default, is None.
        :param chat_name: Chat name for function tracking. Should be handled by caller. By default, is None.

        :return: Returns answers by chunk if 'stream' is false True, otherwise return complete answer.
        """
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-statements
        uid = str(uuid4())
        self.___logger.debug(
            "Processing chat '%s' with prompt '%s', tracking choice %s with uid=%s",
            chat_name,
            prompt,
            default_choice,
            uid,
        )
        # Prepare parameters
        params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "n": self.choices,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "user": self.user,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "stream": self.stream,
            "response_format": self.___response_format,
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        # Add 'prompt' or 'messages' parameter
        if self.prompt_method:
            params["messages"] = [{"role": "system", "content": prompt}]
        else:
            params["messages"] = prompt

        # Get response
        func_response = None
        func_call = {}
        if self.stream:
            try:
                async for chunk in await self.___engine.chat.completions.create(**params):
                    chunk = json.loads(chunk.model_dump_json())
                    if "function_call" in chunk["choices"][default_choice]["delta"]:
                        raw_call = chunk["choices"][default_choice]["delta"]["function_call"]  # noqa: WPS529
                        for key, value in raw_call.items():
                            if key in func_call and isinstance(value, str):
                                func_call[key] += value
                            elif key in func_call and isinstance(func_call[key], dict) and isinstance(value, dict):
                                func_call[key].update(value)
                            else:
                                func_call[key] = value
                    if chunk["choices"][default_choice]["finish_reason"] is not None:
                        if chunk["choices"][default_choice]["finish_reason"] == "function_call":
                            func_response = await self.process_function(function_call=func_call)
                        break
                    if "content" in chunk["choices"][default_choice]["delta"]:
                        if chunk["choices"][default_choice]["delta"]["content"]:  # noqa: WPS529
                            yield chunk
            except GeneratorExit:
                self.___logger.debug("Chat ended with uid=%s", uid)
            except Exception as error:  # pylint: disable=broad-except
                self.___logger.error("Error while processing chat: %s", error)
            try:
                if func_response:
                    # Save to history
                    if chat_name:
                        self.chats[chat_name].append(func_response)
                    # Add new prompt
                    if self.prompt_method:
                        params.pop("prompt", None)
                        params["messages"] = []
                        params["messages"].append(func_response)
                    else:
                        params["messages"].append(func_response)
                    async for func_chunk in await self.___engine.chat.completions.create(**params):
                        yield func_chunk
                        if func_chunk["choices"][default_choice]["finish_reason"] is not None:
                            break
            except GeneratorExit:
                self.___logger.debug("Chat ended with uid=%s", uid)
        else:
            response = await self.___engine.chat.completions.create(**params)
            response = json.loads(response.model_dump_json())
            try:
                if response["choices"][default_choice]["finish_reason"] == "function_call":
                    func_response = await self.process_function(
                        function_call=response["choices"][default_choice]["message"]["function_call"]
                    )
                if func_response:
                    # Save to history
                    if chat_name:
                        self.chats[chat_name].append(func_response)
                    # Add new prompt
                    if self.prompt_method:
                        params.pop("prompt", None)
                        params["messages"] = []
                        params["messages"].append(func_response)
                    else:
                        params["messages"].append(func_response)
                    response = await self.___engine.chat.completions.create(**params)
                    yield response
                else:
                    yield response
            except GeneratorExit:
                self.___logger.debug("Chat ended with uid=%s", uid)
            except Exception as error:  # pylint: disable=W0718
                self.___logger.error("Error while processing chat: %s", error)

    async def chat(self, prompt, chat_name=None, default_choice=0, extra_settings=""):
        """
        Wrapper for the process_chat function. Adds new messages to the chat and calls process_chat.

        :param prompt: Message from the user.
        :param chat_name: Name of the chat. If None, uses self.current_chat.
        :param default_choice: Index of the model's response choice.
        :param extra_settings: Extra system settings for chat. Default is ''.
        """
        # pylint: disable=too-many-branches
        uid = str(uuid4())
        self.___logger.debug(
            "Chatting with prompt '%s' in chat '%s', tracking choice %s, with extra settings = '%s', uid=%s",
            prompt,
            chat_name,
            default_choice,
            extra_settings,
            uid,
        )
        # Call process_chat
        full_prompt = ""
        if self.prompt_method:
            try:
                async for response in self.process_chat(prompt=prompt, default_choice=default_choice):
                    if isinstance(response, dict):
                        finish_reason = response["choices"][default_choice].get("finish_reason", "")
                        yield response
                        if self.stream:
                            if "content" in response["choices"][default_choice]["delta"].keys():
                                full_prompt += response["choices"][default_choice]["delta"]["content"]
                        else:
                            full_prompt += response["choices"][default_choice]["message"]["content"]
                        if finish_reason is not None:
                            yield ""
                    else:
                        break
            except GeneratorExit:
                pass
        else:
            # Set chat_name
            chat_name = chat_name if chat_name is not None else self.current_chat
            chat_name = await self.__handle_chat_name(chat_name, prompt)
            # Add new message to chat
            self.chats[chat_name].append({"role": "user", "content": prompt})
            # Get last 'history_length' messages
            messages = self.chats[chat_name][-self.history_length :]  # noqa: E203
            messages.insert(0, {"role": "system", "content": f"{self.system_settings} {extra_settings}"})

            try:
                async for prompt_response in self.process_chat(  # noqa: WPS352
                    prompt=messages, default_choice=default_choice, chat_name=chat_name
                ):
                    if isinstance(prompt_response, dict):
                        finish_reason = prompt_response["choices"][default_choice].get("finish_reason", "")
                        yield prompt_response
                        if self.stream:
                            if "content" in prompt_response["choices"][default_choice]["delta"].keys():
                                full_prompt += prompt_response["choices"][default_choice]["delta"]["content"]
                        else:
                            full_prompt += prompt_response["choices"][default_choice]["message"]["content"]
                        if finish_reason is not None:
                            yield ""
                    else:
                        break
            except GeneratorExit:
                self.___logger.debug("Chat ended with uid=%s", uid)

            # Add last response to chat
            record = {"role": "assistant", "content": full_prompt}
            self.chats[chat_name].append(record)
            self.___logger.debug("Record added to chat '%s': %s", chat_name, record)

    async def str_chat(self, prompt, chat_name=None, default_choice=0, extra_settings=""):
        """
        Wrapper for the chat function. Returns only the content of the message.

        :param prompt: Message from the user.
        :param chat_name: Name of the chat. If None, uses self.current_chat.
        :param default_choice: Index of the model's response choice.
        :param extra_settings: Extra system settings for chat. Default is ''.

        :return: Content of the message.
        """
        uid = str(uuid4())
        self.___logger.debug(
            "Chatting str with prompt '%s' in chat '%s', tracking choice %s, with extra settings = '%s', uid=%s",
            prompt,
            chat_name,
            default_choice,
            extra_settings,
            uid,
        )
        try:
            async for response in self.chat(prompt, chat_name, default_choice, extra_settings=extra_settings):
                if isinstance(response, dict):
                    if self.stream:
                        if "content" in response["choices"][default_choice]["delta"].keys():
                            yield response["choices"][default_choice]["delta"]["content"]
                        else:
                            yield ""
                    else:
                        yield response["choices"][default_choice]["message"]["content"]
                else:
                    break
        except GeneratorExit:
            self.___logger.debug("String chat ended with uid=%s", uid)

    async def process_function(self, function_call):
        """
        Process function requested by ChatGPT.

        :param function_call: Function name and arguments. In JSON format.

        :return: transcription (text, json, srt, verbose_json or vtt)
        """
        self.___logger.debug("Processing function call: %s", function_call)
        function_name = function_call["name"]
        function_to_call = self.function_dict[function_name]
        function_response = function_to_call(**json.loads(function_call["arguments"]))
        return_value = {"role": "function", "name": function_name, "content": function_response}
        self.___logger.debug("Function response: %s", return_value)
        return return_value

    async def __handle_chat_name(self, chat_name, prompt):
        """
        Handles the chat name. If chat_name is None, sets it to the first chat_name_length characters of the prompt.
        If chat_name is not present in self.chats, adds it.

        :param chat_name: Name of the chat.
        :param prompt: Message from the user.
        :return: Processed chat name.
        """
        if chat_name is None:
            chat_name = prompt[: self.___chat_name_length]
            self.current_chat = chat_name
            self.___logger.debug(
                "Chat name is None, setting it to the first %s characters of the prompt: %s",
                self.___chat_name_length,
                chat_name,
            )
        elif len(chat_name) > self.___chat_name_length:
            self.___logger.debug(
                "Chat name is longer than %s characters, truncating it: %s", self.___chat_name_length, chat_name
            )
            chat_name = chat_name[: self.___chat_name_length]
        if chat_name not in self.chats:
            self.___logger.debug("Chat name '%s' is not present in self.chats, adding it", chat_name)
            self.chats[chat_name] = []
        return chat_name
