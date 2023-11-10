# -*- coding: utf-8 -*-
"""
Filename: chatgpt.py
Author: Iliya Vereshchagin
Copyright (c) 2023. All rights reserved.

Created: 25.08.2023
Last Modified: 10.11.2023

Description:
This file contains implementation for ChatGPT.
"""

import json
import logging
from os import environ
from typing import Optional, Literal
from uuid import uuid4

from openai import AsyncOpenAI

from .gpt_statistics import GPTStatistics
from .logger_config import setup_logger
from .models import COMPLETIONS, TRANSCRIPTIONS, TRANSLATIONS


# pylint: disable=too-many-instance-attributes,too-many-public-methods
class ChatGPT:
    """
    The ChatGPT class is for managing an instance of the ChatGPT model.

    Parameters:
    auth_token (str): Authentication bearer token.
    organization (str): Organization uses auth toke. Required.
    model (str): The name of the model, Default is 'gpt-4'.
    choices (int, optional): The number of response options. Default is 1.
    temperature (float, optional): The temperature of the model's output. Default is 1.
    top_p (float, optional): The top-p value for nucleus sampling. Default is 1.
    stream (bool, optional): If True, the model will return intermediate results. Default is False.
    stop (str, optional): The stop sequence at which the model should stop generating further tokens. Default is None.
    max_tokens (int, optional): The maximum number of tokens in the output. Default is 1024.
    presence_penalty (float, optional): The penalty for new token presence. Default is 0.
    frequency_penalty (float, optional): The penalty for token frequency. Default is 0.
    logit_bias (map, optional): The bias for the logits before sampling. Default is None.
    user (str, optional): The user ID. Default is ''.
    functions (list, optional): The list of functions. Default is None.  # DEPRECATED
    function_call (str, optional): The function call. Default is None.   # DEPRECATED
    function_dict (dict, optional): Dict of functions. Default is None.  # DEPRECATED
    tools (list, optional): The list of tools. Default is None.
    tool_choice (str, optional): The tool call. Default is None.
    tools_dict (str, optional): Dict of tools. Default is None.
    history_length (int, optional): Length of history. Default is 5.
    chats (dict, optional): Chats dictionary, contains all chats. Default is None.
    current_chat (str, optional): Default chat will be used. Default is None.
    prompt_method (bool, optional): prompt method. Use messages if False, otherwise - prompt. Default if False.
    logger (logging.Logger, optional): default logger. Default is None.
    statistic (GPTStatistics, optional): statistics logger. If none, will be initialized with zeros.
    response_format (str, optional): response format. Default is None. Or might be { "type": "json_object" }.
    system_settings (str, optional): general instructions for chat. Default is None.
    """

    def __init__(
        # pylint: disable=too-many-locals
        self,
        auth_token: Optional[str],
        organization: Optional[str],
        model: str = COMPLETIONS[0],
        choices: int = 1,
        temperature: float = 1,
        top_p: float = 1,
        stream: bool = False,
        stop: Optional[str] = None,
        max_tokens: int = 1024,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        logit_bias: Optional[map] = None,
        user: str = "",
        functions: Optional[list] = None,  # deprecated
        function_call: Optional[str] = None,  # deprecated
        function_dict: Optional[dict] = None,  # deprecated
        tools: Optional[list] = None,
        tool_choice: Optional[str] = None,
        tools_dict: Optional[dict] = None,
        history_length: int = 5,
        chats: Optional[dict] = None,
        current_chat: Optional[str] = None,
        chat_name_length: int = 40,
        prompt_method: bool = False,
        logger: Optional[logging.Logger] = None,
        statistics: Optional[GPTStatistics] = None,
        system_settings: Optional[str] = None,
        response_format: Optional[str] = None,
    ):
        """
        General init

        :param auth_token (str): Authentication bearer token. If None, will be taken from env OPENAI_API_KEY.
        :param organization (str): Organization uses auth toke. If None, will be taken from env OPENAI_ORGANIZATION.
        :param model: The name of the model.
        :param choices: The number of response options. Default is 1.
        :param temperature: The temperature of the model's output. Default is 1.
        :param top_p: The top-p value for nucleus sampling. Default is 1.
        :param stream: If True, the model will return intermediate results. Default is False.
        :param stop: The stop sequence at which the model should stop generating further tokens. Default is None.
        :param max_tokens: The maximum number of tokens in the output. Default is 1024.
        :param presence_penalty: The penalty for new token presence. Default is 0.
        :param frequency_penalty: The penalty for token frequency. Default is 0.
        :param logit_bias: The bias for the logits before sampling. Default is None.
        :param user: The user ID. Default is ''.
        :param functions: The list of functions. Default is None.  # DEPRECATED
        :param function_call: The function call. Default is None.  # DEPRECATED
        :param function_dict: Dict of functions. Default is None.  # DEPRECATED
        :param tools: The list of tools. Default is None.
        :param tool_choice: The tool call. Default is None.
        :param tools_dict: Dict of tools. Default is None.
        :param history_length: Length of history. Default is 5.
        :param chats: Chats dictionary, contains all chat. Default is None.
        :param current_chat: Default chat will be used. Default is None.
        :param chat_name_length: Length of chat name. Default is 40.
        :param prompt_method: prompt method. Use messages if False, otherwise - prompt. Default if False.
        :param logger: default logger. Default is None.
        :param statistics: statistics logger. If none, will be initialized with zeros.
        :param response_format: response format. Default is 'text', 'json'.
        :param system_settings: general system instructions for bot. Default is ''.
        """
        self.___logger = logger if logger else setup_logger("ChatGPT", "chatgpt.log", logging.DEBUG)
        self.___logger.debug("Initializing ChatGPT")
        self.___model = model
        self.___choices = choices
        self.___temperature = temperature
        self.___top_p = top_p
        self.___stream = stream
        self.___stop = stop
        self.___max_tokens = max_tokens
        self.___presence_penalty = presence_penalty
        self.___frequency_penalty = frequency_penalty
        self.___logit_bias = logit_bias
        self.___user = user
        self.___functions = functions  # deprecated
        self.___function_call = function_call  # deprecated
        self.___function_dict = function_dict  # deprecated
        self.___tools = tools
        self.___tool_choice = tool_choice
        self.___tools_dict = tools_dict
        self.___history_length = history_length
        self.___chats = chats if chats else {}
        self.___current_chat = current_chat
        self.___chat_name_length = chat_name_length
        self.___prompt_method = prompt_method
        self.___statistics = statistics if statistics else GPTStatistics()  # pylint: disable=W0238
        self.___response_format = response_format
        self.___system_settings = system_settings if system_settings else ""
        auth_token = auth_token if auth_token is not None else environ.get("OPENAI_API_KEY")
        organization = organization if organization is not None else environ.get("OPENAI_ORGANIZATION")
        self.___engine = AsyncOpenAI(api_key=auth_token, organization=organization)

    @property
    def model(self):
        """
        Getter for model.

        :return: The name of the model.
        """
        self.___logger.debug("Getting model %s", self.___model)
        return self.___model

    @model.setter
    def model(self, value):
        """
        Setter for model.

        :param value: The new name of the model.
        """
        self.___logger.debug("Setting model %s", value)
        self.___model = value

    @property
    def choices(self):
        """
        Getter for choices.

        :return: The number of response options.
        """
        self.___logger.debug("Getting choices %s", self.___choices)
        return self.___choices

    @choices.setter
    def choices(self, value):
        """
        Setter for choices.

        :param value: The new number of response options.
        """
        self.___logger.debug("Setting choices %s", value)
        self.___choices = value

    @property
    def temperature(self):
        """
        Getter for temperature.

        :return: The temperature of the model's output.
        """
        self.___logger.debug("Getting temperature %s", self.___temperature)
        return self.___temperature

    @temperature.setter
    def temperature(self, value):
        """
        Setter for temperature.

        :param value: The new temperature of the model's output.
        """
        self.___logger.debug("Setting temperature %s", value)
        self.___temperature = value

    @property
    def top_p(self):
        """
        Getter for top_p.

        :return: The top-p value for nucleus sampling.
        """
        self.___logger.debug("Getting top_p %s", self.___top_p)
        return self.___top_p

    @top_p.setter
    def top_p(self, value):
        """
        Setter for top_p.

        :param value: The new top-p value for nucleus sampling.
        """
        self.___logger.debug("Setting top_p %s", value)
        self.___top_p = value

    @property
    def stream(self):
        """
        Getter for stream.

        :return: If True, the model will return intermediate results.
        """
        self.___logger.debug("Getting stream %s", self.___stream)
        return self.___stream

    @stream.setter
    def stream(self, value):
        """
        Setter for stream.

        :param value: The new value for stream.
        """
        self.___logger.debug("Setting stream %s", value)
        self.___stream = value

    @property
    def stop(self):
        """
        Getter for stop.

        :return: The stop sequence at which the model should stop generating further tokens.
        """
        self.___logger.debug("Getting stop %s", self.___stop)
        return self.___stop

    @stop.setter
    def stop(self, value):
        """
        Setter for stop.

        :param value: The new stop sequence.
        """
        self.___logger.debug("Setting stop %s", value)
        self.___stop = value

    @property
    def max_tokens(self):
        """
        Getter for max_tokens.

        :return: The maximum number of tokens in the output.
        """
        self.___logger.debug("Getting max_tokens %s", self.___max_tokens)
        return self.___max_tokens

    @max_tokens.setter
    def max_tokens(self, value):
        """
        Setter for max_tokens.

        :param value: The new maximum number of tokens in the output.
        """
        self.___logger.debug("Setting max_tokens %s", value)
        self.___max_tokens = value

    @property
    def presence_penalty(self):
        """
        Getter for presence_penalty.

        :return: The penalty for new token presence.
        """
        self.___logger.debug("Getting presence_penalty %s", self.___presence_penalty)
        return self.___presence_penalty

    @presence_penalty.setter
    def presence_penalty(self, value):
        """
        Setter for presence_penalty.

        :param value: The new penalty for new token presence.
        """
        self.___logger.debug("Setting presence_penalty %s", value)
        self.___presence_penalty = value

    @property
    def frequency_penalty(self):
        """
        Getter for frequency_penalty.

        :return: The penalty for token frequency.
        """
        self.___logger.debug("Getting frequency_penalty %s", self.___frequency_penalty)
        return self.___frequency_penalty

    @frequency_penalty.setter
    def frequency_penalty(self, value):
        """
        Setter for frequency_penalty.

        :param value: The new penalty for token frequency.
        """
        self.___logger.debug("Setting frequency_penalty %s", value)
        self.___frequency_penalty = value

    @property
    def logit_bias(self):
        """
        Getter for logit_bias.

        :return: The bias for the logits before sampling.
        """
        self.___logger.debug("Getting logit_bias %s", self.___logit_bias)
        return self.___logit_bias

    @logit_bias.setter
    def logit_bias(self, value):
        """
        Setter for logit_bias.

        :param value: The new bias for the logits before sampling.
        """
        self.___logger.debug("Setting logit_bias %s", value)
        self.___logit_bias = value

    @property
    def user(self):
        """
        Getter for user.

        :return: The user ID.
        """
        self.___logger.debug("Getting user %s", self.___user)
        return self.___user

    @user.setter
    def user(self, value):
        """
        Setter for user.

        :param value: The new user ID.
        """
        self.___logger.debug("Setting user %s", value)
        self.___user = value

    @property
    def functions(self):
        """
        Getter for functions.

        :return: The list of functions.
        """
        self.___logger.debug("Getting functions %s", self.___functions)
        return self.___functions

    @functions.setter
    def functions(self, value):
        """
        Setter for functions.

        :param value: The new list of functions.
        """
        self.___logger.debug("Setting functions %s", value)
        self.___functions = value

    @property
    def function_call(self):
        """
        Getter for function_call.

        :return: The function call.
        """
        self.___logger.debug("Getting function_call %s", self.___function_call)
        return self.___function_call

    @function_call.setter
    def function_call(self, value):
        """
        Setter for function_call.

        :param value: The new function call.
        """
        self.___logger.debug("Setting function_call %s", value)
        self.___function_call = value

    @property
    def function_dict(self):
        """
        Getter for function_dict.

        :return: The function dict.
        """
        self.___logger.debug("Getting function_dict %s", self.___function_dict)
        return self.___function_dict

    @function_dict.setter
    def function_dict(self, value):
        """
        Setter for function_dict.

        :param value: The new function dict.
        """
        self.___logger.debug("Setting function_dict %s", value)
        self.___function_dict = value

    @property
    def tools(self):
        """
        Getter for tools.

        :return: The list of tools.
        """
        self.___logger.debug("Getting tools %s", self.___tools)
        return self.___tools

    @tools.setter
    def tools(self, value):
        """
        Setter for tools.

        :param value: The new list of tools.
        """
        self.___logger.debug("Setting tools %s", value)
        self.___tools = value

    @property
    def tool_choice(self):
        """
        Getter for tool_choice.

        :return: The tool choice.
        """
        self.___logger.debug("Getting tool_call %s", self.___tool_choice)
        return self.___tool_choice

    @tool_choice.setter
    def tool_choice(self, value):
        """
        Setter for tool_choice.

        :param value: The new tool_choice.
        """
        self.___logger.debug("Setting tool choice %s", value)
        self.___tool_choice = value

    @property
    def tools_dict(self):
        """
        Getter for tools_dict.

        :return: The tools_dict.
        """
        self.___logger.debug("Getting function_dict %s", self.___tools_dict)
        return self.___tools_dict

    @tools_dict.setter
    def tools_dict(self, value):
        """
        Setter for tools_dict.

        :param value: The new tools dict.
        """
        self.___logger.debug("Setting tools_dict %s", value)
        self.___tools_dict = value

    @property
    def history_length(self):
        """
        Getter for history_length.

        :return: The history length.
        """
        self.___logger.debug("Getting history_length %s", self.___history_length)
        return self.___history_length

    @history_length.setter
    def history_length(self, value):
        """
        Setter for history_length.

        :param value: The new history length.
        """
        self.___logger.debug("Setting history_length %s", value)
        self.___history_length = value

    @property
    def chats(self):
        """
        Getter for chats.

        :return: The chats.
        """
        self.___logger.debug("Getting chats %s", len(self.___chats) if self.___chats else 0)
        return self.___chats

    @chats.setter
    def chats(self, value):
        """
        Setter for chats.

        :param value: The new chats.
        """
        self.___logger.debug("Setting chats %s", len(value) if value else 0)
        self.___chats = value

    @property
    def current_chat(self):
        """
        Getter for current_chat.

        :return: The current chat.
        """
        self.___logger.debug("Getting current_chat %s", self.___current_chat)
        return self.___current_chat

    @current_chat.setter
    def current_chat(self, value):
        """
        Setter for current_chat.

        :param value: The current chat.
        """
        self.___logger.debug("Setting current_chat %s", value)
        self.___current_chat = value

    @property
    def prompt_method(self):
        """
        Getter for prompt_method.

        :return: The prompt method.
        """
        self.___logger.debug("Getting prompt_method %s", self.___prompt_method)
        return self.___prompt_method

    @prompt_method.setter
    def prompt_method(self, value):
        """
        Setter for prompt_method.

        :param value: The prompt method.
        """
        self.___logger.debug("Setting prompt_method %s", value)
        self.___prompt_method = value

    @property
    def system_settings(self):
        """
        Getter for system_settings.

        :return: The system settings method.
        """
        self.___logger.debug("Getting system_settings %s", self.___system_settings)
        return self.___system_settings

    @system_settings.setter
    def system_settings(self, value):
        """
        Setter for system_settings.

        :param value: The system settings.
        """
        self.___logger.debug("Setting system_settings %s", value)
        self.___system_settings = value

    @property
    def response_format(self):
        """
        Getter for response_format.

        :return: The response format.
        """
        self.___logger.debug("Getting response_format %s", self.___response_format)
        return self.___response_format

    @response_format.setter
    def response_format(self, value):
        """
        Setter for response_format.

        :param value: The response format.
        """
        self.___logger.debug("Setting response_format %s", value)
        self.___response_format = value

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
            "response_format": self.___response_format
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
            except Exception as error:
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
            except Exception as error:
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
        Wrapper for the translate function. Returns only the content of the message.

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
