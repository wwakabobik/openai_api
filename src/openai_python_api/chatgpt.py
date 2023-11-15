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

import logging
from os import environ
from typing import Optional

from openai import AsyncOpenAI

from .gpt_statistics import GPTStatistics
from .logger_config import setup_logger
from .models import COMPLETIONS
from .chatgpt_audio import ChatGPTAudioMixin
from .chatgpt_dump import ChatGPTDumpMixin


# pylint: disable=too-many-instance-attributes,too-many-public-methods
class ChatGPT(ChatGPTAudioMixin, ChatGPTDumpMixin):
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
