# -*- coding: utf-8 -*-
"""
Filename: gpt_statistics.py
Author: Iliya Vereshchagin
Copyright (c) 2023. All rights reserved.

Created: 14.10.2023
Last Modified: 14.10.2023

Description:
This file contains implementation for Statistics module for ChatGPT.
"""


class GPTStatistics:
    """
    The GPTStatistics class is for managing an instance of the GPTStatistics model.

    Parameters:
    prompt_tokens (int): The number of tokens in the prompt. Default is 0.
    completion_tokens (int): The number of tokens in the completion. Default is 0.
    total_tokens (int): The total number of tokens. Default is 0.
    """

    def __init__(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ):
        """
        Constructs all the necessary attributes for the GPTStatistics object.

        :param prompt_tokens: The number of tokens in the prompt. Default is 0.
        :param completion_tokens: The number of tokens in the completion. Default is 0.
        :param total_tokens: The total number of tokens. Default is 0.
        """
        self.___prompt_tokens = prompt_tokens
        self.___completion_tokens = completion_tokens
        self.___total_tokens = total_tokens

    @property
    def prompt_tokens(self):
        """
        Getter for prompt_tokens.

        :return: The prompt tokens.
        """
        return self.___prompt_tokens

    @prompt_tokens.setter
    def prompt_tokens(self, value):
        """
        Setter for prompt_tokens.

        :param value: The prompt tokens.
        """
        self.___prompt_tokens = value

    @property
    def completion_tokens(self):
        """
        Getter for completion_tokens.

        :return: The completion tokens.
        """
        return self.___completion_tokens

    @completion_tokens.setter
    def completion_tokens(self, value):
        """
        Setter for completion_tokens.

        :param value: The completion tokens.
        """
        self.___completion_tokens = value

    @property
    def total_tokens(self):
        """
        Getter for completion_tokens.

        :return: The completion tokens.
        """
        return self.___total_tokens

    @total_tokens.setter
    def total_tokens(self, value):
        """
        Setter for total_tokens.

        :param value: The total tokens.
        """
        self.___total_tokens = value

    def add_prompt_tokens(self, value):
        """
        Adder for prompt_tokens.

        :param value: The prompt tokens.
        """
        self.prompt_tokens += value

    def add_completion_tokens(self, value):
        """
        Adder for completion_tokens.

        :param value: The completion tokens.
        """
        self.completion_tokens += value

    def add_total_tokens(self, value):
        """
        Adder for total_tokens.

        :param value: The total tokens.
        """
        self.total_tokens += value

    def set_tokens(self, prompt_tokens, completion_tokens, total_tokens):
        """
        Sets all tokens statistics in bulk

        :param prompt_tokens: The prompt tokens.
        :param completion_tokens: The prompt tokens.
        :param total_tokens: The prompt tokens.
        """
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens

    def add_tokens(self, prompt_tokens, completion_tokens, total_tokens):
        """
        Adds all tokens statistics in bulk

        :param prompt_tokens: The prompt tokens.
        :param completion_tokens: The prompt tokens.
        :param total_tokens: The prompt tokens.
        """
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total_tokens

    def get_tokens(self):
        """
        Returns a dictionary of the class attributes and their values.

        :return: dict with tokens statistics
        """
        return {
            "prompt_tokens": self.___prompt_tokens,
            "completion_tokens": self.___completion_tokens,
            "total_tokens": self.___total_tokens,
        }
