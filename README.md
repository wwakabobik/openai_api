# OpenAI Python API

This package provides a Python API for [OpenAI](https://openai.com/), based on the official [API documentation](https://openai.com/blog/openai-api) and wraps-up original [OpenAI API](https://pypi.org/project/openai/).

## Installation

Package based on Python 3.9 and might be incompatible with older versions.
To install the package, use the package from [pypi](https://pypi.org/project/openai-api/):

```bash
pip install openai-api
```

This package contains API for ChatGPT and DALL-E2, but they not fully covered yet. More functionality will be added in the future.

You need to be registered on [OpenAI](https://openai.com/) and have an API key to use this package. You can find your API key on the [API tokens](https://platform.openai.com/account/api-keys) page.

## ChatGPT

The `ChatGPT` class is for managing an instance of the ChatGPT model. 

### Fast start

Here's a basic example of how to use the API:

```python
from openai_api import ChatGPT

# Use your API key and any organization you wish
chatgpt = ChatGPT(auth_token='your-auth-token', organization='your-organization', prompt_method=True)
response = chatgpt.str_chat("Hello, my name is John Connor!")
print(response)
```

This will produce the following output:

<IMAGE>

### Creating personalized a ChatGPT instance

You may need to create a custom ChatGPT instance to use the API. You can do this by passing the following parameters to the `ChatGPT` constructor:

- `model` (str): The name of the model, Default is 'gpt-4'. List of models you can find in models.py or [here](https://platform.openai.com/docs/models/overview).
- `temperature` (float, optional): The temperature of the model's output. Default is 1.
- `top_p` (float, optional): The top-p value for nucleus sampling. Default is 1.
- `stream` (bool, optional): If True, the model will return intermediate results. Default is False.
- `stop` (str, optional): The stop sequence at which the model should stop generating further tokens. Default is None.
- `max_tokens` (int, optional): The maximum number of tokens in the output. Default is 1024.
- `presence_penalty` (float, optional): The penalty for new token presence. Default is 0.
- `frequency_penalty` (float, optional): The penalty for token frequency. Default is 0.
- `logit_bias` (map, optional): The bias for the logits before sampling. Default is None.
- `history_length` (int, optional): Length of history. Default is 5.
- `prompt_method` (bool, optional): prompt method. Use messages if False, otherwise - prompt. Default if False.
- `system_settings` (str, optional): general instructions for chat. Default is None.

Most of these params reflects OpenAI model parameters. You can find more information about them in the [OpenAI API documentation](https://platform.openai.com/docs/api-reference/chat/create). If you need to get/change them, just use it directly via `chatgpt.model.temperature = 0.5` or `current_temperature = chatgpt.model.temperature`.
But several params are specific for this API: `prompt_method` is stub to use direct input to model without usage of "messages" and managing/storing them. It might be an option if you need to trigger chat only once, or you don't need to pass extra messages and instructions to chat. `system_settings` is used to store bot global instructions, like how to behave, how to act and format output. Refer to [Best practices](https://platform.openai.com/docs/guides/gpt-best-practices/tactic-ask-the-model-to-adopt-a-persona). `history_length` is used to store history of messages. It's used to pass messages to model in a single request. Default is 5, but you can change it if you need to store more messages. More you pass, more expensive request will be.

```python
chatgpt = ChatGPT(auth_token='your-auth-token', organization='your-organization', model='chatgpt3.5', history_length=10)
chatgpt.model.temperature = 0.5
chatgpt.model.top_p = 0.9
```

Here is a difference between `prompt_method=True` and `prompt_method=False` wih message history:

<IMAGE>

### Managing chats

If you plan to use several users or chats, you can use next params while creating `ChatGPT` instance or set them later. Parameters are:

- `user` (str, optional): The user ID. Default is ''. This field is used to identify global user model. Usually, it's a master of the ChatGPT instance.
- `current_chat` (str, optional): Default chat will be used. Default is None. This field is used to identify current chat. If user uses some chat, or it's not created yet, it will be created and stored into this value.
- `chats` (dict, optional): Chats dictionary, contains all chats. Default is None. It's placeholder for all chats. You can set this value to any dict to restore, replace or flush chat history for ChatGPT instance.

So, in general, you can use `ChatGPT` instance as a chatbot for one user, or as a chatbot for several users. If you need to use it as a chatbot for several users, you need to create a `ChatGPT` instance for each user and store it somewhere. You can use `chats` parameter to store all chats in one place, or you can store them in a database, or you can store them in a file. It's up to you.

### Obtaining enhanced responses

In most cases you need to get a response from the model as string (which you can use directly or format in your frontend). But in some rare cases you may need to get raw answer from the model. In this case you can use `process_chat` method. It returns a `ChatCompletion` object, which contains all information about the model's response. You can use it to get raw response, or you can use it to get formatted response. Moreover, it's an only way to obtain several choices at once (i.e. you need 4 different answers from the model).

```python
chatgpt = ChatGPT(auth_token='your-auth-token', organization='your-organization', choices=4)
chatgpt.chat({"role": "user", "content": "Do Androids Dream of Electric Sheep?"})
chatgpt.process_chat({"role": "user", "content": "What use was time to those who'd soon achieve Digital Immortality?"})
```

<IMAGE>

### Service methods

Fore some reasons you may use ChatGPT instance to transcribe audiofile into text or translate text to English.

```python
# you may use response_format with following values: json, text, srt, verbose_json, or vtt. Default is text.
transcripted_string = chatgpt.transcript('audiofile.mp3', language='russian', response_format='text') 
translated_string = chatgpt.translate(transcripted_string, response_format='json')
```

<IMAGE>

For details refer to [OpenAI API documentation](https://platform.openai.com/docs/api-reference/audio) for Audio topic.


### Store data

You also may need to store your settings and chats. You may use following methods:

```python
settings = chatgpt.dump_settings()  # dumps ChatGPT settings to JSON
chats = chatgpt.dump_chats()  # dumps all chats to JSON
chat = chatgpt.dump_chat('my_secure_chat')  # dumps chat to JSON
```

### Using functions

To empower your ChatGPT you may want to use _functions_. Functions are a way to extend the functionality of the model. You can use functions to get some data from the model, or to change the model's behavior. To use them you need to pass several parameters to the `ChatGPT` constructor or it's attributes:

```python
gpt_functions = [{
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },]
chatgpt.functions = gpt_functions
chatgpt.function_dict = {"get_current_weather": obtain_weather}  # this is your function somewhere in your code
chatgpt.function_call = "auto"  # none to disable or use dict like {"name": "my_function"} to call them manually
```
Now, when you're going to asking the ChatGPT about something, it will return related info using your function.

<IMAGE>

For details refer to [OpenAI API documentation](https://platform.openai.com/docs/guides/gpt/function-calling) for functions.

## DALL-E

The `DALLE` class is for managing an instance of the DALL-E models. You can generate or edit images using this class.

### Fast start

Here's a basic example of how to use the API:

```python
from openai_api import DALLE

# Use your API key and any organization you wish
dalle = DALLE(auth_token='your-auth-token', organization='your-organization')
images = dalle.create_image_url("cybernetic cat")  # will return list of urls to images
```

### Creating personalized DALL-E instance

You may need to create a custom DALL-E instance to use the API. You can do this by passing the following parameters to the `DALLE` constructor or just set them later:

- `default_count` (int): Default count of images to produce. Default is 1.
- `default_size` (str): Default dimensions for output images. Default is "512x512". "256x256" and "1024x1024" as option.
- `default_file_format` (str): Default file format. Optional. Default is 'PNG'.
- `user` (str, optional): The user ID. Default is ''.

```python
dalle = DALLE(auth_token='your-auth-token', organization='your-organization', default_count=3, default_size="256x256")
dalle.default_file_format = 'JPG'
```

### Methods

You can use following methods to generate images:

```python
image_bytes = dalle.create_image_data("robocop")  # will return list of images (bytes format).
image_dict = dalle.create_image("night city")  # will return list of images (dict format).
```

You can save bytes image to file:

```python
# if file format is None, it will be taken from class attribute
image_file = dalle.save_image(image = image_bytes[0], filename="robocop", file_format=None) # will return filename
```

You can use following methods to edit images:

```python
with open("robocop.jpg", "rb") as image_file:
    edited_image1 = dalle.edit_image_from_file(file=image_file, prompt="change color to pink")  # return of bytes format
# or use url
edited_image2 = dalle.edit_image_from_url(url=night_city_url, prompt="make it daylight")  # return of bytes format
```

You can use following methods to create variations of images:

```python
with open("robocop.jpg", "rb") as image_file:
    variated_image1 = dalle.create_variation_from_file(file=image_file)  # return of bytes format
# or use url
variated_image2 = dalle.create_variation_from_url(url=night_city_url)  # return of bytes format
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Donations
If you like this project, you can support it by donating via [DonationAlerts](https://www.donationalerts.com/r/rocketsciencegeek).
