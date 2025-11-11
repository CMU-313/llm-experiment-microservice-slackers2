import os
import json
import ollama

def get_translation(client, model_name: str, post: str) -> str:
    context = """You are a professional translator. Please take the inputted
    text and identify the language and translate it to English. If the post
    is already English then please just output the same text. Please output
    the translated text and only the translated text.
    """

    translation_input = f"INPUT: {post}"

    response = client.chat(
      model=model_name,  # model name
      messages=[
          {
              "role": "system",
              "content": context
          },
          {
              "role": "user",
              "content": translation_input
          }
      ]
    )

    return response.message.content

def get_language(client, model_name: str, post: str) -> str:
    context = """You are a professional language classifier. Your job is to take
    input text and return the language of that text. Please return the english
    name of the language and only the language.
    """

    translation_input = f"INPUT: {post}"

    response = client.chat(
      model=model_name,  # model name
      messages=[
          {
              "role": "system",
              "content": context
          },
          {
              "role": "user",
              "content": translation_input
          }
      ]
    )

    print("Got language response: ", response.message.content)

    return response.message.content

def translate_content(content: str) -> tuple[bool, str]:
    """
    Detects if content is in English and translates it if not.
    
    Args:
        content: The text to analyze and potentially translate
        
    Returns:
        tuple: (is_english: bool, translated_content: str)
    """
    # Handle empty or whitespace-only strings
    if not content or content.strip() == "":
        return False, content
    
    # Get Ollama configuration from environment
    ollama_host = ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    ollama_model = 'qwen3:0.6b'
    
    # Create Ollama client
    client = ollama.Client(host=ollama_host)
    
    try:
        # Attempt language detection
        llm_language = get_language(client, ollama_model, content)
        if not isinstance(llm_language, str) or not llm_language.strip():
            return (False, "I was not able to translate")

        # Attempt translation
        llm_translation = get_translation(client, ollama_model, content)
        if not isinstance(llm_translation, str) or not llm_translation.strip():
            return (False, "I was not able to translate")

        # Check if original text was in English
        is_english = llm_language.strip().lower() == "english"

        return (is_english, llm_translation.strip())

    except Exception as e:
        return (False, "I was not able to translate")
