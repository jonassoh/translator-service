import openai
import os

client = openai.OpenAI()

def get_translation(post: str) -> str:
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
          "role": "system",
          "content":
          '''You are a professional translator who knows all languages.
            If the parts or all of the post contains numbers or symbols, keep those parts exactly as they are.
            Translate any post written in a non-English language into American English.
            If any parts of the post do not need to be translated, keep them as they are.
            Treat posts with numbers, IP addresses, or symbols as valid text.
            If a post is malformed, return the original post
          '''
        },
        {
            "role": "user",
            "content": post
        }
    ]
)
    return response.choices[0].message.content

def get_language(post: str) -> str:
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
          "role": "system",
          "content": "You are a professional translator who can classify English and non-English text based on their language. All of your responses will be in English and just the language that the post is in."
        },
        {
            "role": "user",
            "content": post
        }
    ]
)
    return response.choices[0].message.content
    
def translate_content(post: str) -> tuple[bool, str]:
    english = False
    translation = post

    if get_language(post) == "English":
        english = True
        translation = post
    else:
        # Ensure translation defaults to original post if get_translation fails
        translation = get_translation(post) or post
        if type(translation) != str:
          translation = post

    # Check if translation is malformed or indicates no translation needed
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": ''' Your job is to determine if the translation provided is valid or not.
                If the input states along the lines of `does not require translation` or `malformed`, respond with `failed translate`.
                Otherwise, return the input as is.
                '''
            },
            {
                "role": "user",
                "content": f"original post: {post} translation: {translation}"
            }
        ]
    )

    if response.choices[0].message.content == 'failed translate':
        translation = post

    return (english, translation)
