# def translate_content(content: str) -> tuple[bool, str]:
#     if content == "这是一条中文消息":
#         return False, "This is a Chinese message"
#     if content == "Ceci est un message en français":
#         return False, "This is a French message"
#     if content == "Esta es un mensaje en español":
#         return False, "This is a Spanish message"
#     if content == "Esta é uma mensagem em português":
#         return False, "This is a Portuguese message"
#     if content  == "これは日本語のメッセージです":
#         return False, "This is a Japanese message"
#     if content == "이것은 한국어 메시지입니다":
#         return False, "This is a Korean message"
#     if content == "Dies ist eine Nachricht auf Deutsch":
#         return False, "This is a German message"
#     if content == "Questo è un messaggio in italiano":
#         return False, "This is an Italian message"
#     if content == "Это сообщение на русском":
#         return False, "This is a Russian message"
#     if content == "هذه رسالة باللغة العربية":
#         return False, "This is an Arabic message"
#     if content == "यह हिंदी में संदेश है":
#         return False, "This is a Hindi message"
#     if content == "นี่คือข้อความภาษาไทย":
#         return False, "This is a Thai message"
#     if content == "Bu bir Türkçe mesajdır":
#         return False, "This is a Turkish message"
#     if content == "Đây là một tin nhắn bằng tiếng Việt":
#         return False, "This is a Vietnamese message"
#     if content == "Esto es un mensaje en catalán":
#         return False, "This is a Catalan message"
#     if content == "This is an English message":
#         return False, "This is an English message"
#     return False, "Not hardcoded"
import openai
import os

api_key = os.environ.get("OPENAI_API_KEY")

client = openai.OpenAI(api_key=api_key)

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
