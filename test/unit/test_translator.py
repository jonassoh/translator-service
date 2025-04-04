import openai
import os
import anthropic
from src.translator import translate_content
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
from typing import Callable
from mock import patch

OPEN_API_KEY = os.environ.get("OPEN_API_KEY")

client = openai.OpenAI(
    api_key = OPEN_API_KEY
)

normal_eval_set = [
    {
        "post": "Hier ist dein erstes Beispiel.",
        "expected_answer": (False, "Here is your first example.")
    },
    {
      "post": "Je suis ravi de vous rencontrer aujourd'hui.",
      "expected_answer": (False, "I am delighted to meet you today.")
    },
    {
      "post": "こんにちは。お元気ですか？",
      "expected_answer": (False, "Hello. How are you?")
    },
    {
      "post": "Estoy aprendiendo un nuevo idioma.",
      "expected_answer": (False, "I am learning a new language.")
    },
    {
      "post": "Как твои дела в этот прекрасный день?",
      "expected_answer": (False, "How are you on this beautiful day?")
    },
    {
      "post": "Merhaba, bugün nasılsınız?",
      "expected_answer": (False, "Hello, how are you today?")
    },
    {
      "post": "Это очень интересный проект.",
      "expected_answer": (False, "This is a very interesting project.")
    },
    {
      "post": "你今天过得怎么样？",
      "expected_answer": (False, "How are you doing today?")
    },
    {
      "post": "Καλημέρα! Πώς είστε;",
      "expected_answer": (False, "Good morning! How are you?")
    },
    {
      "post": "Nam, điều đó thật tuyệt vời!",
      "expected_answer": (False, "Yes, that is wonderful!")
    },
    {
      "post": "La vie est un voyage extraordinaire plein de défis et d'opportunités. Chaque jour nous apporte de nouvelles expériences, nous permettant d'apprendre, de grandir et de nous transformer. Les obstacles que nous rencontrons ne sont pas des barrières, mais des chances de développer notre résilience et notre compréhension du monde qui nous entoure.",
      "expected_answer": (False, "Life is an extraordinary journey full of challenges and opportunities. Each day brings us new experiences, allowing us to learn, grow, and transform. The obstacles we encounter are not barriers, but chances to develop our resilience and understanding of the world around us.")
    },
    {
      "post": "環境保護は現代社会における最も重要な課題の一つです。気候変動、生物多様性の喪失、持続不可能な資源の利用は、私たちの惑星の未来に深刻な脅威をもたらしています。個人、企業、政府が協力して、より持続可能な生活様式と経済モデルを追求することが不可欠です。",
      "expected_answer": (False, "Environmental protection is one of the most important challenges in modern society. Climate change, loss of biodiversity, and unsustainable resource use pose serious threats to the future of our planet. It is essential for individuals, businesses, and governments to work together to pursue more sustainable lifestyles and economic models.")
    },
    {
      "post": "La educación es la clave para el desarrollo personal y social. No se trata solo de adquirir conocimientos académicos, sino de desarrollar pensamiento crítico, empatía y habilidades que nos permitan contribuir positivamente a nuestra comunidad. Cada persona tiene el potencial de ser un agente de cambio, independientemente de su origen o circunstancias.",
      "expected_answer": (False, "Education is the key to personal and social development. It is not just about acquiring academic knowledge, but about developing critical thinking, empathy, and skills that allow us to contribute positively to our community. Every person has the potential to be an agent of change, regardless of their background or circumstances.")
    },
    {
      "post": "Психическое здоровье является столь же важным, как и физическое. В современном быстро меняющемся мире люди подвергаются огромному стрессу и эмоциональным нагрузкам. Крайне важно уделять внимание своему эмоциональному благополучию, практиковать самозаботу и не стесняться обращаться за профессиональной помощью, когда это необходимо.",
      "expected_answer": (False, "Mental health is just as important as physical health. In today's rapidly changing world, people are subjected to enormous stress and emotional burdens. It is crucial to pay attention to one's emotional well-being, practice self-care, and not be afraid to seek professional help when needed.")
    },
    {
        "post": "データプライバシーは現代社会における重要な課題となっています。",
        "expected_answer": (False, "Data privacy has become an important issue in modern society.")
    },
    {
        "post": "Technology is transforming every aspect of our lives, from how we work and communicate to how we learn and entertain ourselves.",
        "expected_answer": (True, "Technology is transforming every aspect of our lives, from how we work and communicate to how we learn and entertain ourselves.")
    },
    {
        "post": "Climate change represents one of the most significant challenges facing our global community in the 21st century.",
        "expected_answer": (True, "Climate change represents one of the most significant challenges facing our global community in the 21st century.")
    },
    {
        "post": "Mental health awareness is crucial for building a compassionate and supportive society.",
        "expected_answer": (True, "Mental health awareness is crucial for building a compassionate and supportive society.")
    },
    {
        "post": "Artificial intelligence continues to push the boundaries of what's possible in technology and scientific research.",
        "expected_answer": (True, "Artificial intelligence continues to push the boundaries of what's possible in technology and scientific research.")
    },
    {
        "post": "Sustainable development requires collaboration between governments, businesses, and individual citizens.",
        "expected_answer": (True, "Sustainable development requires collaboration between governments, businesses, and individual citizens.")
    },
    {
        "post": "Education is the most powerful weapon we can use to change the world.",
        "expected_answer": (True, "Education is the most powerful weapon we can use to change the world.")
    },
    {
        "post": "Cultural diversity enriches our understanding of the world and promotes mutual respect.",
        "expected_answer": (True, "Cultural diversity enriches our understanding of the world and promotes mutual respect.")
    },
    {
        "post": "Renewable energy technologies are crucial for addressing global climate challenges.",
        "expected_answer": (True, "Renewable energy technologies are crucial for addressing global climate challenges.")
    },
    {
        "post": "Empathy and emotional intelligence are key skills for personal and professional success.",
        "expected_answer": (True, "Empathy and emotional intelligence are key skills for personal and professional success.")
    },
    {
        "post": "Digital literacy has become as important as traditional literacy in the modern world.",
        "expected_answer": (True, "Digital literacy has become as important as traditional literacy in the modern world.")
    },
    {
        "post": "Innovation thrives when we create environments that encourage creativity and critical thinking.",
        "expected_answer": (True, "Innovation thrives when we create environments that encourage creativity and critical thinking.")
    },
    {
        "post": "Global interconnectedness requires us to develop a more nuanced understanding of different cultures and perspectives.",
        "expected_answer": (True, "Global interconnectedness requires us to develop a more nuanced understanding of different cultures and perspectives.")
    },
    {
        "post": "Personal growth is a lifelong journey of learning, self-discovery, and continuous improvement.",
        "expected_answer": (True, "Personal growth is a lifelong journey of learning, self-discovery, and continuous improvement.")
    },
    {
        "post": "Ethical considerations in technology development are more important now than ever before.",
        "expected_answer": (True, "Ethical considerations in technology development are more important now than ever before.")
    },
    {
        "post": "The power of human connection transcends geographical boundaries and technological barriers.",
        "expected_answer": (True, "The power of human connection transcends geographical boundaries and technological barriers.")
    }
]

def eval_single_response_complete(expected_answer: tuple[bool, str], llm_response: tuple[bool, str]) -> float:
  '''TODO: Compares an LLM response to the expected answer from the evaluation dataset using one of the text comparison metrics.'''
  (llm_language_boolean, llm_translation) = llm_response
  (expected_language_boolean, expected_translation) = expected_answer
  if expected_language_boolean != llm_language_boolean:
    return 0
  embeddings = model.encode([expected_translation, llm_translation])
  similarities = model.similarity(embeddings, embeddings)
  # print(f"expected_translation: {expected_translation}")
  # print(f"llm_translation: {llm_translation}")
  # print(f"similarity: {similarities[1][0]}")
  return similarities[1][0]

def evaluate_set(query_fn: Callable[[str], str], eval_fn: Callable[[str, str], float], dataset) -> float:
  for data in dataset:
    llm_response = query_fn(data['post'])
    score = eval_fn(data['expected_answer'], llm_response)
    # print(f"{data}")
    # print(f"{llm_response}")
    # print(f"{data['expected_answer']}")
    assert score >= 0.9

def test_llm_normal_response():
    evaluate_set(translate_content, eval_single_response_complete, normal_eval_set)
    return

def test_llm_gibberish_response():
    gibberish_eval_set = [
    {
        "post": "sdfjk23r0q3j4r0q3j4r0q3j4r0q3j4r0q3j4",
        "expected_answer": (False, "sdfjk23r0q3j4r0q3j4r0q3j4r0q3j4r0q3j4")
    },
    {
        "post": "ㅓ닏ㄹ ㅐㅑㅈㄷ  дылоуа дЫуша зчьу",
        "expected_answer": (False, "ㅓ닏ㄹ ㅐㅑㅈㄷ  дылоуа дЫуша зчьу")
    },
    {
        "post": "ㅣㄷ메푸니뱌ㅣㅁ 디ㅏㄴㄷ 베ㅑㅈ덜 베ㅐ자",
        "expected_answer": (False, "ㅣㄷ메푸니뱌ㅣㅁ 디ㅏㄴㄷ 베ㅑㅈ덜 베ㅐ자")
    },
    {
        "post": "шшшшш ддддд нннн ккккк",
        "expected_answer": (False, "шшшшш ддддд нннн ккккк")
    },
    {
        "post": "واوووووو يا سلاaaaaamseflj",
        "expected_answer": (False, "واوووووو يا سلاaaaaamseflj")
    }
    ]
    evaluate_set(translate_content, eval_single_response_complete, gibberish_eval_set)
    return

def test_llm_unexpected_response():
    @patch.object(client.chat.completions, 'create')
    def test_response_not_require_translation(mocker):
        # we mock the model's response to return a random message
        mocker.return_value.choices[0].message.content = "The post contains a symbol and does not require translation."

        # TODO assert the expected behavior
        assert translate_content("1+1") == (False, "The post contains a symbol and does not require translation.")
        assert translate_content("λ") == (False, "The post contains a symbol and does not require translation.")
        assert translate_content("??") == (False, "The post contains a symbol and does not require translation.")
        assert translate_content("!=") == (False, "The post contains a symbol and does not require translation.")

    @patch.object(client.chat.completions, 'create')
    def test_response_malformed(mocker):
        # we mock the model's response to return a random message
        mocker.return_value.choices[0].message.content = "the post is malformed"

        # TODO assert the expected behavior
        assert translate_content("ㄱㅅ") == (False, "the post is malformed")
        assert translate_content(":D") == (False, "the post is malformed")
        assert translate_content("ㅇㅇ") == (False, "the post is malformed")
        assert translate_content("ㅇㅅㅇ") == (False, "the post is malformed")

    @patch.object(client.chat.completions, 'create')
    def test_response_fails_translation(mocker):
        # we mock the model's response to return a random message
        mocker.return_value.choices[0].message.content = "Это очень интересный проект."

        # TODO assert the expected behavior
        assert translate_content("Это очень интересный проект.") == (False, "Это очень интересный проект.")

    @patch.object(client.chat.completions, 'create')
    def test_response_empty(mocker):
        # we mock the model's response to return a random message
        mocker.return_value.choices[0].message.content = ""

        # TODO assert the expected behavior
        assert translate_content("") == (False, "")
        assert translate_content(":D") == (False, ":D")
        assert translate_content("Hier ist dein erstes Beispiel.") == (False, "Hier ist dein erstes Beispiel.")

    @patch.object(client.chat.completions, 'create')
    def test_response_request_more_info(mocker):
        # we mock the model's response to return a random message
        mocker.return_value.choices[0].message.content = "The post appears to be a single Greek letter, 'Σ' (Sigma), which has no full context to translate. If you have more text or a specific request, please provide it!"

        # TODO assert the expected behavior
        assert translate_content("Σ") == (False, "The post appears to be a single Greek letter, 'Σ' (Sigma), which has no full context to translate. If you have more text or a specific request, please provide it!")

    @patch.object(client.chat.completions, 'create')
    def test_unexpected_language(mocker):
        # we mock the model's response to return a random message
        mocker.return_value.choices[0].message.content = "I don't understand"

        # TODO assert the expected behavior
        assert translate_content("ㅇㅅㅇ") == (False, "I don't understand")

    @patch.object(client.chat.completions, 'create')
    def test_response_cannot_say_that(mocker):
        # we mock the model's response to return a random message
        mocker.return_value.choices[0].message.content = "I am not permitted to translate this statement"

        # TODO assert the expected behavior
        assert translate_content("some super innappropriate text") == (False, "I am not permitted to translate this statement")
        assert translate_content("something that would trigger the ccp") == (False, "I am not permitted to translate this statement")

    @patch.object(client.chat.completions, 'create')
    def test_response_is_none(mocker):
        # we mock the model's response to return a random message
        mocker.return_value.choices[0].message.content = None

        # TODO assert the expected behavior
        assert translate_content("text that turned into None") == (False, "text that turned into None")

    @patch.object(client.chat.completions, 'create')
    def test_response_is_not_string(mocker):
        # we mock the model's response to return a random message
        mocker.return_value.choices[0].message.content = 123

        # TODO assert the expected behavior
        assert translate_content("stays a string") == (False, "stays a string")

        # we mock the model's response to return a random message
        mocker.return_value.choices[0].message.content = True

        # TODO assert the expected behavior
        assert translate_content("stays a string") == (False, "stays a string")

        # we mock the model's response to return a random message
        mocker.return_value.choices[0].message.content = 1.2

        # TODO assert the expected behavior
        assert translate_content("stays a string") == (False, "stays a string")

        # we mock the model's response to return a random message
        mocker.return_value.choices[0].message.content = []

        # TODO assert the expected behavior
        assert translate_content("stays a string") == (False, "stays a string")

        # we mock the model's response to return a random message
        mocker.return_value.choices[0].message.content = {}

        # TODO assert the expected behavior
        assert translate_content("stays a string") == (False, "stays a string")