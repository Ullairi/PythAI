import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chains.combine_documents import create_stuff_documents_chain

# Загружаем ключ
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("NO API KEY FOUND")

# Загружаем документы с веб-страницы
url = "https://de.wikipedia.org/wiki/Wikipedia:Hauptseite"
loader = WebBaseLoader(url)
docs = loader.load()

# Инициализируем генеративную модель Google
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Создаем шаблон для промптов,
prompt = ChatPromptTemplate.from_template("Text short summatize :\n\n{context}")

# создаем цепочку которая объединяет документы
chain = create_stuff_documents_chain(llm, prompt)

summary = chain.invoke({"context": docs})

# Выводим полученный результат
print(" Web-page summary:")
print(summary)