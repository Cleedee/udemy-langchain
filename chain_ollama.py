
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_huggingface import ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

def main():
    llm = HuggingFaceHub(
        repo_id='meta-llama/Meta-Llama-3-8B-Instruct',
        model_kwargs={
            'temperature': 0.1,
            'return_full_text': False,
            'max_new_tokens': 512,
        }
    )
    system_prompt = "Você é um sistema prestativo e está respondendo perguntas gerais."
    user_prompt = "{input}"
    token_s, token_e = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    prompt = ChatPromptTemplate.from_messages([
        ("system", token_s + system_prompt),
        ("user", user_prompt + token_e)
    ])

    chain = prompt | llm

    input = "Explique para mim em até um parágrafo o conceito de redes neurais, de forma clara e objetiva"

    res = chain.invoke({"input": input})

    print(res)
    print("----------")

    ### Exemplo com Ollama

    llm = ChatOllama(
        model="phi3",
        temperature=0.1
    )

    chain3 = prompt | llm
    res = chain3.invoke({"iput": input})
    print(res.content)


if __name__ == "__main__":
    main()
