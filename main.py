from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_huggingface import ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
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

if __name__ == "__main__":
    main()
