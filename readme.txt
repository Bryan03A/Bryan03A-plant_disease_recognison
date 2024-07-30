f1408d640a5e38ddbdb8b8123568e8a5

./model/saved_model


from openai import OpenAI

client = OpenAI(
    api_key = "sk-None-E4oN7muqvNOs7s2Xsc47T3BlbkFJtUF54VxxwEndne5lold1"
)

completion = client.chat.completions.create(
    model = "gpt-3.5-turbo",
    messages = [
        {"role": "system", "content": "Eres un asistente virtual" },
        {"role": "user", "content": "Crea un poema"}
    ]
)

print(completion.choices[0].message)