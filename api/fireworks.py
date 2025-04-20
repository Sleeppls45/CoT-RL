import openai

client = openai.OpenAI(
    base_url = "https://api.fireworks.ai/inference/v1",
    api_key="<FIREWORKS_API_KEY>",
)
response = client.chat.completions.create(
  model="accounts/fireworks/models/llama-v3-8b-instruct",
  messages=[{
    "role": "user",
    "content": "Say this is a test",
  }],
)
print(response.choices[0].message.content)
