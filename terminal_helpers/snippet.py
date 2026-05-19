response = requests.post("https://openrouter.ai/api/v1/chat/completions",
    headers={
      "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
      "Content-Type": "application/json"
    },
    json={
        "model": model,
        "messages": prompt,
        "temperature": "FLOAT",
        "max_tokens": "INT",
        "top_p": "FLOAT",
        "frequency_penalty": "FLOAT",
        "presence_penalty": "FLOAT",
        "provider": {
            "data_collection": "deny",
            "quantizations": quantizations
        }
    }
)
