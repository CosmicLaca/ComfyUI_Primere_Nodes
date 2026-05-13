res = mistral.chat.complete(
    model=model,
    messages=[
        {
            "role": role,
            "content": prompt
        },
    ],
    stream=False,
    response_format={
        "type": response,
    }
)