FROM huggingface/transformers-pytorch-gpu

EXPOSE 5000

WORKDIR /app

CMD ["python3", "app.py"]