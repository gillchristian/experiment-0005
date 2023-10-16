import transformers

from transformers import pipeline

summarizer = pipeline("summarization", model = "tiiuae/falcon-7b")

text = """
This is a text to be summarized
"""

summarizer(text)
