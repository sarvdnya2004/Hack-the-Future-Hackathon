from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

def generate_summary(text, sentence_count=3):
    """
    Generates a concise summary from the given text using LSA (Latent Semantic Analysis).
    
    Args:
        text (str): The input text to summarize.
        sentence_count (int): Number of sentences in the summary.

    Returns:
        str: A summarized version of the text.
    """
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)
