"""
:synopsis: Global NLP models available. The models available here
are all derived from Hugging Face[https://huggingface.co/models].
Each of the functions available here take in the model strings from Hugging Face's website.
.. moduleauthor:: Naval Bhandari <naval@neuro-ai.co.uk>
"""

from ..core import nlp_constructor


def QuestionAnswering(pretrained=""):
    return nlp_constructor("QuestionAnswering", pretrained)


def SequenceClassification(pretrained=""):
    return nlp_constructor("SequenceClassification", pretrained)


def TokenClassification(pretrained=""):
    return nlp_constructor("TokenClassification", pretrained)


def MultipleChoice(pretrained=""):
    return nlp_constructor("MultipleChoice", pretrained)


def Seq2SeqLM(pretrained=""):
    return nlp_constructor("Seq2SeqLM", pretrained)


def MaskedLM(pretrained=""):
    return nlp_constructor("MaskedLM", pretrained)


def CausalLM(pretrained=""):
    return nlp_constructor("CausalLM", pretrained)


def LMHead(pretrained=""):
    return nlp_constructor("LMHead", pretrained)


def Headless(pretrained=""):
    return nlp_constructor("", pretrained)
