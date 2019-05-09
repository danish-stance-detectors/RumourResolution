from afinn import Afinn

afinn = Afinn(language='da', emoticons=True)

def get_afinn_sentiment(text):
    return afinn.score(text)