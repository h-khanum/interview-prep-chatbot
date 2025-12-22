from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def similarity_score(user_answer, model_answer):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([user_answer, model_answer])
    
    score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return score


def keyword_score(user_tokens, keywords):
    matched = 0
    for word in keywords:
        if word.lower() in user_tokens:
            matched += 1
    return matched / len(keywords)
