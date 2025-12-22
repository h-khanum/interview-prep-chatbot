def generate_feedback(similarity, keyword_score):
    final_score = (similarity + keyword_score) / 2

    if final_score >= 0.7:
        return "✅ Correct answer! Well done."
    elif final_score >= 0.4:
        return "⚠️ Partially correct. You’re on the right track."
    else:
        return "❌ Needs improvement. Review this concept again."
