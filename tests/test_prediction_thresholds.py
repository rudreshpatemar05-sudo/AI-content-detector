from src.predict import predict, split_paragraphs


def test_split_paragraphs_keeps_each_paragraph_separate():
    text = "First paragraph has a few words.\n\nSecond paragraph has different words."

    assert split_paragraphs(text) == [
        "First paragraph has a few words.",
        "Second paragraph has different words.",
    ]


def test_human_and_ai_examples_are_not_given_the_same_label():
    human_text = (
        "I was walking home from the station when I noticed the sky turning orange behind the city buildings. "
        "The street felt unusually quiet, and for a moment I stopped to listen to the wind moving through the trees. "
        "It was a small moment, but it made me think about how ordinary details can feel meaningful when you slow down enough to notice them."
    )
    ai_text = (
        "In modern digital workflows, organizations increasingly rely on automation to streamline repetitive tasks, "
        "improve accessibility, and generate structured outputs across departments. This approach can help teams "
        "process large volumes of information more efficiently while maintaining consistency, but it also requires "
        "careful oversight to ensure quality, fairness, and accountability in decision-making."
    )

    human_result = predict(human_text)
    ai_result = predict(ai_text)

    assert human_result["label"] == "likely_human"
    assert ai_result["label"] == "likely_ai"
