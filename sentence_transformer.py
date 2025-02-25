import json
import torch
import numpy as np
import sentence_transformers as st

# Pretrained models based on search engine recommendations for a task involving sentence transformers
# The numbers on the right are the average cosine similarities among all sentences in the chosen groups,
# all-mpnet-base-v2 had better performance on average and was chosen for that reason.
model_name = "sentence-transformers/all-mpnet-base-v2"  #  0.2627643154727088 0.3482766144805484 0.32765392826663126
# model_name = "sentence-transformers/all-MiniLM-L6-v2"   #  0.2306225241886245 0.2881481038199531 0.23493114097250833

class SentenceTransformer():
    def __init__(self, model_name):
        self.model = st.SentenceTransformer(model_name)

    def encode_sentences(self, sentences):
        return self.model.encode(sentences)

    def classify_sentence(self, sample_sentence, labeled_embeddings, min_cos_sim=0.2):
        """For this classification function, I chose to add a parameter with minimum cosine similarity in order
        to classify a sentence. If the sample sentence is not at least 'min_cos_sim' similar, then a label of
        None is returned"""

        best_label = None
        top_similarity = -1
        new_embedding = self.model.encode(sample_sentence)

        for label, embeddings in labeled_embeddings.items():
            similarities = st.util.cos_sim(new_embedding, embeddings)
            avg_similarity = torch.mean(similarities)

            if avg_similarity > top_similarity:
                top_similarity = avg_similarity
                best_label = label

        return best_label if top_similarity >= min_cos_sim else None


def task_one(sentence_transformer, sentences_dict):

    similarities = {key: [] for key in sentences_dict.keys()}
    for label, sentences in sentences_dict.items():
        embeddings = [sentence_transformer.encode_sentences(sentence) for sentence in sentences]
        # labeled_embeddings[label] = sentence_transformer.encode_sentences(sentences)
        for i in range(len(embeddings)-1):
            for x in range(i+1, len(embeddings)):
                similarity = st.util.cos_sim(embeddings[i], embeddings[x]).item()
                similarities[label].append((f"{i},{x}", similarity))
        avg_cos_sim = sum([x[1] for x in similarities[label]]) / len(similarities[label])
        print(f"Cosine Similarity for top 5 matches for label '{label}':" )
        print(*sorted(similarities[label], key=lambda item: -item[1])[:5], sep='\n')
        print(f"Average Cosine Similarty for label group '{label}': {avg_cos_sim}")

def task_two(sentence_transformer, sentences_dict):
    sample_sentences = [
        "The universe is so big.",
        "Integration is a core concept in calculus.",
        "Trees provide oxygen.",
        "I dislike dragonfruit and coconut."
    ]
    
    labeled_embeddings = {}
    for label, sentences in sentences_dict.items():
        labeled_embeddings[label] = sentence_transformer.encode_sentences(sentences)
    for sentence in sample_sentences:
        label = sentence_transformer.classify_sentence(sentence, labeled_embeddings)
        print(f"The sentence '{sentence}' is associated with the label '{label}'")

def main():
    sentence_transformer = SentenceTransformer(model_name)
    with open("sentence_classes.json", 'r') as fp:
        sentences_dict = json.load(fp)
    task_one(sentence_transformer, sentences_dict)
    task_two(sentence_transformer, sentences_dict)

main()