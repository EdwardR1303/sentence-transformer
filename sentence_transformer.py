import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util

# Pretrained models based on search engine recommendations for a task involving sentence transformers
# The numbers on the right are the average cosine similarities among all sentences in the chosen groups,
# all-mpnet-base-v2 had better performance on average and was chosen for that reason.
model_name = "sentence-transformers/all-mpnet-base-v2"  #  0.2627643154727088 0.3482766144805484 0.32765392826663126
# model_name = "sentence-transformers/all-MiniLM-L6-v2"   #  0.2306225241886245 0.2881481038199531 0.23493114097250833


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_mean_pooling_sentence_embedding(sentence):
    """ While there are other ways to do embeddings, such as Max pooling, CLS tokens, weighted pooling, etc,
    I chose to use mean pooling due to its simplicy and quick efficiency. For bigger data sets, more nuanced
    approaches would make more sense.
    """
    inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings

with open("sentence_classes.json", 'r') as fp:
   sentences = json.load(fp)

similarities = {key: [] for key in sentences.keys()}
for label, sentences in sentences.items():
    embeddings = [get_mean_pooling_sentence_embedding(sentence) for sentence in sentences]

    for i in range(len(embeddings)-1):
        for x in range(i+1, len(embeddings)):
            similarity = util.cos_sim(embeddings[i], embeddings[x]).item()
            similarities[label].append((f"{i},{x}", similarity))
    avg_cos_sim = sum([x[1] for x in similarities[label]]) / len(similarities[label])
    print(f"Cosine Similarity for top 5 matches for label '{label}':" )
    print(*sorted(similarities[label], key=lambda item: -item[1])[:5], sep='\n')
    print(f"Average Cosine Similarty for label group '{label}': {avg_cos_sim}")