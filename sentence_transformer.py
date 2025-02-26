import json
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score
import sentence_transformers as st
from transformers import pipeline

# Pretrained models based on search engine recommendations for a task involving sentence transformers
# The numbers on the right are the average cosine similarities among all sentences in the chosen groups,
# all-mpnet-base-v2 had better performance on average and was chosen for that reason.
model_name = "sentence-transformers/all-mpnet-base-v2"  #  0.2627643154727088 0.3482766144805484 0.32765392826663126
sp_model_name = "sentiment-analysis"
# model_name = "sentence-transformers/all-MiniLM-L6-v2"   #  0.2306225241886245 0.2881481038199531 0.23493114097250833

class SentenceTransformer():
    def __init__(self, model_name, sentiment_pipeline):
        self.model = st.SentenceTransformer(model_name)
        self.sp_pipeline = pipeline(sentiment_pipeline)

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
    
    def train(self,model, train_data, val_data, epochs=5, learning_rate=1e-5, weight_decay = 0.01): 
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
        criterion_classification = CrossEntropyLoss()
        criterion_ner = CrossEntropyLoss(ignore_index=-100)  # Ignore padding in NER loss

        # Setting the device to GPU if exists
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        model.to(device)

        best_val_loss = float('inf')

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for batch in train_data:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                classification_labels = batch["classification_labels"].to(device)
                ner_labels = batch["ner_labels"].to(device)

                optimizer.zero_grad()
                classification_logits, ner_logits = model(input_ids, attention_mask)

                loss_classification = criterion_classification(classification_logits, classification_labels)
                loss_ner = criterion_ner(ner_logits.view(-1, ner_logits.size(-1)), ner_labels.view(-1))

                total_loss = loss_classification + loss_ner  # You can add weights here if needed
                train_loss += total_loss.item()

                total_loss.backward()
                optimizer.step()

            avg_train_loss = train_loss / len(train_data)

            # Validation Loop
            model.eval() # Set model to eval mode
            val_loss = 0.0
            all_classification_preds = []
            all_classification_labels = []
            all_ner_preds = []
            all_ner_labels = []

            with torch.no_grad():
                for batch in val_data:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    classification_labels = batch["classification_labels"].to(device)
                    ner_labels = batch["ner_labels"].to(device)

                    classification_logits, ner_logits = model(input_ids, attention_mask)

                    loss_classification = criterion_classification(classification_logits, classification_labels)
                    loss_ner = criterion_ner(ner_logits.view(-1, ner_logits.size(-1)), ner_labels.view(-1))

                    total_loss = loss_classification + loss_ner
                    val_loss += total_loss.item()

                    # Metrics Calculation
                    classification_preds = torch.argmax(classification_logits, dim=-1).cpu().numpy()
                    classification_labels_batch = classification_labels.cpu().numpy()

                    ner_preds = torch.argmax(ner_logits, dim=-1).cpu().numpy()
                    ner_labels_batch = ner_labels.cpu().numpy()

                    all_classification_preds.extend(classification_preds)
                    all_classification_labels.extend(classification_labels_batch)

                    all_ner_preds.extend(ner_preds)
                    all_ner_labels.extend(ner_labels_batch)


            avg_val_loss = val_loss / len(val_data)

            # Metrics
            classification_accuracy = accuracy_score(all_classification_labels, all_classification_preds)

            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Classification Accuracy: {classification_accuracy:.4f}")

            # Model Saving
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "multi_task_model.pth")

    def get_sentiment(self, sentence):
        sentiment = self.sp_pipeline(sentence)[0]
        return sentiment['label']


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

def task_two_A(sentence_transformer, sentences_dict):
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

def task_two_B(sentence_transformer):
    
    sample_sentences = [
        "The universe is so big.",
        "Integration is a core concept in calculus.",
        "Trees provide oxygen.",
        "I dislike dragonfruit and coconut."
    ]

    for sentence in sample_sentences:
        sentiment = sentence_transformer.get_sentiment(sentence)
        print(f"{sentence} : sentiment '{sentiment}'")

def main():
    sentence_transformer = SentenceTransformer(model_name, sp_model_name)
    with open("sentence_classes.json", 'r') as fp:
        sentences_dict = json.load(fp)
    task_one(sentence_transformer, sentences_dict)
    task_two_A(sentence_transformer, sentences_dict)
    task_two_B(sentence_transformer)

if __name__ == '__main__':
    main()