import string
from collections import Counter
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import whisper
import nltk
import string
import torch
import torch.nn as nn
import torchtext
import gradio as gr
import librosa
import speech_recognition as sr
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
nltk.download('stopwords')
nltk.download('punkt')
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline


stop_words = set(stopwords.words('english'))

def clean_text(sent):
    sent = sent.translate(str.maketrans('','',string.punctuation)).strip()

    stop_words = set(stopwords.words('english'))
    words = word_tokenize(sent)
    words = [word for word in words if word not in stop_words]

    return " ".join(words).lower()

class DiseaseDataset(torch.utils.data.Dataset):
    def __init__(self, symptoms,labels):
        self.symptoms = symptoms
        self.labels= torch.tensor(labels.to_numpy())
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.symptoms[idx]
        label = self.labels[idx]

        # Convert the text to a sequence of word indices
        text_indices = [vocab[word] for word in text.split()]

        # padding for same length sequence
        if len(text_indices)<max_words:
            text_indices = text_indices + [1]*(max_words - len(text_indices))

        return torch.tensor(text_indices), label
    
class RNNModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim,num_classes,drop_prob,num_layers=1,bidir=False,seq="lstm"):
        super(RNNModel, self).__init__()
        self.seq = seq
        self.bidir_f = 2 if bidir else 0
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        if seq=="lstm":
            self.rnn = torch.nn.LSTM(embedding_dim, hidden_dim,
                                     num_layers=num_layers,
                                     batch_first=True,
                                     bidirectional=bidir)
        else:
            self.rnn = torch.nn.GRU(embedding_dim, hidden_dim,
                                 num_layers=num_layers,
                                 batch_first=True,
                                bidirectional=bidir)

        self.dropout = torch.nn.Dropout(drop_prob) #dropout layer
        self.fc = torch.nn.Linear(hidden_dim*self.bidir_f, num_classes) # fully connected layer

    def forward(self, text_indices):
        # Embed the text indices
        embedded_text = self.embedding(text_indices)
        rnn_output,hidden_states = self.rnn(embedded_text)
        # Take the last output of the RNN
        last_rnn_output = rnn_output[:, -1, :]
        x = self.dropout(last_rnn_output)
        x = self.fc(x)

        return x
    

def train(model,num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #choose device for training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.cuda()
    print("IS CUDA: ",next(model.parameters()).is_cuda)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for data in train_loader:
            inputs,labels = data
            inputs,labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc = (labels == outputs.argmax(dim=-1)).float().mean().item()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in val_loader:
                inputs,labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = outputs.argmax(-1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = (labels == outputs.argmax(dim=-1)).float().mean().item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {val_loss}, Train Accuracy: {acc:.2f}  Val Accuracy: {accuracy:.2f}')

def make_pred(model, text):
    text = clean_text(text)
    text_indices = [vocab[word] for word in text.split()]

    if len(text_indices) < max_words:
        text_indices = text_indices + [1] * (max_words - len(text_indices))
    text_indices = torch.tensor(text_indices).cuda()

    model.eval()

    # Forward pass
    with torch.no_grad():
        pred_probs = torch.softmax(model(text_indices.unsqueeze(0)), dim=1)

    # Get the top 5 probabilities and their indices
    top5_probs, top5_indices = torch.topk(pred_probs.squeeze(), 5)

    for prob, idx in zip(top5_probs.tolist(), top5_indices.tolist()):
        print(f"Probability of {idx2dis[idx]}: {prob:.4f}")

    # Print predicted class
    predicted_class = pred_probs.argmax(1).item()
    #print(f"Predicted Disease: {idx2dis[predicted_class]}")
    return idx2dis[predicted_class]


def transcribe_audio(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=None)
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    print(transcriber({"sampling_rate": sr, "raw": y})["text"])
    return transcriber({"sampling_rate": sr, "raw": y})["text"]

def answer_question(prompt,temperature=0.1,top_p=0.75,top_k=40,num_beams=2,**kwargs):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")
    model.to("cuda")
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=600,
            eos_token_id=tokenizer.eos_token_id

        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    return output.split(" Response:")[1]

def answer_GRU_Classification(prompt):
    return make_pred(model_gru, prompt)

def answer_LSTM_Classification(prompt):
  return make_pred(model_lstm, prompt)


def answer_BioGPT(prompt):
    prompt_template = """
    Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    As a nutritionist, provide dietary advice based on the patient's description.

    ### Input:
    {}
    ### Response:
    """.format(prompt)
    return answer_question(prompt_template)

def show_Disclaimer_message(prompt):
    disclaimer_message = "This application provides general predictions. For accurate medical advice, please consult with a qualified healthcare professional."
    return disclaimer_message

def wrapper_function(audio_data):
    gr.Warning("This application provides general predictions. For accurate medical advice, please consult with a qualified healthcare professional.")
    text = transcribe_audio(audio_data)
    print(text)
    return [answer_GRU_Classification(text), answer_LSTM_Classification(text), answer_BioGPT(text)]

    

df = pd.read_csv("/content/Symptom2Disease.csv")
df.drop("Unnamed: 0",inplace=True,axis=1)
df
df["text"] = df["text"].apply(clean_text)

diseases = df["label"].unique()

idx2dis = {k:v for k,v in enumerate(diseases)}
dis2idx = {v:k for k,v in idx2dis.items()}

df["label"] = df["label"].apply(lambda x: dis2idx[x])

X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(max_features=1500)

tfidf_train = tfidf_vectorizer.fit_transform(X_train).toarray()
tfidf_test = tfidf_vectorizer.transform(X_test).toarray()

#Comapring different model performance

rf_classifier = RandomForestClassifier()
xgb_classifier = XGBClassifier()
ada_classifier = AdaBoostClassifier()
gb_classifier = GradientBoostingClassifier()
svc_classifier = SVC(probability=True)

ensemble_classifier = VotingClassifier(estimators=[
    ('rf', rf_classifier),
    ('gb', gb_classifier),
    ('svc', svc_classifier)
], voting='soft')

ensemble_classifier.fit(tfidf_train, y_train)
rf_classifier.fit(tfidf_train, y_train)
xgb_classifier.fit(tfidf_train, y_train)
ada_classifier.fit(tfidf_train, y_train)
gb_classifier.fit(tfidf_train, y_train)
svc_classifier.fit(tfidf_train, y_train)

predictions = ensemble_classifier.predict(tfidf_test)
predictions_rf = rf_classifier.predict(tfidf_test)
predictions_xgb = xgb_classifier.predict(tfidf_test)
predictions_ada = ada_classifier.predict(tfidf_test)
predictions_gb = gb_classifier.predict(tfidf_test)
predictions_svc = svc_classifier.predict(tfidf_test)

accuracy = accuracy_score(y_test, predictions)
accuracy_rf = accuracy_score(y_test, predictions_rf)
accuracy_xgb = accuracy_score(y_test, predictions_xgb)
accuracy_ada = accuracy_score(y_test, predictions_ada)
accuracy_gb = accuracy_score(y_test, predictions_gb)
accuracy_svc = accuracy_score(y_test, predictions_svc)
print("Ensemble Classifier Accuracy:", accuracy)
print("RF Classifier Accuracy:", accuracy_rf)
print("Xgb Classifier Accuracy:", accuracy_xgb)
print("Ada Classifier Accuracy:", accuracy_ada)
print("Gb Classifier Accuracy:", accuracy_gb)
print("SVC Classifier Accuracy:", accuracy_svc)

classifiers = ['Ensemble', 'RF', 'XGB', 'AdaBoost', 'GradientBoost', 'SVC']
accuracies = [accuracy, accuracy_rf, accuracy_xgb, accuracy_ada, accuracy_gb, accuracy_svc]

plt.figure(figsize=(10, 6))
plt.bar(classifiers, accuracies, color=['blue', 'orange', 'green', 'red', 'purple', 'brown'])
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different Classifiers')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)

max_words = X_train.apply(lambda x:x.split()).apply(len).max()

counter = Counter()
for text in X_train:
    counter.update(text.split())

vocab = torchtext.vocab.vocab(counter,specials=['<unk>', '<pad>'])

vocab.set_default_index(vocab['<unk>'])



train_dataset = DiseaseDataset(X_train, y_train)
val_dataset = DiseaseDataset(X_test, y_test)

batch_size = 8

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

num_classes = len(np.unique(y_train))
vocab_size = len(vocab)
emb_dim = 256
hidden_dim = 128
drop_prob = 0.4

model_gru = RNNModel(vocab_size, emb_dim, hidden_dim, num_classes, drop_prob, num_layers=1, bidir=True, seq="gru")
model_lstm = RNNModel(vocab_size,emb_dim,hidden_dim,num_classes,drop_prob,num_layers=3,bidir=True, seq="lstm")
model_id = "Narrativaai/BioGPT-Large-finetuned-chatdoctor"

tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large")

model = AutoModelForCausalLM.from_pretrained(model_id)


transcriber = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")






train(model_gru,20)
print("===============================================\n")
train(model_lstm, 20)

prompt = input("Please enter symptoms: ")
answer_lstm = make_pred(model_lstm, prompt)
answer_gru = make_pred(model_gru, prompt)
print("prediction from GRU model: ",answer_gru)
print("===============================================\n")
print("prediction from LSTM model: ", answer_lstm)


answer_gpt = answer_BioGPT(prompt)
print("Answer from BioGPT : ", answer_gpt)

