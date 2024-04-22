# Smart-HealthCare-Medical_ChatBot_Application

Smart HealthCare Navigator: Leveraging Open Source LLMS for User Consultation and Disease Classification

# Introduction

This repository contains the codebase for an  "AI-powered healthcare assistant for personalized medical advice with the functionality of either Text or speech input". It leverages advanced Natural Language Processing, and state-of-the-art models to deliver accurate and personalized responses tailored to users' health concerns.

**Installation of Dependencies**:
   
   ```bash
   pip install -r requirements.txt
   ```
---

# Disease Classification and Medical QA System

This section of the project involves disease classification using machine learning models and a medical question-answering system using the BioGPT language model.


1. **Data Preprocessing**: The dataset (Symptom2Disease.csv) containing symptoms and corresponding disease labels is preprocessed. Text cleaning techniques such as removing punctuation and stop words are applied.

2. **Text Classification**: Several machine learning models are trained on the preprocessed text data to classify diseases based on symptoms. The models include Random Forest, XGBoost, AdaBoost, Gradient Boosting, and Support Vector Classifier (SVC). Ensemble learning is also employed to combine the predictions of multiple models.

3. **Recurrent Neural Network (RNN) Models**: Two RNN models, one using GRU and the other using LSTM, are trained on the disease classification task. These models utilize word embeddings to process the text input and predict the disease label.

4. **Medical Question-Answering System**: The BioGPT language model is fine-tuned on medical text data to create a question-answering system. Given a medical query or symptom description, the system generates appropriate responses using the trained model.

5. **Speech Recognition**: The system includes functionality for transcribing audio files containing patient symptom descriptions. This transcribed text is then used as input for disease classification and question-answering.

---

# Fine-tuning LLAMA Model on PubMedQA Dataset

The LLAMA model is a powerful language model developed by NousResearch, specifically tailored for understanding medical text and generating accurate responses to medical queries. Fine-tuning the LLAMA model on a specific dataset like PubMedQA involves training the model on the new dataset to adapt its parameters to the specific task or domain.

## Fine-Tuning Process

1. **Dataset Preparation**: The PubMedQA dataset is loaded using the `datasets` library from Hugging Face. This dataset contains question-answer pairs extracted from PubMed articles.

2. **Model Selection**: The LLAMA model (`NousResearch/Llama-2-7b-chat-hf`) is chosen as the base model for fine-tuning. This model is pre-trained on a large corpus of medical text and has demonstrated strong performance in understanding medical language.

3. **Tokenization**: The selected dataset is tokenized using the LLAMA tokenizer (`AutoTokenizer`) provided by Hugging Face. Tokenization involves breaking down the text into individual tokens, which are then encoded as numerical representations suitable for input to the model.

4. **Data Preparation**: The dataset is processed to merge the question and answer pairs into single text sequences. This step involves concatenating the question and its corresponding long answer from the PubMed article.

5. **Model Configuration**: Various model configurations are set up, including the use of QLoRA (Question-Answering with Logic Relation Attention) architecture, BitsAndBytes quantization, and SFT (Soft Fine-Tuning) training strategy.

6. **Training**: The model is trained using the SFTTrainer provided by the `trl` library. This involves iterating through the training dataset, adjusting the model's parameters based on the loss computed from comparing its predictions to the ground truth labels.

## Quantization Methods

### BitsAndBytes Quantization

- **Use of 4-bit Precision**: The fine-tuning process utilizes 4-bit precision for loading the base model (`use_4bit=True`). This reduces the memory footprint and speeds up inference while maintaining performance.

- **Quantization Type**: The 4-bit quantization type is specified as "nf4" (`bnb_4bit_quant_type="nf4"`), which stands for nested floating-point 4-bit quantization. This method optimizes the quantization process to achieve a balance between model size and accuracy.

- **Nested Quantization**: Nested quantization (`use_nested_quant=False`) is disabled in this implementation. Nested quantization applies double quantization, further reducing the model's precision but potentially sacrificing some accuracy.

### LoRA Implementation

- **Logic Relation Attention**: LoRA is a novel attention mechanism introduced in the LLAMA model. It enhances the model's ability to capture logic relationships between tokens in the input sequence, particularly beneficial for question-answering tasks where understanding context is crucial.

- **LoRA Configuration**: The LoRA configuration (`LoraConfig`) includes parameters such as the attention dimension (`lora_r`), alpha parameter for scaling (`lora_alpha`), and dropout probability (`lora_dropout`). These parameters are carefully chosen to optimize the performance of the LoRA mechanism for the given task.

## Usage

1. **Run Fine-Tuning Script**:
   After cloning the repository, execute the fine-tuning script (`fine_tune_llama.py`) to start the fine-tuning process. This script will load the PubMedQA dataset, configure the model, and train it using the specified parameters.
   
3. **Generate Answers**: After fine-tuning, the model can generate answers to medical questions. Use the provided `generate_answer` function to input symptoms or medical queries and receive responses from the fine-tuned LLAMA model.

---

## Instructions

# Implementation of RAG using Langchain and Google Gemini

This script utilizes the RAG (Retrieval-Augmented Generation) architecture to generate responses to user queries in the context of healthcare. The user is also able to give speech input. Here's how it incorporates RAG:

1. **Retrieval of Relevant Documents**: 
   - The script loads medical data from JSON files (`iCliniq.json` and `GenMedGPT-5k.json`) and concatenates them to form a context.
   - It then splits the concatenated context into smaller texts to facilitate efficient retrieval.

2. **Document Retrieval**:
   - A vector index is constructed using Chroma to retrieve relevant documents based on user queries.
   - The retrieved documents serve as contextual information for generating responses.

3. **Generation of Responses**:
   - A prompt template is constructed that includes placeholders for context and user queries. 
   - It utilizes Langchain's `load_qa_chain` function to set up a question-answering chain (specifically the "stuff" chain) with the specified prompt.
   - When a user query is input, the script retrieves relevant documents, fills in the prompt template with the retrieved context and user query, and generates an answer using the RAG model.
   - The generated answer is then displayed to the user.

## Usage

1. Make sure all dependencies are installed.
2. Run the Streamlit app:

```bash
streamlit run streamlit_Gemini_RAG.py
```

3. The app will launch in your default web browser. Enter your health issues into the chat input.
4. The AI chat doctor will respond with personalized suggestions and advice based on the provided information.

## Files

- `streamlit_Gemini_RAG.py`: Contains the main code for the Streamlit app.
- `iCliniq.json`: JSON file containing medical data.
- `GenMedGPT-5k.json`: Additional JSON file containing medical data.

## Configuration

- Update the `GOOGLE_API_KEY` variable in `streamlit_Gemini_RAG.py` with your own Google API key.
- Ensure that the file paths for `iCliniq.json` and `GenMedGPT-5k.json` are correctly set in the code.

# Evaluation of Fine-tuned LLAMA and Microsoft-BioGPT

The assessment of the fine-tuned LLAMA and Microsoft-BioGPT models entails an examination across various evaluation metrics including BLEU, ROUGE, BERT Score, Novelty, Diversity, and Levenshtein distance. The fine-tuned LLAMA model exhibits comparable performance to BioGPT in BERT Score and Levenshtein distance measures, while demonstrating superior performance in terms of Diversity metric when compared to BioGPT.

# Combined Deployment

This notebook contains the combined deployment of Fine-tuned LLAMA, RAG using Gemini and Prescription Parsing using PaddleOCR.


## Disclaimer

This application is for educational and demonstration purposes only. It should not be used as a substitute for professional medical advice or diagnosis. Always consult with a qualified healthcare provider for accurate medical information and treatment.


