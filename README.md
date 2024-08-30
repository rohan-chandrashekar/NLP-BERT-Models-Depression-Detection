# **Mental Health Text Classification Using BERT**

## **Project Overview**

This repository hosts the code and dataset for a research project focused on the classification of mental health-related text using the BERT (Bidirectional Encoder Representations from Transformers) model. The project demonstrates the application of advanced NLP techniques for the detection and categorization of textual data associated with mental health issues. This model has the potential to aid in early identification and intervention by analyzing user-generated content on social platforms or forums.

## **Objective**

The primary goal of this project is to build a robust NLP model capable of classifying text data into mental health-related categories. Leveraging BERT's deep understanding of language context, the model provides high accuracy in distinguishing between neutral and potentially concerning mental health content.

## **Dataset Description**

The dataset utilized in this project, `mental_health.csv`, contains a collection of text data labeled with binary annotations:

- **text**: Contains various user-generated content, such as social media posts or comments.
- **label**: Binary labels (`0` for neutral, `1` for mental health concern), indicating the presence of mental health-related content.

The dataset comprises **27,977** entries, each featuring a unique text sample paired with a corresponding label. This dataset provides a realistic benchmark for evaluating the performance of the BERT-based classification model.

## **Technical Architecture**

### **1. Data Preprocessing**

The dataset undergoes several preprocessing steps to prepare it for input into the BERT model:

- **Tokenization**: The BERT tokenizer, `BertTokenizer`, is employed to convert the text into tokens that the model can process. This includes splitting the text into subwords, padding, and creating attention masks.
- **Dataset Class**: A custom PyTorch `Dataset` class, `MentalHealthDataset`, is defined to handle the tokenization and formatting of text samples and labels. This class is crucial for efficient batching and data loading during training.

### **2. Model Setup**

The model architecture is built upon BERT's sequence classification capabilities:

- **BERT for Sequence Classification**: `BertForSequenceClassification` is used as the core model, fine-tuned on the mental health dataset. The model architecture includes:
  - **Input Layer**: Tokenized text data.
  - **BERT Encoder**: Processes the input to capture contextual information across all tokens.
  - **Classification Head**: A dense layer with a softmax activation function for binary classification.

### **3. Training and Optimization**

The model training pipeline includes:

- **Data Splitting**: The dataset is split into training and validation sets using `random_split`, ensuring the model is evaluated on unseen data.
- **Data Loaders**: PyTorch `DataLoader` is employed for efficient batching and shuffling during training.
- **Optimization**: The model is trained using the `AdamW` optimizer, which incorporates weight decay to prevent overfitting. The learning rate and other hyperparameters are tuned to achieve optimal performance.

### **4. Evaluation Metrics**

The model's performance is rigorously evaluated using the following metrics:

- **Accuracy**: Measures the overall correctness of the model's predictions.
- **Precision**: Evaluates the proportion of true positive predictions among all positive predictions.
- **Recall**: Assesses the model's ability to identify all relevant instances of mental health-related text.
- **F1-Score**: The harmonic mean of precision and recall, providing a balanced metric for classification performance.

These metrics are computed for both the training and validation datasets to monitor the model's effectiveness and generalization capabilities.

## **Research Contributions**

This work contributes to the growing field of mental health informatics by:

1. **Applying BERT to Mental Health Classification**: Demonstrating the effectiveness of transfer learning with BERT for identifying mental health-related content in textual data.
2. **Developing a Reproducible Pipeline**: Providing a comprehensive and reproducible pipeline that can be adapted for similar classification tasks in other domains.
3. **Highlighting Interpretability**: Emphasizing the importance of interpretability in NLP models, especially in sensitive applications like mental health.

## **Future Work**

The current model lays the foundation for further research in several areas:

- **Model Fine-Tuning**: Explore the impact of fine-tuning BERT on domain-specific corpora to enhance model performance.
- **Multi-Class Classification**: Extend the binary classification task to a multi-class problem to differentiate between various types of mental health concerns.
- **Explainability**: Integrate explainability methods (e.g., LIME, SHAP) to make the model's decisions more interpretable, particularly for clinical applications.
