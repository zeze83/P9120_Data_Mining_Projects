# P9120_Data_Mining_Projects

This is a concept summary for the **P9120 Data Mining** coursework. 

---

## Lecture 1 - Introduction to Statistical Learning and Data Mining

1. **Overview of Statistical Learning**
  - Introduction to data mining and machine learning.
  - The relationship between supervised and unsupervised learning.
  - Key concepts: training data, validation data, testing data.
  - Supervised learning: Classification and Regression.
  - Unsupervised learning: Clustering.
2. **Basic Terminology**
  - Data matrix, responses, and features.
  - The concept of training error vs. test error.
  - Bias-variance tradeoff.
  - Tools for visualization: Matplotlib, Seaborn, Plotly.
  
---

## Lecture 2 - Linear Regression

1. **Ordinary Least Squares (OLS) Regression**
  - Simple and multiple linear regression models.
  - Fitting the model and minimizing the residual sum of squares.
  - Matrix formulation of OLS.
	- Evaluating model fit using R^2.
2. **Bias-Variance Tradeoff in Linear Models**
  - Overfitting vs. underfitting.
  - The importance of regularization.
3. **Evaluation Metrics**
   - Accuracy, precision, recall, F1 score, ROC curve, and AUC.

---

## Lecture 3 - Classification Methods

1. **Logistic Regression**
  - The logistic function and odds ratios.
  - Implementing binary classification with logistic regression.
  - Model fitting using maximum likelihood estimation.
2. **Decision Trees**
  - The concept of decision nodes and leaf nodes.
  - Recursive partitioning.
  - Overfitting and pruning.
3. **Hyperparameter Tuning**
  - K-fold cross-validation and leave-one-out cross-validation.
  - Grid search and random search for hyperparameter optimization.

---

## Lecture 4 - Supervised Learning(SVM)
1. **Linear Models**
   - Linear regression and logistic regression.
   - Regularization: Lasso and Ridge regression.
2. **SVM for Classification**
  - Maximum margin classification and the kernel trick.
  - Hard vs. soft margin SVMs.
  - Implementing SVM using different kernels.
3. **Multiclass SVMs**
  - One-vs-One (OvO) and One-vs-All (OvA) strategies.
4. **Ensemble Methods**
   - Random Forests and Gradient Boosting.

---

## Lecture 5 - Unsupervised Learning
	
1. **K-Means Clustering**
  - Understanding the K-means algorithm.
  - Evaluating clustering performance using inertia.
  - Choosing the right number of clusters with the elbow method.
2. **Principal Component Analysis (PCA)**
  - Dimensionality reduction through PCA.
  - The role of eigenvectors and eigenvalues.
  - Visualizing data in lower dimensions.
3. **Anomaly Detection**
   - Techniques for detecting outliers in data.

---

## Lecture 6 - Regularization and Feature Selection
	
1. **Ridge and Lasso Regression**
  - Differences between Ridge, Lasso, and Elastic Net.
  - The importance of regularization for feature selection and model complexity.
2. **Principal Component Regression (PCR)**
  - Combining PCA with regression for better prediction models.
3. **Activation Functions**
   - Sigmoid, ReLU, and tanh activation functions.
   - Why activation functions are important in deep learning.
4. **Deep Learning Frameworks**
   - Keras, TensorFlow, and PyTorch.

---

## Lecture 7 - Convolutional Neural Networks (CNNs)
1. **CNN Architecture**
   - Convolutional layers, pooling layers, and fully connected layers.
   - Application of CNNs in image recognition.

2. **Training CNNs**
   - How to train CNNs for image classification tasks.

3. **Transfer Learning**
   - Using pretrained models like VGG, ResNet for new tasks.

---

## Lecture 8 - Recurrent Neural Networks (RNNs) and LSTMs
1. **Introduction to RNNs**
   - Sequential data modeling using RNNs.
   - Training RNNs for time series prediction.

2. **LSTM and GRU**
   - Long Short-Term Memory networks for solving vanishing gradient problem.
   - Gated Recurrent Units and their differences from LSTMs.

3. **Applications of RNNs**
   - Language modeling, machine translation, and speech recognition.

---

## Lecture 9 - Natural Language Processing (NLP)
1. **Text Preprocessing**
   - Tokenization, stopwords, stemming, and lemmatization.

2. **Word Embeddings**
   - Word2Vec, GloVe, and FastText.
   - Embedding words into vectors for NLP tasks.

3. **NLP with Deep Learning**
   - Using RNNs and transformers for text generation and classification.

---

## Lecture 10 - Advanced Reinforcement Learning
1. **Contextual Bandit Problems**
   - Multi-armed bandits with contextual information.
   - Policy learning and regret minimization.

2. **Full Reinforcement Learning: Q-Learning, Deep Q Networks**
   - Solving sequential decision-making problems in unknown environments.
   - Implementing Q-learning and model-free reinforcement learning.

---

## Lecture 11 - Advanced Reinforcement Learning
1. **Contextual Bandit Problems**
   - Multi-armed bandits with contextual information.
   - Policy learning and regret minimization.

2. **Full Reinforcement Learning: Q-Learning, Deep Q Networks**
   - Solving sequential decision-making problems in unknown environments.
   - Implementing Q-learning and model-free reinforcement learning.

---

## Lecture 12 - Natural Language Processing (NLP) and Deep Learning
1. **Word Embeddings and Word2Vec**
   - Introduction to word embeddings and vector space models.
   - Implementing Word2Vec for semantic similarity and word relationships.

2. **Recurrent Neural Networks (RNNs)**
   - Understanding RNNs for sequence modeling.
   - Applications of RNNs in NLP tasks like text generation and sentiment analysis.

3. **Transformers and Attention Mechanisms**
   - Introduction to transformer models.
   - The significance of self-attention in NLP.

4. **Pretrained Language Models (BERT, GPT)**
   - Fine-tuning large pretrained language models for specific tasks.
   - Using BERT and GPT for language understanding and generation.

---

## Lecture 13 - Computer Vision and Image Processing
1. **Convolutional Neural Networks (CNNs)**
   - Introduction to CNNs and their use in image recognition.
   - CNN architecture: convolutional layers, pooling, and fully connected layers.

2. **Transfer Learning in Computer Vision**
   - Leveraging pretrained models for transfer learning.
   - Fine-tuning models like ResNet and VGG for new image classification tasks.

3. **Object Detection and Segmentation**
   - Overview of object detection and segmentation tasks in computer vision.
   - Implementing algorithms like YOLO (You Only Look Once) for real-time object detection.

4. **Generative Adversarial Networks (GANs)**
   - Introduction to GANs and their applications in image generation.
   - Understanding the architecture and training process of GANs.