**Day 3: Deep Learning Fundamentals (Extended)**

Below is an expanded set of questions, challenges, and prompts for Day 3, focusing on foundational neural network concepts. By the end of this day, you should feel confident discussing how neural nets work, key architectures (CNNs, RNNs, Transformers), and common strategies for training and tuning deep learning models.

---

## 1. Building a Simple Neural Network

### Goal

Revisit the basics of forward and backward propagation, ensuring you can conceptually build and train a small neural network.

### Challenges & Prompts

1. **Multilayer Perceptron (MLP)**

   - Outline a simple feed-forward network with a few hidden layers.
   - Manually track a small subset of weights through a forward pass and a backward pass to solidify the concept of partial derivatives.
   - Compare a manual implementation (or pseudo-implementation) with how modern frameworks (PyTorch, TensorFlow) handle autograd.

2. **MNIST or Simple Regression Task**
   - If you have a GPU or a lightweight environment, run a small experiment training an MLP on MNIST or a single hidden-layer network for a regression dataset.
   - Observe how the loss changes over epochs, and note where your model might overfit.

### Questions to Answer

- In your own words, how does backpropagation compute gradients?
- Why do activation functions like ReLU help mitigate vanishing gradients compared to sigmoids?
- What are some techniques to prevent overfitting (dropout, weight decay, early stopping, etc.)?

---

## 2. CNN or NLP Example (Pick One)

### Goal

Dive deeper into either computer vision with CNNs or NLP with RNNs/Transformers to cover modern use cases.

### A. CNN Challenge (Vision-Focused)

1. **2–3 Layer CNN**
   - Conceptualize how a simple CNN architecture (conv → pool → fully connected) works for image classification tasks (like MNIST or CIFAR-10).
   - Note how convolution filters learn spatial features.
2. **Pooling**
   - Explain how max pooling or average pooling helps reduce spatial dimensionality.
   - How does pooling introduce translation invariance?

#### Questions to Answer (CNN)

- What does a convolution operation do, and why is it more efficient than a fully connected layer for images?
- How does padding affect the output size of a CNN layer?
- Why might you choose a certain filter size (3x3, 5x5, etc.)?

### B. RNN/Transformer Challenge (NLP-Focused)

1. **Simple LSTM or Tiny Transformer**
   - Outline how an LSTM cell processes sequences (e.g., one token at a time).
   - If you choose a Transformer, focus on self-attention and how queries, keys, and values interact.
2. **Text Classification**
   - Collect a small text dataset (IMDB movie reviews, or your own text) for binary classification.
   - Consider how you’d tokenize, embed words, and feed sequences into an LSTM or a Transformer.

#### Questions to Answer (RNN/Transformer)

- What’s the difference between LSTM and GRU cells in terms of gates and parameters?
- How does an attention mechanism help a Transformer model capture long-range dependencies better than a basic RNN?
- What are positional encodings and why are they needed?

---

## 3. Hyperparameter Tuning & Training

### Goal

Learn how hyperparameters (learning rate, batch size, number of layers/units) can drastically affect neural network performance.

### Challenges & Prompts

1. **Learning Rate Experiments**
   - Adjust learning rate (e.g., 0.1, 0.01, 0.001) in your chosen MLP or CNN model and observe differences in training stability and speed.
   - Check for exploding gradients or stalled training.
2. **Batch Sizes & Optimizers**
   - Compare a small batch vs. large batch approach.
   - Switch between SGD and Adam optimizers, noting how quickly losses converge.
3. **Regularization & Batch Normalization**
   - Try adding dropout or weight decay (L2 regularization) to curb overfitting.
   - Insert a batch normalization layer in your CNN or MLP and see how it affects training dynamics.

### Questions to Answer

- How do you pick an initial learning rate, and why might you employ a learning rate schedule (step decay, exponential decay)?
- What is batch normalization actually doing under the hood, and why can it speed up training?
- If your training loss keeps decreasing but validation loss plateaus or increases, what does that indicate (and how might you respond)?

---

## 4. Connecting It Back to E-Commerce (Optional Brainstorm)

While experimenting with deep learning, consider how these methods might apply to your e-commerce context:

1. **Product Image Classification**
   - CNNs for detecting product categories or flags (e.g., adult content filtering).
2. **Chatbot or NLP for Product Descriptions**
   - Using an LSTM or a Transformer to classify user queries or product descriptions.
   - Handling synonyms, brand names, and domain-specific jargon.
3. **User Behavior Prediction**
   - A simpler sequential model to track user actions over time (page views, clicks), predicting future purchases.

---

## Day 3 Action Items Recap

By the end of Day 3, aim to have:

1. Sketched or noted down the core steps of building a small MLP, CNN, or RNN/Transformer.
2. Familiarized yourself with forward/backprop at a conceptual level.
3. Experimented (or at least read up on) training a small model (MNIST, text classification, etc.) and observed typical pitfalls (overfitting, vanishing/exploding gradients).
4. Asked yourself deeper questions on hyperparameters, optimizers, and regularization strategies.

Keep a record of your observations and any “aha!” moments—this will help you articulate lessons learned in an interview setting.

---

> **Next Steps (Preview for Day 4):**  
> You’ll move onto advanced topics—semantic search, chatbots, and big data. These will tie together classical ML with deep learning concepts in the context of large-scale data engineering and e-commerce use cases.

Good luck with Day 3—solidifying your deep learning fundamentals is key to tackling real-world applications like semantic search, recommendation systems, and more!
