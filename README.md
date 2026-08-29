# Emotion Classification with BiGRU + FastAPI

> **An end-to-end NLP project that classifies text into six emotions using
a Bidirectional GRU model and exposes the trained model through a
FastAPI application.**

**Live App:** [https://analyzing-sentiments-using-fastapi-1.onrender.com/
](https://analyzing-sentiments-using-fastapi-1.onrender.com)

**Dataset:** [dair-ai/emotion on Hugging Face](https://huggingface.co/datasets/dair-ai/emotion)

**GitHub:** [https://github.com/debaratiD2/analyzing_sentiments_using_FastAPI ](https://github.com/debaratiD2/emotion-classification-BiGRU-FastAPI/) 

**Tutorial that inspired this project:** [Emotion classification / FastAPI tutorial](https://youtu.be/mXW4NzapGhQ?si=0JNV4-fNKQ6QAa5f)

### Example

**Input**

> I am extremely happy today!

**Prediction**

> Joy

## 📰 The Project Journal

### The Beginning — I Didn't Start With a Perfect Plan

I did not begin this project by already knowing how to build an NLP application from scratch. I started by watching a YouTube tutorial from **Sheryians AI School** about emotion classification with deep learning and FastAPI. The tutorial gave me a practical starting point, but as I followed along, I became more interested in understanding **why each step was necessary** rather than simply reproducing the code.

That curiosity gradually turned the tutorial into a learning project of my own. I started experimenting with the preprocessing pipeline, comparing recurrent neural-network architectures, paying attention to class imbalance, saving the trained artifacts, and finally trying to serve the model through FastAPI.

The part I enjoyed most was seeing individual concepts connect into one complete system. **Tokenization** stopped being just a term from an NLP lecture. **Padding** became a real requirement for getting text into a neural network. **Class weights** showed me that the way data is distributed can influence what a model learns. And when the BiGRU finally produced a result I was satisfied with, the numbers felt meaningful because I understood more of the journey behind them.

I have experimented with many datasets and trained and evaluated different models before, but this time, I went one step further. After finishing the training and evaluation, I deployed the model to **Render** and got a live URL that could actually serve predictions. That experience made me understand that building a machine-learning project does not end when the model achieves a satisfactory score. Hosting the model for production brings its own set of challenges—saving and loading the model correctly, handling dependencies, connecting the prediction logic to an API, and making sure the deployed application responds reliably to real requests. For the first time, I truly understood the practical hassle of taking a trained model from a notebook and turning it into something that can actually be hosted and used.

That transition is what this README is really about. It is not just documentation of the final files. It is a journal of what I learned, what confused me, what broke, and what finally worked.

---

### Issue #01 — From Text to Numbers

I started this project because I wanted to understand what actually happens between a sentence such as **“I feel so happy today”** and a machine-learning model making an emotion prediction.

The first lesson was surprisingly fundamental: a neural network cannot directly understand words. Before I could talk about GRU or BiGRU, I had to understand **tokenization**.

I used the `dair-ai/emotion` dataset from Hugging Face. The dataset contains six emotion classes: **sadness, joy, love, anger, fear, and surprise**, with 16,000 training examples, 2,000 validation examples, and 2,000 test examples.

I learned to turn text into integer sequences with a Keras `Tokenizer` and then make those sequences the same length using padding. In this project, I limited the vocabulary to 10,000 words and used a maximum sequence length of 50.

That was one of those moments where a concept that had seemed abstract suddenly became concrete. A sentence was no longer just text. It became a sequence of numbers that a neural network could process.

---

### Issue #02 — The Problem of Imbalance

Once the data was represented numerically, another problem appeared: the emotion classes were not perfectly balanced.

Instead of simply hoping that the model would learn every class equally well, I experimented with **class weights**. I used `compute_class_weight(..., class_weight='balanced', ...)` and passed the resulting weights into model training.

This taught me an important lesson: model performance is not only about choosing a sophisticated architecture. Sometimes the data itself needs attention first.

I also used early stopping, monitoring validation loss and restoring the best weights, rather than blindly training for a fixed number of epochs.

---

### Issue #03 — My First Experiments With Sequence Models

I started comparing recurrent architectures rather than jumping immediately to a complicated model.

I experimented with:

- Simple RNN
- LSTM
- GRU
- Bidirectional GRU (BiGRU)

The early experiments were humbling. The first RNN/LSTM/GRU results were nowhere near the performance I eventually wanted. For example, the notebook records test accuracies around 14.2% for the simple RNN and 13.5% for the LSTM in one of the initial experiments.

That was useful rather than discouraging. It made me realize that trying a model is not the same thing as understanding a model.

**Why BiGRU?** 

The **GRU** became especially interesting to me because it could model sequence information while being relatively simpler than an LSTM. Then I took the next step and experimented with a **Bidirectional GRU (BiGRU)**. Unlike a standard GRU, which processes a sentence sequentially from left to right, a BiGRU processes the sequence in **both directions**—from left to right and from right to left. By capturing information from both directions, the BiGRU was able to understand the context more effectively, which ultimately helped improve the model's performance and gave me a more satisfactory accuracy.

---

### Issue #04 — Why BiGRU Became the Turning Point

The idea behind a bidirectional recurrent model fascinated me. Instead of reading a sequence in only one direction, a BiGRU processes it from both directions, giving the model information about what comes before and after a word in the sequence. The notebook explicitly documents the two directions as left-to-right and right-to-left.

My final architecture was:

```text
Embedding
   ↓
Bidirectional GRU (128 units)
   ↓
Dropout (0.5)
   ↓
Bidirectional GRU (64 units)
   ↓
Dropout (0.5)
   ↓
Dense (6 classes, softmax)
```

The same architecture is rebuilt inside `main.py` before the saved weights are loaded. The deployed model uses a 10,000-word vocabulary, 300-dimensional embeddings, two bidirectional GRU layers, dropout, and a six-class softmax output.

And then came the result that made me genuinely happy: the BiGRU reached approximately **92.95% test accuracy**, with a test loss of about **0.1824** in the notebook evaluation.

It was not a perfect model, but for me it was a very satisfactory result because I could finally see the connection between preprocessing, architecture, training strategy, and performance.

---

## 🧠 What I Actually Learned From the Model

Before this project, terms such as *tokenization*, *padding*, *class weights*, *GRU*, and *bidirectional networks* were mostly things I had read about. After implementing them, they became parts of a **complete pipeline**.

The flow became much clearer:

```text
Raw sentence
    ↓
Text preprocessing
    ↓
Tokenization
    ↓
Integer sequence
    ↓
Padding / truncation to 50 tokens
    ↓
Embedding
    ↓
BiGRU
    ↓
Softmax probabilities
    ↓
Predicted emotion
```

That pipeline is probably the single biggest conceptual takeaway I gained from this project.

---

## The Part That Challenged Me Most: FastAPI

Training the model was challenging, but **turning the trained model into something that could actually answer an HTTP request was a completely different kind of challenge**.

This was the part where I started to feel less like I was only doing machine learning and more like I was building a software system.

I could train a BiGRU in a notebook and get a good score. But getting to the point where I could send text to an API and receive a proper **200 response** after defining the prediction module was genuinely difficult for me.

The challenge was not simply writing a `/predict` endpoint. I had to think about what happens when the server starts, where the model lives, how the tokenizer is loaded, how input text is transformed exactly the same way it was during training, and how the prediction is returned to the client.

That is why the `main.py` implementation became the most valuable part of the project for me.

---

## 🔧 The Lifespan Solution

One of the most important pieces I learned was FastAPI's **lifespan** mechanism.

Instead of loading the model for every request, I reconstruct the BiGRU architecture when the application starts, load the saved weights once, and load the tokenizer alongside it.

The core idea looks like this:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model and tokenizer...")

    dl_model["BiGRU"] = build_bigru_model()
    dl_model["BiGRU"].load_weights(weights_path)

    with open(tokenizer_path, 'rb') as file:
        dl_model['Tokenizer'] = pickle.load(file)

    print("Model are loaded successfully!")

    yield

    dl_model.clear()
```

For me, this was the breakthrough.

I finally understood that **model training and model serving are two separate stages**. The notebook is where the model learns. The API is where the learned model becomes a service.

---

## 🔁 What Happens Inside `/predict`

The prediction endpoint follows almost the same logic as the training pipeline.

The API receives text, cleans it, tokenizes it with the saved tokenizer, pads the sequence to 50 tokens, runs the BiGRU, extracts the highest-probability emotion, and returns the complete probability distribution.

In simplified form:

```text
POST /predict
      ↓
Validate input with Pydantic
      ↓
Preprocess text
      ↓
Tokenizer → sequence
      ↓
Pad to length 50
      ↓
BiGRU prediction
      ↓
Argmax → predicted emotion
      ↓
Return emotion + confidence + probabilities
```

The API exposes three main routes:

- `GET /` — serves the web interface
- `GET /health` — checks whether the server and model are loaded
- `POST /predict` — performs emotion prediction

The `/health` endpoint was particularly useful because it gave me a simple way to verify that the application had actually loaded the model before I tested prediction requests.

---

## From Localhost to Deployment 🌐

Getting the API working locally was only half the story.

Deploying a TensorFlow-based FastAPI application introduced another layer of problems. The dependency environment mattered, the model files had to be available, and the server had to start correctly in a hosted environment.

The repository is now deployed on **Render**, which means the project is not just a notebook experiment anymore. It can be accessed as a running application from the web.

The dependency file currently pins the major runtime pieces, including FastAPI, Uvicorn, TensorFlow, NumPy, h5py, Pydantic, and `python-multipart`.

This part taught me something I had underestimated before:

> A model that works on my machine is not automatically a model that works in production.

Deployment adds operating-system, dependency, startup, file-path, memory, and serving concerns that do not appear when everything is running inside a notebook.

---

## Results 📊

The final BiGRU evaluation in the notebook produced:

| Metric | Result |
|---|---:|
| Test Loss | ~0.1824 |
| Test Accuracy | **~92.95%** |
| Number of Classes | 6 |
| Maximum Sequence Length | 50 |
| Vocabulary Limit | 10,000 |

The six target emotions are:

`Sadness · Joy · Love · Anger · Fear · Surprise`

The reported BiGRU result is from the existing notebook evaluation, so it should be interpreted as an evaluation on this dataset rather than as evidence that the model will generalize equally well to every kind of real-world text.

---

## Caveats & Limitations

- **Existing Dataset:**  
  The biggest limitation of this project is the dataset. I used an existing
  [Hugging Face Emotion dataset](https://huggingface.co/datasets/dair-ai/emotion)
  rather than collecting a new, domain-specific dataset. This made the project
  excellent for learning the complete NLP workflow, but it also means I should
  be careful about claiming that the model is broadly generalized.

- **Benchmark Performance vs. Real-World Generalization:**  
  A model can achieve a strong test score on a familiar benchmark and still
  struggle with text that differs substantially from its training data. For
  example, real-world text may contain slang, sarcasm, domain-specific
  language, unusual phrasing, or completely different patterns.

- **Interpreting the 92.95% Accuracy:**  
  I therefore consider the **92.95% test accuracy** to be satisfactory
  benchmark performance for this learning project, rather than proof of
  universal emotion understanding. One of the important lessons I learned
  from this project was that a good evaluation metric does not automatically
  mean that a model will be reliable in every real-world situation.

- **TensorFlow/Keras Version Compatibility:**  
  Another important challenge appeared when I tried to move the trained model
  from the training environment to the FastAPI application. The complete
  `.keras` model could not always be loaded successfully because of
  **TensorFlow/Keras version differences** between the environment where the
  model was trained and the environment where it was being deployed.

- **Saving Weights Instead of the Complete Model:**  
  To avoid depending on the exact model configuration serialized inside the
  `.keras` file, I saved the model's weights separately and rebuilt the
  BiGRU architecture in `main.py` before loading those weights. This is why
  the FastAPI application contains the `build_bigru_model()` function and
  loads the weights during the application's lifespan:

  ```python
  def build_bigru_model():
      model = Sequential([
          Embedding(input_dim=10000, output_dim=300),
          Bidirectional(GRU(units=128, return_sequences=True)),
          Dropout(0.5),
          Bidirectional(GRU(units=64)),
          Dropout(0.5),
          Dense(6, activation='softmax')
      ])

      model.build(input_shape=(None, 50))
      return model
  ```
-**Deployment Compatibility:**
Because TensorFlow and Keras are sensitive to version compatibility,
the deployed POST /predict endpoint may not always work if the model
weights, architecture, tokenizer, and serving environment are not kept
consistent.
  

---

## 🧩 Project Structure

```text
analyzing_sentiments_using_FastAPI/
│
├── Artifacts/
│   ├── BiGRU_model.weights.h5
│   └── tokenizer.pkl
│
├── static/
│   └── index.html
│
├── Sentiment_analyzer.ipynb
├── main.py
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## ▶️ Running It Locally

### 1. Clone the repository

```bash
git clone https://github.com/debaratiD2/analyzing_sentiments_using_FastAPI.git
cd analyzing_sentiments_using_FastAPI
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI server

```bash
uvicorn main:app --reload
```

Then open the local application in your browser and test the `/health` and `/predict` endpoints.

FastAPI also provides interactive API documentation with `/docs` at the end of your localhost link, which was another useful part to send requests and inspect responses.

---

## 🧪 Example Request

```json
{
  "text": "I feel incredibly happy today!"
}
```

Example response shape:

```json
{
  "text": "I feel incredibly happy today!",
  "predicted_emotion": "joy",
  "confidence": 0.95,
  "all_probabilities": {
    "sadness": 0.01,
    "joy": 0.95,
    "love": 0.01,
    "anger": 0.01,
    "fear": 0.01,
    "surprise": 0.01
  }
}
```

---

# Final Note
The notebook contains the experimentation and model-training journey, while `main.py` contains the FastAPI application and inference pipeline. The saved artifacts allow the API to reconstruct the model and load the learned weights without retraining it.

The tutorial helped me with the implementation and debugging process to turn those ideas into something I could explain and maintain myself. Special thanks to **Sheryians AI School** for explaning the overall direction.

---

