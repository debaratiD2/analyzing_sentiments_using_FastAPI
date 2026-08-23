from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dropout, Dense
import numpy as np
import re
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model
import pickle
from fastapi.middleware.cors import CORSMiddleware



def greet():
    return {"message": "Hello world!"}


'''
1. Model path
2. Tokenizer path
3. Max Sequence length
4. Emotion labels
5. Emotion emojis
'''

#model_path = '/Artifacts/BiGRU_model.h5 Artifacts'
weights_path = 'Artifacts/BiGRU_model.weights.h5'
tokenizer_path = 'Artifacts/tokenizer.pkl'

max_sequence_length = 50

emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

emotion_emojis = {
    'sadness': '😢',
    'joy': '😄',
    'love': '❤️',
    'anger': '😠',
    'fear': '😨',
    'surprise': '😲'
}

# preprocessing the text
'''
1. convert to lowercase
2. remove apostropes
3. remove special characters and punctuation
4. remove extra spaces
'''

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text



#Request and Response Models
'''
1. Text Input
2. Prediction Output
3. Health Response [server health]
'''

class TextInput(BaseModel):
    text : str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Input text for emotion prediction",
        json_schema_extra={"examples":["I feel so happy today"]}
    )


class PredictionResponse(BaseModel):
    text: str 
    predicted_emotion: str 
    confidence: float
    all_probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool    


#Model Loading and lifespan management
'''load the model and tokenizer when the server starts up'''

dl_model = {}


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model and tokenizer...")
 
    dl_model["BiGRU"] = build_bigru_model()
    dl_model["BiGRU"].load_weights(weights_path)
 
    with open(tokenizer_path, 'rb') as file:
        dl_model['Tokenizer'] = pickle.load(file)
 
    print("Model are loaded successfully!")
 
    yield  # pause, model is loaded and server is running
 
    dl_model.clear()


'''
Mount the static files  to the FastAPI app
CORS --> Cross Origin REsource Sharing
'''

app = FastAPI(
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount('/static', StaticFiles(directory='static'), name='static')

'''
api endpoints
1. server ui homepage
2. health check endpoint('/health')
3. predict endpoint('/predict')
'''

@app.get('/', include_in_schema=False)
def server_ui():
    return FileResponse('static/index.html')

@app.get('/health', response_model = HealthResponse)
def health_check():
    return HealthResponse(status = "Server is running", model_loaded = bool(dl_model))

@app.post('/predict', response_model = PredictionResponse)
def predict_emotion(text_input:TextInput):

    BiGRU_model = dl_model.get("BiGRU")
    tokenizer_model = dl_model.get("Tokenizer")

    if BiGRU_model is None or tokenizer_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please try again later.")


    cleaned_text = preprocess_text(text_input.text)
    print(f"Cleaned text: {cleaned_text}")

    tokenized_text = tokenizer_model.texts_to_sequences([cleaned_text])

    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen = max_sequence_length,
        padding = 'post',
        truncating = 'post'
    )

    probabilities = BiGRU_model.predict(padded_sequence)[0]

    top_emotion_index = int(np.argmax(probabilities))

    all_probabilities = {
        label: float(prob)for label, prob in zip(emotion_labels, probabilities)
    }

    return PredictionResponse(
        text = text_input.text,
        predicted_emotion = emotion_labels[top_emotion_index],
        confidence = float(probabilities[top_emotion_index]),
        all_probabilities = all_probabilities
    )



'''
1. Imports
       ↓
2. Configuration
       ↓
3. Functions
       ↓
4. Pydantic models
       ↓
5. Lifespan
       ↓
6. Create FastAPI app
       ↓
7. Middleware
       ↓
8. Static files
       ↓
9. API routes
'''