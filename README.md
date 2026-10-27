# SentimentSense

SentimentSense is a web application for analysing the sentiment of movie reviews using two recurrent neural network models: a Simple Recurrent Neural Network (Simple RNN) and a Long Short-Term Memory network (LSTM).

The application provides a browser-based interface for analysing one review or uploading a file containing multiple reviews. It displays a sentiment label, positive-class score, and confidence estimate. For batch analysis, it also displays an overall summary and allows the results to be downloaded as a CSV file.

## Features

- Analyse a single review or a batch of reviews.
- Choose the Simple RNN or LSTM model, or compare both models for a single review.
- Display one of three labels: `Positive`, `Negative`, or `Uncertain`.
- Show the predicted positive sentiment score and confidence estimate.
- Upload `.txt` files with one review per line.
- Upload `.csv` files containing a `review` or `text` column.
- View batch sentiment counts and a distribution bar.
- Download batch predictions as a CSV file.
- Use a health endpoint to check server status and loaded models.

## Technology Stack

- Python
- TensorFlow / Keras
- Simple RNN and LSTM neural networks
- Flask
- HTML, CSS, and JavaScript
- NumPy, pandas, scikit-learn, Matplotlib, and Seaborn

## Project Structure

The application expects the project to follow this structure:

```text
SentimentSense/
├── app/
│   ├── server.py
│   └── static/
│       ├── index.html
│       ├── script.js
│       └── style.css
├── models/
│   ├── lstm.keras
│   └── simple_rnn.keras
├── sentimentSense.ipynb
├── requirements.txt
└── README.md
```

The saved model files are expected in the root-level `models/` directory. If the models are not present, the server will start without those models and prediction requests for unavailable models will fail.

## Requirements

- Python 3.10 or a compatible Python version supported by your installed TensorFlow release.
- `pip`
- The trained model files, or the notebook and environment needed to train them.

The backend imports Flask, so make sure Flask is installed. The supplied `requirements.txt` lists the other project dependencies but does not currently list Flask.

## Installation

### 1. Clone or download the project

Open a terminal and move into the project directory:

```bash
cd SentimentSense
```

### 2. Create and activate a virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install Flask
```

If you plan to use the notebook, make sure Jupyter is installed in the environment:

```bash
pip install notebook
```

## Model Training

The `sentimentSense.ipynb` notebook contains the model-training workflow using the IMDB movie-review dataset provided by Keras.

The notebook configures the following main parameters:

- Vocabulary size: `10,000`
- Maximum review sequence length: `250`
- Embedding dimension: `128`
- RNN/LSTM units: `64`
- Batch size: `64`
- Optimizer: Adam
- Loss function: binary cross-entropy
- Validation split: 20% of the training data
- Early stopping based on validation loss

It builds and trains two models:

1. **Simple RNN**
2. **LSTM**

The notebook also compares validation accuracy and loss, evaluates predictions on the test set, generates classification reports and confusion matrices, and compares model metrics.

The training code saves the best model checkpoints as:

```text
models/simple_rnn.keras
models/lstm.keras
```

Run the notebook from the project root so that the relative `models/` output path points to the expected directory. Create the `models/` directory before training if it does not exist.

## Running the Application

From the project root, run:

```bash
python app/server.py
```

The server runs locally at:

```text
http://127.0.0.1:5000
```

Open that address in a browser. The backend loads the IMDB word index and any model files it finds at startup.

To check the server status, visit:

```text
http://127.0.0.1:5000/health
```

A successful health response has the following shape:

```json
{
  "status": "ok",
  "models": ["lstm", "rnn"]
}
```

The `models` list contains only models successfully loaded by the server, so its contents may differ depending on which model files are available.

## How to Use the Web Interface

### Analyse one review

1. Open the application in your browser.
2. Select **One review**.
3. Enter or paste a movie review, or use one of the built-in examples.
4. Choose **Compare both**, **LSTM**, or **Simple RNN**.
5. Select **Analyze review**.
6. Review the predicted label, positive score, and confidence.

You can also press `Ctrl+Enter` on Windows/Linux or `Cmd+Enter` on macOS while the review field is focused.

### Analyse multiple reviews

1. Select **Many reviews**.
2. Choose either the LSTM or Simple RNN model.
3. Upload a `.txt` or `.csv` file.
4. Select **Analyze file**.
5. Review the batch summary and the first 25 results shown in the table.
6. Select **Download results (CSV)** to export predictions for the full batch.

For a `.txt` file, put one review on each line.

For a `.csv` file, use a column named `review` or `text`. If neither header is found, the application uses the first column.

## Sentiment Labels

The backend interprets the model output as the probability of positive sentiment.

| Positive score | Label |
|---|---|
| `>= 0.60` | Positive |
| `<= 0.40` | Negative |
| `> 0.40` and `< 0.60` | Uncertain |

The confidence value is calculated as:

```text
confidence = max(positive_score, 1 - positive_score)
```

The `Uncertain` label is a threshold-based category added by the application; it is not a separate class trained by the neural networks.

## API Endpoints

### `GET /`

Serves the web interface.

### `GET /health`

Returns the server status and names of models loaded successfully.

### `POST /predict`

Analyses one review.

Example request:

```json
{
  "text": "The story was engaging and the acting was excellent.",
  "model": "lstm"
}
```

Supported `model` values are `lstm`, `rnn`, and `both`. If omitted, the backend defaults to `lstm`.

Example response:

```json
{
  "results": {
    "lstm": {
      "label": "Positive",
      "score": 0.91,
      "confidence": 0.91
    }
  }
}
```

The score shown here is illustrative; actual values depend on the loaded model and submitted text.

### `POST /predict_batch`

Analyses a list of reviews using one selected model.

Example request:

```json
{
  "texts": [
    "An excellent movie with a compelling story.",
    "A slow and disappointing film."
  ],
  "model": "lstm"
}
```

The response contains the selected model, one result per review, and a summary with total, positive, negative, and uncertain counts.

## Input Limits and Behaviour

- A single review may contain up to 5,000 characters.
- A batch may contain up to 2,000 non-empty reviews.
- Batch reviews longer than 5,000 characters are truncated by the backend.
- Text is normalised and encoded using the Keras IMDB word index.
- Sequences are padded or truncated to 250 tokens.
- Model inference is protected by a lock to avoid simultaneous prediction calls through the same loaded model.

## Important Notes and Limitations

- The models are trained on IMDB movie reviews. Results on product reviews, service feedback, sarcasm, or other domains may be less reliable.
- Sentiment predictions are model estimates, not guaranteed assessments of a review's meaning.
- Both trained model files must exist in the expected `models/` directory for the **Compare both** option to work.
- The backend downloads or accesses the Keras IMDB word index when it initialises if that resource is not already available locally, so the first run may require internet access.
- The development server binds to `127.0.0.1`, making it available on the local machine by default. This configuration is not intended as a production deployment setup.

## Troubleshooting

**The page does not open**

- Confirm that the virtual environment is activated.
- Run `python app/server.py` from the project root.
- Check the terminal for startup errors.

**A model is reported as unavailable**

- Confirm that the relevant `.keras` file exists in the root-level `models/` directory.
- Check that the model filename matches `lstm.keras` or `simple_rnn.keras`.
- If needed, run the training notebook and save the model files to that directory.

**A dependency import fails**

- Activate the intended Python environment.
- Install the dependencies with `pip install -r requirements.txt`.
- Install Flask separately with `pip install Flask` unless it has been added to `requirements.txt`.

**The prediction seems unreliable**

- Try a clear movie review in English.
- Keep in mind that the model was trained on IMDB movie reviews and may not generalise well to other kinds of text.

## License

This project is licensed under the MIT License, which permits anyone to use, copy, modify, merge, publish, distribute, sublicense, and sell copies of the software, subject to the conditions of the license. The MIT License requires that the original copyright notice and permission notice be included in all copies or substantial portions of the software. See the LICENSE file for the complete license terms.
