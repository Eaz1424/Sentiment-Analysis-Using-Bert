# Sentiment Analysis with BERT

This project demonstrates how to perform sentiment analysis using the `transformers` library and a pre-trained BERT model. It includes steps to scrape reviews from a website, process them, and calculate sentiment scores.

---

## 📂 Files

- `Sentiment.ipynb`: Jupyter Notebook containing the code for sentiment analysis.

---

## 📚 Requirements

This project uses:

- Python 3.x
- `torch`
- `transformers`
- `requests`
- `beautifulsoup4`
- `pandas`
- `numpy`

Install dependencies:

```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers requests beautifulsoup4 pandas numpy

```markdown
# Sentiment Analysis with Transformers

This project demonstrates how to perform sentiment analysis using the `transformers` library and a pre-trained BERT model. It includes steps to scrape reviews from a website, process them, and calculate sentiment scores.

---

## 📂 Files

- `Sentiment.ipynb`: Jupyter Notebook containing the code for sentiment analysis.

---

## 📚 Requirements

This project uses:

- Python 3.x
- `torch`
- `transformers`
- `requests`
- `beautifulsoup4`
- `pandas`
- `numpy`

Install dependencies:

```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers requests beautifulsoup4 pandas numpy
```

---

## 🚀 How to Run

1. Clone the repository and navigate to the project directory.
2. Open the `Sentiment.ipynb` notebook in Jupyter Notebook or JupyterLab.
3. Follow the steps in the notebook to:
   - Install dependencies.
   - Instantiate the pre-trained BERT model.
   - Scrape reviews from a website.
   - Calculate sentiment scores for the reviews.

---

## 🔍 What It Does

1. **Install Dependencies**: Installs required Python libraries.
2. **Load Pre-trained Model**: Loads the `nlptown/bert-base-multilingual-uncased-sentiment` model for sentiment analysis.
3. **Scrape Reviews**: Scrapes reviews from a specified website using `requests` and `BeautifulSoup`.
4. **Calculate Sentiment**: Encodes reviews, passes them through the model, and calculates sentiment scores (1 to 5).
5. **Store Results**: Loads reviews into a Pandas DataFrame and appends sentiment scores.

---

## 📄 Usage

### 1. Import Libraries and Load Model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
```

### 2. Test a Sample Review

```python
tokens = tokenizer.encode("It was good but could've been better. Great", return_tensors='pt')
result = model(tokens)
sentiment = int(torch.argmax(result.logits)) + 1
print("Sentiment score:", sentiment)
```

### 3. Scrape Yelp Reviews

```python
import requests
from bs4 import BeautifulSoup
import re

url = 'https://www.yelp.com/biz/social-brew-cafe-pyrmont'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

regex = re.compile('.*comment.*')
results = soup.find_all('p', {'class': regex})
reviews = [r.text for r in results]
```

### 4. Analyze Sentiment for All Reviews

```python
import pandas as pd
import numpy as np

df = pd.DataFrame(np.array(reviews), columns=['review'])

def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits)) + 1

df['sentiment'] = df['review'].apply(lambda x: sentiment_score(x[:512]))
print(df)
```

---

## 📊 Example Output

| Review                               | Sentiment |
|--------------------------------------|-----------|
| Great food and vibes!                | 5         |
| It was okay, not the best.           | 3         |
| Terrible service, not coming back.   | 1         |

---

## 🧠 Model Info

- **Model**: `nlptown/bert-base-multilingual-uncased-sentiment`
- **Type**: Sequence classification (1 to 5 stars)
- **Languages**: Multilingual BERT


---

## 🛠 Future Improvements

- Add sentiment visualization (e.g., pie chart, histogram)
- Deploy as a web app using Streamlit or Flask
- Support multiple review sources like Amazon or Google Maps
- Use batched inference to improve speed

---


## 📜 Notebook Steps

### 1. Install and Import Dependencies
- Install required libraries using `pip`.
- Import necessary modules like `transformers`, `torch`, `requests`, `BeautifulSoup`, and `pandas`.

### 2. Instantiate Model
- Load the pre-trained `nlptown/bert-base-multilingual-uncased-sentiment` model and tokenizer.

### 3. Encode and Calculate Sentiment
- Encode a sample review and calculate its sentiment score using the model.

### 4. Collect Reviews
- Scrape reviews from a website (e.g., Yelp) using `requests` and `BeautifulSoup`.

### 5. Load Reviews into DataFrame and Score
- Store the reviews in a Pandas DataFrame.
- Apply the sentiment scoring function to each review and append the results to the DataFrame.

---


## 📚 References

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [nlptown BERT Sentiment Model](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)
- [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)

---



