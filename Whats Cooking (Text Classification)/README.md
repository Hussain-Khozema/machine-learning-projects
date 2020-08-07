### Whats Cooking (Text classification via Scikit-Learn)

#### Problem
> A text classification model to predict cuisine given a list of ingredients

#### Dataset 

* [Ingredients dataset](https://www.kaggle.com/c/whats-cooking/data)

#### Loading and running model


```python
import pickle
model = pickle.load(open("model/cuisine_classification_model.pkl", "rb"))
tfidf = pickle.load(open("model/tfidf.pkl", "rb"))
```


```python
def predictor(model, tfidf, text):
    print("query: ", text)
    result = model.predict(tfidf.transform([text]))
    print("Predicted cuisine: ", result[0])
```


```python
text = 'sweet potatoes, pumpkin pie spice, eggs, whipped topping'
predictor(model, tfidf, text)
```

    query:  sweet potatoes, pumpkin pie spice, eggs, whipped topping
    Predicted cuisine:  southern_us
