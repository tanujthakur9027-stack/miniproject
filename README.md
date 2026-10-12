# Customer Support Chat

This project classifies customer questions into one of the supported customer
service intents and returns general, intent-specific guidance in a browser chat.
It uses the trained scikit-learn model; it does not access customer accounts,
orders, payment systems, or live store policies.

## Run the chat

Install the dependencies if needed, train the model if it has not been generated,
then start the local web app:

```powershell
python -m pip install -r requirements.txt
python train.py
python app.py
```

Open <http://127.0.0.1:8000> in a browser. The server listens only on the local
computer. Stop it with `Ctrl+C`.

The app predicts an intent and selects a general response for that topic. If
the model is unsure, it asks the customer to rephrase. Connect this app to an
authorized order or support system before using it to provide account-specific
answers or take actions such as cancelling orders.
