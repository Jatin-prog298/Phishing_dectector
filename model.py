import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Sample Dataset
# In a real scenario, you can load a large CSV file here.
data = {
    'text': [
        "Urgent: Your bank account is locked. Click here to verify: http://fake-bank.com",
        "Hey, are we still meeting for lunch tomorrow at noon?",
        "Congratulations! You won a free iPhone. Claim here: http://win-free-gift.net",
        "Here is the project report you requested for our meeting on Monday.",
        "Security Alert: Unusual sign-in attempt. Reset here: http://update-security.org",
        "Your Amazon order #12345 has shipped and is out for delivery.",
        "Dear customer, update your billing details immediately at http://phish-link.com",
        "Can you send me the password for the shared Wi-Fi network?",
        "Get rich quick! Click this link to earn $5000 a day: http://scam-url.biz",
        "Thanks for signing up for our weekly tech newsletter."
    ],
    'label': ['Phishing', 'Safe', 'Phishing', 'Safe', 'Phishing', 'Safe', 'Phishing', 'Safe', 'Phishing', 'Safe']
}

print("--- Step 1: Loading Dataset ---")
df = pd.DataFrame(data)
print(f"Dataset loaded with {len(df)} sample emails.\n")

# 2. Feature Extraction (Extracting keywords, URL patterns, text structure)
print("--- Step 2: Extracting Features via TF-IDF ---")
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['label']

# 3. Split Dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Model
print("--- Step 3: Training Logistic Regression Model ---")
model = LogisticRegression()
model.fit(X_train, y_train)
print("Model training complete.\n")

# 5. Make Predictions & Evaluate
print("--- Step 4: Model Evaluation ---")
y_pred = model.predict(X_test)

# Calculate Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc * 100:.2f}%\n")

# Generate Confusion Matrix
# It charts [True Safe, False Phishing] vs [False Safe, True Phishing]
cm = confusion_matrix(y_test, y_pred, labels=['Safe', 'Phishing'])
print("Confusion Matrix:")
print("               Predicted Safe | Predicted Phishing")
print(f"Actual Safe:         {cm[0][0]}        |        {cm[0][1]}")
print(f"Actual Phishing:     {cm[1][0]}        |        {cm[1][1]}")
print("\n--------------------------------")

# 6. Test on a Brand New Email
print("\n--- Step 5: Testing with a New Live Email ---")
new_email = ["URGENT: Your account password has expired. Log in to http://secure-the-theft.com to reset it."]
new_email_features = vectorizer.transform(new_email)
prediction = model.predict(new_email_features)
print(f"Email Text: '{new_email[0]}'")
print(f"Model Classification -> The email is: **{prediction[0]}**")
