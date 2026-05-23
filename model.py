import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("clean_fakejobs.csv")

# Splitting dataset in train and test
X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["fraudulent"],
    test_size=0.3,
    random_state=42,
    stratify=data["fraudulent"],
)

# Converting the data into vector format
#  instantiate the vectorizer
vect = CountVectorizer()

# learn training data vocabulary, then use it to create a document-term matrix
# fit
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

# instantiate a Decision Tree Classifier
rf = RandomForestClassifier(random_state=42)

clf = rf.fit(X_train_dtm, y_train)
y_pred = clf.predict(X_test_dtm)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Save artifacts for app.py
pickle.dump(vect, open("vectorizer.pkl", "wb"))
pickle.dump(clf, open("model.pkl", "wb"))
