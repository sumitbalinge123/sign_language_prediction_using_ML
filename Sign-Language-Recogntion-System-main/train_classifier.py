import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load data from pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Pad or trim sequences to a fixed length
max_sequence_length = 42  # Choose a suitable maximum length

# Group sequences into words
word_sequences = []
word_labels = []

current_word = ""
current_sequence = []
for sequence, label in zip(data_dict['data'], data_dict['labels']):
    if label != current_word:
        if current_sequence:
            word_sequences.append(current_sequence[:max_sequence_length] + [0] * (max_sequence_length - len(current_sequence)))
            word_labels.append(current_word)
        current_word = label
        current_sequence = sequence
    else:
        current_sequence.extend(sequence)

# Convert sequences to NumPy array
data = np.array(word_sequences)

# Encode word labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(word_labels)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, shuffle=True, stratify=labels)

# Initialize and train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions on the test set
y_predict = model.predict(x_test)

# Calculate accuracy
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

# Save the model and label encoder
with open('model.p', 'wb') as f:
    pickle.dump({'model': model, 'label_encoder': label_encoder}, f)