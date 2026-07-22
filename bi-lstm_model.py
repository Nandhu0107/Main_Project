import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.utils import to_categorical

X = np.load("X_time_windows.npy")
y = np.load("y_labels.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)

label_encoder = LabelEncoder()
y_int = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)
y = to_categorical(y_int, num_classes=num_classes)

print("Classes:", num_classes)
print("Label mapping:", dict(zip(label_encoder.classes_, range(num_classes))))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

model = Sequential()

model.add(
    Bidirectional(
        LSTM(64, return_sequences=True),
        input_shape=(X.shape[1], X.shape[2])
    )
)

model.add(Dropout(0.3))

model.add(
    Bidirectional(
        LSTM(32)
    )
)

model.add(Dropout(0.3))

model.add(Dense(32, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))


model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[]
)

loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

model.save("bilstm_traffic_model.keras")
print("Bi-LSTM model saved successfully.")
