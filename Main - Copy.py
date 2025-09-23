from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
# Keras via TensorFlow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

main = Tk()
main.title("Fake Account Detection Using Machine Learning and Data Science")
main.geometry("1000x800")
main.config(bg="lightgreen")

# globals
filename = None
dataset = None
X = None
Y = None
X_train = X_test = y_train = y_test = None
accuracy_history = None
model = None

def loadProfileDataset():
    global filename, dataset
    outputarea.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir=".", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    if not filename:
        outputarea.insert(END, "No file selected.\n")
        return
    try:
        dataset = pd.read_csv(filename)
    except Exception as e:
        outputarea.insert(END, f"Error loading file: {e}\n")
        return
    outputarea.insert(END, filename + " loaded\n\n")
    outputarea.insert(END, str(dataset.head()) + "\n\n")
    outputarea.insert(END, f"Dataset shape: {dataset.shape}\n")

def preprocessDataset():
    global X, Y, dataset, X_train, X_test, y_train, y_test
    outputarea.delete('1.0', END)

    if dataset is None:
        outputarea.insert(END, "Please load a dataset first.\n")
        return

    # Expecting features in first 8 columns and target in 9th column (index 8)
    if dataset.shape[1] < 9:
        outputarea.insert(END, "Dataset must have at least 9 columns (8 features + 1 target). Found: " + str(dataset.shape[1]) + "\n")
        return

    # Extract features and labels
    X = dataset.values[:, 0:8].astype(float)
    Y = dataset.values[:, 8].astype(int)

    # Shuffle
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    # One-hot encode target for categorical_crossentropy
    Y = to_categorical(Y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=np.argmax(Y, axis=1), random_state=42)

    outputarea.insert(END, "\nDataset contains total Accounts : " + str(len(X)) + "\n")
    outputarea.insert(END, "Total profiles used to train ANN algorithm : " + str(len(X_train)) + "\n")
    outputarea.insert(END, "Total profiles used to test ANN algorithm  : " + str(len(X_test)) + "\n")

def executeANN():
    global model, X_train, X_test, y_train, y_test, accuracy_history
    outputarea.delete('1.0', END)

    if X_train is None:
        outputarea.insert(END, "Preprocess dataset before running the model.\n")
        return

    input_dim = X_train.shape[1]
    model = Sequential()
    model.add(Dense(200, input_shape=(input_dim,), activation='relu', name='fc1'))
    model.add(Dense(200, activation='relu', name='fc2'))
    model.add(Dense(y_train.shape[1], activation='softmax', name='output'))  # y_train.shape[1] classes

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print('ANN Neural Network Model Summary: ')
    print(model.summary())

    # Fit model
    hist = model.fit(X_train, y_train, verbose=2, batch_size=32, epochs=50, validation_split=0.15,
                     callbacks=[EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)])
    accuracy_history = hist.history

    results = model.evaluate(X_test, y_test, verbose=0)
    ann_acc = results[1] * 100  # accuracy
    outputarea.insert(END, f"ANN model generated and its test accuracy is: {ann_acc:.2f}%\n")

def graph():
    global accuracy_history
    outputarea.delete('1.0', END)
    if accuracy_history is None:
        outputarea.insert(END, "Run the ANN first to get accuracy/loss history.\n")
        return

    acc = accuracy_history.get('accuracy', [])
    val_acc = accuracy_history.get('val_accuracy', [])
    loss = accuracy_history.get('loss', [])
    val_loss = accuracy_history.get('val_loss', [])

    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Metric')
    # accuracy curves
    plt.plot(acc, label='Train Accuracy')
    if val_acc:
        plt.plot(val_acc, label='Val Accuracy')
    # loss curves
    plt.plot(loss, label='Train Loss')
    if val_loss:
        plt.plot(val_loss, label='Val Loss')
    plt.legend(loc='best')
    plt.title('ANN Training Accuracy & Loss')
    plt.show()

def predictProfile():
    global model
    outputarea.delete('1.0', END)

    if model is None:
        outputarea.insert(END, "Model not trained. Run ANN Algorithm first.\n")
        return

    file_to_test = filedialog.askopenfilename(initialdir=".", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    if not file_to_test:
        outputarea.insert(END, "No file selected for prediction.\n")
        return

    try:
        test_df = pd.read_csv(file_to_test)
    except Exception as e:
        outputarea.insert(END, f"Error reading test file: {e}\n")
        return

    if test_df.shape[1] < 8:
        outputarea.insert(END, "Test file must have at least 8 feature columns.\n")
        return

    test = test_df.values[:, 0:8].astype(float)
    # get probabilities then class index
    probs = model.predict(test)
    preds = np.argmax(probs, axis=1)

    for i in range(len(test)):
        msg = "Given Account Details Predicted As Genuine" if preds[i] == 0 else "Given Account Details Predicted As Fake"
        outputarea.insert(END, str(test[i].tolist()) + " --> " + msg + "\n\n")

def close():
    main.destroy()

font = ('times', 15, 'bold')
title = Label(main, text='Fake Account Detection Using Machine Learning and Data Science')
title.config(font=font)
title.config(height=3, width=80)
title.place(x=0, y=5)

ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Social Network Profiles Dataset", command=loadProfileDataset)
uploadButton.place(x=20, y=100)
uploadButton.config(font=ff)

processButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
processButton.place(x=20, y=150)
processButton.config(font=ff)

annButton = Button(main, text="Run ANN Algorithm", command=executeANN)
annButton.place(x=20, y=200)
annButton.config(font=ff)

graphButton = Button(main, text="ANN Accuracy & Loss Graph", command=graph)
graphButton.place(x=20, y=250)
graphButton.config(font=ff)

predictButton = Button(main, text="Predict Fake/Genuine Profile using ANN", command=predictProfile)
predictButton.place(x=20, y=300)
predictButton.config(font=ff)

exitButton = Button(main, text="Logout", command=close)
exitButton.place(x=20, y=350)
exitButton.config(font=ff)

font1 = ('times', 12, 'bold')
outputarea = Text(main, height=30, width=80)
scroll = Scrollbar(main, command=outputarea.yview)
outputarea.configure(yscrollcommand=scroll.set)
outputarea.place(x=320, y=100)
scroll.place(x=320 + 640, y=100, height=480)  # adjust scrollbar placement
outputarea.config(font=font1)

main.mainloop()
