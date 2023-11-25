import tkinter as tk
from tkinter import filedialog
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class DataTrainer:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Trainer")
        self.root.geometry("800x600")

        self.data = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.model_lr, self.model_ridge, self.model_nn = None, None, None
        self.running = True  # Flag to check if the loop is running

        self.fig, self.ax = plt.subplots(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Data", command=self.load_data)
        file_menu.add_command(label="Preprocess Data", command=self.preprocess_data)
        file_menu.add_command(label="Train Models", command=self.train_models)
        file_menu.add_command(label="Evaluate Models", command=self.evaluate_models)
        file_menu.add_command(label="Exit", command=self.exit_app)

        # Add a menu for cross-validation
        cross_val_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Cross-Validation", menu=cross_val_menu)
        cross_val_menu.add_command(label="Linear Regression", command=self.cross_validate_lr)
        cross_val_menu.add_command(label="Ridge Regression", command=self.cross_validate_ridge)

        # Add a menu for neural network visualization
        nn_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Neural Network", menu=nn_menu)
        nn_menu.add_command(label="Visualize Architecture", command=self.visualize_neural_network)

        self.root.after(100, self.check_loop)  # Start the loop after a delay
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)  # Handle window close event
        self.root.mainloop()

    def check_loop(self):
        if self.running:
            self.root.after(100, self.check_loop)
        else:
            print("Main loop terminated.")
            self.root.quit()

    def load_data(self):
        file_path = filedialog.askopenfilename(title="Select Data File", filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.data = pd.read_csv(file_path)
            print("Data loaded successfully.")
        else:
            print("No file selected.")

    def preprocess_data(self):
        if self.data is not None:
            # Assuming the target variable is in the last column
            X = self.data.iloc[:, :-1]
            y = self.data.iloc[:, -1]

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            print("Data preprocessed.")
        else:
            print("Please load data first.")

    def train_models(self):
        if self.X_train is not None and self.y_train is not None:
            # Linear Regression
            self.model_lr = LinearRegression()
            self.model_lr.fit(self.X_train, self.y_train)

            # Ridge Regression
            self.model_ridge = Ridge()
            self.model_ridge.fit(self.X_train, self.y_train)

            # Simple Neural Network
            self.model_nn = Sequential()
            self.model_nn.add(Dense(64, input_dim=self.X_train.shape[1], activation='relu'))
            self.model_nn.add(Dense(1, activation='linear'))
            self.model_nn.compile(optimizer='adam', loss='mean_squared_error')
            self.model_nn.fit(self.X_train, self.y_train, epochs=50, batch_size=32, validation_split=0.2)

            print("Models trained.")

            # Plot training results
            y_pred_lr_train = self.model_lr.predict(self.X_train)
            y_pred_ridge_train = self.model_ridge.predict(self.X_train)
            y_pred_nn_train = self.model_nn.predict(self.X_train)

            print("Linear Regression Mean Squared Error (Train):", mean_squared_error(self.y_train, y_pred_lr_train))
            print("Ridge Regression Mean Squared Error (Train):", mean_squared_error(self.y_train, y_pred_ridge_train))

            # Evaluate Neural Network
            mse_nn_train = mean_squared_error(self.y_train, y_pred_nn_train)
            print(f"Neural Network Mean Squared Error (Train): {mse_nn_train}")

            self.plot_results(self.X_train, self.y_train, y_pred_lr_train, label="LR Train")
            self.plot_results(self.X_train, self.y_train, y_pred_ridge_train, label="Ridge Train")
            self.plot_results(self.X_train, self.y_train, y_pred_nn_train, label="NN Train")
        else:
            print("Please preprocess data first.")

    def evaluate_models(self):
        if self.X_test is not None and self.y_test is not None and self.model_lr is not None and self.model_ridge is not None and self.model_nn is not None:
            # Linear Regression
            y_pred_lr = self.model_lr.predict(self.X_test)
            mse_lr = mean_squared_error(self.y_test, y_pred_lr)
            print("Linear Regression Mean Squared Error:", mse_lr)

            # Ridge Regression
            y_pred_ridge = self.model_ridge.predict(self.X_test)
            mse_ridge = mean_squared_error(self.y_test, y_pred_ridge)
            print("Ridge Regression Mean Squared Error:", mse_ridge)

            # Neural Network
            y_pred_nn = self.model_nn.predict(self.X_test)
            mse_nn = mean_squared_error(self.y_test, y_pred_nn)
            print(f"Neural Network Mean Squared Error: {mse_nn}")

            # Plot evaluation results
            self.plot_results(self.X_test, self.y_test, y_pred_lr, label="LR")
            self.plot_results(self.X_test, self.y_test, y_pred_ridge, label="Ridge")
            self.plot_results(self.X_test, self.y_test, y_pred_nn, label="NN")
        else:
            print("Please train the models first.")

    def cross_validate_lr(self):
        if self.X_train is not None and self.y_train is not None and self.model_lr is not None:
            # Cross-validation for Linear Regression
            y_pred_lr_cv = cross_val_predict(self.model_lr, self.X_train, self.y_train, cv=5)
            print("Linear Regression Mean Squared Error (Cross-Validation):", mean_squared_error(self.y_train, y_pred_lr_cv))
            self.plot_results(self.X_train, self.y_train, y_pred_lr_cv, label="LR CV")
        else:
            print("Please preprocess and train the Linear Regression model first.")

    def cross_validate_ridge(self):
        if self.X_train is not None and self.y_train is not None and self.model_ridge is not None:
            # Cross-validation for Ridge Regression
            y_pred_ridge_cv = cross_val_predict(self.model_ridge, self.X_train, self.y_train, cv=5)
            print("Ridge Regression Mean Squared Error (Cross-Validation):", mean_squared_error(self.y_train, y_pred_ridge_cv))
            self.plot_results(self.X_train, self.y_train, y_pred_ridge_cv, label="Ridge CV")
        else:
            print("Please preprocess and train the Ridge Regression model first.")

    def visualize_neural_network(self):
        if self.model_nn is not None:
            # Get the model architecture
            model_summary = []
            self.model_nn.summary(print_fn=lambda x: model_summary.append(x))
            model_summary = "\n".join(model_summary)

            # Display the architecture in a tkinter window
            window = tk.Toplevel(self.root)
            window.title("Neural Network Architecture")

            text_widget = tk.Text(window, wrap=tk.WORD)
            text_widget.insert(tk.END, model_summary)
            text_widget.pack(fill=tk.BOTH, expand=True)

            save_button = tk.Button(window, text="Save Summary", command=self.save_model_summary)
            save_button.pack()

        else:
            print("Neural Network model not trained yet. Please train the model first.")

    def save_model_summary(self):
        # Save the model architecture summary to a text file
        model_summary = []
        self.model_nn.summary(print_fn=lambda x: model_summary.append(x))
        model_summary = "\n".join(model_summary)

        with filedialog.asksaveasfile(defaultextension=".txt", filetypes=[("Text files", "*.txt")]) as file:
            if file:
                file.write(model_summary)
                print("Model summary saved successfully.")

    def plot_results(self, X_test, y_test, y_pred, label=None):
        self.ax.clear()
        if isinstance(X_test, pd.DataFrame):
            self.ax.scatter(X_test.iloc[:, 0], y_test, color='blue', label='Actual')
            self.ax.scatter(X_test.iloc[:, 0], y_pred, color='red', label=f'Predicted ({label})')
        else:
            self.ax.scatter(X_test, y_test, color='blue', label='Actual')
            self.ax.scatter(X_test, y_pred, color='red', label=f'Predicted ({label})')
        self.ax.legend()
        self.ax.set_xlabel('X_test')
        self.ax.set_ylabel('y')
        self.canvas.draw()

    def exit_app(self):
        print("Exiting application.")
        self.running = False  # Stop the loop
        self.root.quit()

    def on_close(self):
        print("Closing window.")
        self.running = False  # Stop the loop
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DataTrainer(root)
