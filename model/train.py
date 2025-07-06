from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

def train():
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    joblib.dump(model, "model.pkl")

    preds = model.predict(X)
    acc = accuracy_score(y, preds)

    # Write accuracy to README
    update_readme(acc)

def update_readme(acc):
    line_to_write = f"**Accuracy:** {acc:.2f}\n"
    readme_path = "README.md"
    new_lines = []
    found = False

    with open(readme_path, "r") as f:
        for line in f:
            if line.startswith("**Accuracy:**"):
                new_lines.append(line_to_write)
                found = True
            else:
                new_lines.append(line)

    if not found:
        new_lines.append("\n## 📈 Latest Model Accuracy\n")
        new_lines.append(line_to_write)

    with open(readme_path, "w") as f:
        f.writelines(new_lines)

if __name__ == "__main__":
    train()
