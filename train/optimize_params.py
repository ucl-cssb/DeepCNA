import optuna
import subprocess

def objective(trial):
    # Define hyperparameter search space
    hidden1 = trial.suggest_int("hidden1", 200, 1000, step=100)
    hidden2 = trial.suggest_int("hidden2", 200, 1000, step=100)
    subnetout = trial.suggest_int("subnetout", 40, 100, step=10)
    momentum = trial.suggest_float("momentum", 0.5, 0.95, step=0.05)
    batch = trial.suggest_int("batch", 64, 256, step=32)
    pdrop = trial.suggest_float("pdrop", 0.0, 0.6, step=0.1)
    lr = trial.suggest_float("lr", 0.0001, 0.01, log=True)

    # Call train_cv.py with selected hyperparameters
    cmd = [
        "python", "train_cv.py",
        "--epochs", "50",
        "--lr", str(lr),
        "--dropout", str(pdrop),
        "--wd", "0.01",
        "--batch_size", str(batch),
        "--momentum", str(momentum),
        "--hidden1", str(hidden1),
        "--hidden2", str(hidden2),
        "--subnetout", str(subnetout),
        "--output_file", "results_cv_optuna.txt"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Error executing train_cv.py")
        print("STDERR Output:\n", e.stderr)
        print("STDOUT Output:\n", e.stdout)

    # Extract F1 score from the output file
    with open("results_cv_optuna.txt", "r") as f:
        last_line = f.readlines()[-1]
    
    mean_f1_score = float(last_line.strip().split(" ")[-1])

    return mean_f1_score  

# Run Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)  

# Print the best parameters
print("Best hyperparameters:", study.best_params)


import subprocess



