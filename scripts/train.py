import datetime
import numpy as np
import utils_config
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
config = utils_config.load_config("./config.json")

def train(base, X, y, X_test, y_test, model_klass, model_kwargs = {}):
    day_hour = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    cv = StratifiedKFold(n_splits=config.k_folds)
    f1_score_val_list = []
    f1_score_train_list = []
    acc_train = []
    acc_val = []
    recall_train = []
    recall_val = []
    prec_train = []
    prec_val = []
    model_list =[]
    scaler_list = []

    train_model = None
    if "SVC" in str(model_klass):
        train_model = "SVC"
    if "Tree" in str(model_klass):
        train_model = "DecisionTree"
    if "Logistic" in str(model_klass):
        train_model = "LogisticRegression"  

    with open(f"{config.results}/results_{train_model}_{day_hour}_{base}.txt", "a") as file:
            file.write(f"""Configs: {model_kwargs}\n""")

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train = X[train_idx, :]
        y_train = y[train_idx]
        X_val = X[val_idx, :]
        y_val = y[val_idx]

        # Escala
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        scaler_list.append(scaler)

        # Treino
        model = model_klass(**model_kwargs)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_train_scaled)
        y_pred_val = model.predict(X_val_scaled)
        
        with open(f"{config.results}/results_{train_model}_{day_hour}_{base}.txt", "a") as file:
            file.write(f"""========================= FOLD {fold} ==========================
        O Resultado da ACURÁCIA em TREINO é: {100 * accuracy_score(y_train, y_pred):.2f}%
        O Resultado da ACURÁCIA na VALIDAÇÃO é: {100 * accuracy_score(y_val, y_pred_val):.2f}%
        O Resultado da SENSIBILIDADE em TREINO é: {100 *  recall_score(y_train, y_pred):.2f}%
        O Resultado da SENSIBILIDADE na VALIDAÇÃO é: {100 *  recall_score(y_val, y_pred_val):.2f}%
        O Resultado da PRECISÃO em TREINO é: {100*  precision_score(y_train, y_pred):.2f}%
        O Resultado da PRECISÃO na VALIDAÇÃO é: {100*  precision_score(y_val, y_pred_val):.2f}%
        O Resultado da de F1-Score em TREINO é: {f1_score(y_train, y_pred):.2}
        O Resultado da de F1-Score na VALIDAÇÃO: {f1_score(y_val, y_pred_val):.2}\n\n""")
        
        acc_train.append(accuracy_score(y_train, y_pred))
        acc_val.append(accuracy_score(y_val, y_pred_val))
        recall_train.append(recall_score(y_train, y_pred))
        recall_val.append(recall_score(y_val, y_pred_val))
        prec_train.append(precision_score(y_train, y_pred))
        prec_val.append(precision_score(y_val, y_pred_val))
        f1_score_train_list.append(f1_score(y_train, y_pred))
        f1_score_val_list.append(f1_score(y_val, y_pred_val))
        
        model_list.append(model)
    with open(f"{config.results}/results_{train_model}_{day_hour}_{base}.txt", "a") as file:
            file.write(f"""========================= Resultado Médio =========================
        O resultado Médio da ACURÁCIA em TREINO é: {np.mean(acc_train): .2} +- {np.std(acc_train): .2}
        O resultado Médio da ACURÁCIA em VALIDAÇÃO é: {np.mean(acc_val): .2} +- {np.std(acc_val): .2}
        O resultado Médio da SENSIBILIDADE em TREINO é: {np.mean(recall_train): .2} +- {np.std(recall_train): .2}
        O resultado Médio da SENSIBILIDADE em VALIDAÇÃO é: {np.mean(recall_val): .2} +- {np.std(recall_val): .2}
        O resultado Médio da PRECISÃO em TREINO é: {np.mean(prec_train): .2} +- {np.std(prec_train): .2}
        O resultado Médio da PRECISÃO em VALIDAÇÃO é: {np.mean(prec_val): .2} +- {np.std(prec_val): .2}
        O resultado Médio da F1-Score em TREINO é {np.mean(f1_score_train_list): .2} +- {np.std(f1_score_train_list): .2}
        O resultado Médio da F1-Score em VALIDAÇÃO é: {np.mean(f1_score_val_list): .2} +- {np.std(f1_score_val_list): .2}\n
===================================================\n""")
            
            
    best_model_idx = np.argmax(f1_score_val_list)
    with open(f"{config.results}/results_{train_model}_{day_hour}_{base}.txt", "a") as file:
            file.write(f"""Meu melhor fold é: {best_model_idx}\n""")
            
    best_model = model_list[best_model_idx]
    best_scaler = scaler_list[best_model_idx]
    X_test_scaled = best_scaler.transform(X_test)
    y_pred_test = model.predict(X_test_scaled)

    with open(f"{config.results}/results_{train_model}_{day_hour}_{base}.txt", "a") as file:
            file.write(f"""Meu resultado de F1-Score para o conjunto de TESTE é: {f1_score(y_test, y_pred_test):.2}
O resultado Médio da ACURÁCIA em TESTE é: {100 * accuracy_score(y_test, y_pred_test):.2f}%
O resultado Médio da SENSIBILIDADE em TESTE é: {100 * recall_score(y_test, y_pred_test):.2f}%
O resultado Médio da PRECISÃO em TESTE é: {100 * precision_score(y_test, y_pred_test):.2f}%
===================================================\n
===================================================\n""")

    return best_model, best_scaler, f1_score_val_list, X_val_scaled, y_val