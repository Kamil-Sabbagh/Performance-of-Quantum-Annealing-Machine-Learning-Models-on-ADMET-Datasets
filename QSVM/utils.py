
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

def  plot_figure(SVM,alpha,data,t,b,N,sampler_type):
    plt.figure()
    cm = plt.cm.RdBu

    xx, yy = np.meshgrid(np.linspace(0.0, 1.0, 80),
                            np.linspace(0.0, 1.0, 80))
    Z = []
    for row in range(len(xx)):
        Z_row = []
        for col in range(len(xx[row])):
            target = np.array([xx[row][col], yy[row][col]])
            Z_row.append(SVM.predict_class(target, alpha, b))
        Z.append(Z_row)

    cnt = plt.contourf(xx, yy, Z, levels=np.arange(-1, 1.1,
                                                    0.1), cmap=cm, alpha=0.8, extend="both")

    plt.contour(xx, yy, Z, levels=[0.0], colors=(
        "black",), linestyles=("--",), linewidths=(0.8,))
    plt.colorbar(cnt, ticks=[-1, 0, 1])

    red_sv = []
    blue_sv = []
    red_pts = []
    blue_pts = []

    for i in range(N):
        if(alpha[i]):
            if(t[i] == 1):
                blue_sv.append(data[i, :2])
            else:
                red_sv.append(data[i, :2])

        else:
            if(t[i] == 1):
                blue_pts.append(data[i, :2])
            else:
                red_pts.append(data[i, :2])

    plt.scatter([el[0] for el in blue_sv],
                [el[1] for el in blue_sv], color='b', marker='^', edgecolors='k', label="Type 1 SV")

    plt.scatter([el[0] for el in red_sv],
                [el[1] for el in red_sv], color='r', marker='^', edgecolors='k', label="Type -1 SV")

    plt.scatter([el[0] for el in blue_pts],
                [el[1] for el in blue_pts], color='b', marker='o', edgecolors='k', label="Type 1 Train")

    plt.scatter([el[0] for el in red_pts],
                [el[1] for el in red_pts], color='r', marker='o', edgecolors='k', label="Type -1 Train")

    
    plt.legend(loc='lower right', fontsize='x-small')
    plt.savefig(f'{sampler_type}.jpg')

def compute_metrics(SVM, alpha, data, t, b):
    tp, fp, tn, fn = 0, 0, 0, 0
    y_scores = []  # To store the predicted scores for AUC calculation

    for i in range(len(data)):
        predicted_score = SVM.predict_class(data[i], alpha, b)  # Get the decision score
        predicted_cls = 1 if predicted_score > 0 else -1  # Classify based on the score
        y_scores.append(predicted_score)  # Store the score
        y_i = t[i]
        
        if y_i == 1:
            if predicted_cls > 0:
                tp += 1
            else:
                fp += 1
        else:
            if predicted_cls < 0:
                tn += 1
            else:
                fn += 1

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_score = tp / (tp + 0.5 * (fp + fn)) if (tp + 0.5 * (fp + fn)) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Calculate AUC score
    auc_roc = roc_auc_score(t, y_scores)

    return precision, recall, f_score, accuracy, auc_roc


