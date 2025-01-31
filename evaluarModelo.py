from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from sklearn.preprocessing import label_binarize


def get_metrics(y_test, y_pred):
  cm = confusion_matrix(y_test, y_pred)
  tp, fn, fp, tn = cm.ravel()
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  report = classification_report(y_test, y_pred)
  fpr, tpr, thresholds = roc_curve(y_test, y_pred)
  roc_auc = auc(fpr, tpr)

  plt.figure(figsize=(6, 6))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
  plt.xlabel('Valor Real')
  plt.ylabel('Predicción')
  plt.title('Matriz de Confusión')
  tick_labels = ['Negativo', 'Positivo']
  plt.xticks(np.arange(2) + 0.5, tick_labels)
  plt.yticks(np.arange(2) + 0.5, tick_labels)
  plt.savefig('binaryMetrics.png')
  print()
  print("Accuracy:", accuracy)
  print("Precision:", precision)
  print("Recall:", recall)
  print("F1-score:", f1)
  print()
  print("Clasification report:")
  print(report)
  print()
  plt.figure(figsize=(6, 6))
  plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlabel('Tasa de Falsos Positivos')
  plt.ylabel('Tasa de Verdaderos Positivos')
  plt.title('Curva ROC')
  plt.legend(loc="lower right")
  plt.savefig('roc_curve.png')


def get_metrics_multiclass(y_test, y_pred, class_names, class_names_str):
    total = 0
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, target_names=class_names_str)

    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.title('Matriz de Confusión')
    plt.xticks(np.arange(len(class_names)) + 0.5, class_names_str, rotation=45)
    plt.yticks(np.arange(len(class_names)) + 0.5, class_names_str, rotation=0)
    plt.tight_layout()
    plt.savefig('multiclass.png')

    print()
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print()
    print("Clasification report:")
    print(report)

    y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
    y_pred_bin = label_binarize(y_pred, classes=range(len(class_names)))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(10, 8))

    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        total += roc_auc[i]
        plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {class_names[i]} (AUC = {roc_auc[i]:.2f})')

    media = total / (i+1)
    print("ROC-AUC medio :", media)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Curva ROC multiclass')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_multiclass.png')

if __name__ == "__main__":

    y_test = np.array([3,4,2,4,5,2,3,4,3,0,1,0,5,4,1,3,2,2,2,4,4,1,4,1,3,3,1,2,0,4,5,4,2,3,4,1,4,0,1,2,2,3,2,1,5,4,2,3,5,2])
    y_pred = np.array([3,4,1,4,5,1,3,4,0,3,1,0,0,4,2,4,1,2,5,4,4,1,3,1,3,4,1,2,0,4,5,4,2,3,4,1,4,0,1,2,2,3,2,1,2,4,2,3,5,2])
    class_names_str = ["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "schatches"]
    class_names = [0, 1, 2, 3, 4, 5]
    get_metrics_multiclass(y_test, y_pred, class_names, class_names_str)


    y_test2 = np.array([0,1,0,0,0,1,1,1,0,1,1,1,0,0,1,0,1,0,1,0])
    y_pred2 = np.array([0,0,1,0,1,1,1,1,0,0,1,1,0,1,1,0,1,0,1,1])
    get_metrics(y_test2, y_pred2)