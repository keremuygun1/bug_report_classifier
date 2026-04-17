from scipy.stats import wilcoxon
import pandas as pd

#1. Import the required documents (5 from NB baseline an 5 from LR imrpoved)

datasets = ["tensorflow","caffe","pytorch","incubator-mxnet","keras"]

#1.1 Create arrays for each result
accuracies  = []
precisions  = []
recalls     = []
f1_scores   = []
auc_values  = []

#1.2 Create names for the wilcoxon pvalue files
out_file_name = f"wilcoxon_pvalues.csv"

#1.1 Loop thorugh the datasets for both
for dataset in datasets:

    file_nb = f'NB_results/{dataset}_NB.csv'
    file_lr = f'LR_results/{dataset}_LR.csv'

    df_nb = pd.read_csv(file_nb)
    df_lr = pd.read_csv(file_lr)

    #2. Extract each metric and have a 10 element array

    accuracy_nb = df_nb["Accuracies"]
    accuracy_lr = df_lr["Accuracies"]

    # 2.1 Calculate the Wilcoxon hypothesis in a one sided test for the baseline is less than the latter.

    acc_res = wilcoxon(accuracy_nb, accuracy_lr, alternative="less")
    pval_acc = acc_res.pvalue
    accuracies.append(pval_acc)

    precision_nb = df_nb["Precisions"]
    precision_lr = df_lr["Precisions"]

    pre_res = wilcoxon(precision_nb, precision_lr, alternative="less")
    pval_pre = pre_res.pvalue
    precisions.append(pval_pre)

    recall_nb = df_nb["Recalls"]
    recall_lr = df_lr["Recalls"]

    rec_res = wilcoxon(recall_nb, recall_lr, alternative="less")
    pval_rec = rec_res.pvalue
    recalls.append(pval_rec)

    f1_nb = df_nb["F1s"]
    f1_lr = df_lr["F1s"]

    f1_res = wilcoxon(f1_nb, f1_lr, alternative="less")
    pval_f1 = f1_res.pvalue
    f1_scores.append(pval_f1)

    auc_nb = df_nb["AUCs"]
    auc_lr = df_lr["AUCs"]

    auc_res = wilcoxon(auc_nb, auc_lr, alternative="less")
    pval_auc = auc_res.pvalue
    auc_values.append(pval_auc)

    #3. Store and Print every metric.

    print(f"======= Wilcoxon P-value results for {dataset} =======")
    print(f"Accuracy:      {pval_acc:.6f}")
    print(f"Precision:     {pval_pre:.6f}")
    print(f"Recall:        {pval_rec:.6f}")
    print(f"F1:             {pval_f1:.6f}")
    print(f"AUC:           {pval_auc:.6f}")

#3.1 Create a dataframe and store the results as csv using pandas.

results_df = pd.DataFrame(
        {
        "Project": datasets,
        "Accuracy": accuracies,
        "Precision": precisions,
        "Recall": recalls,
        "F1": f1_scores,
        "AUC": auc_values
    }
)

#3.2 open pandas and store the dataframe inside a csv file

results_df.to_csv(out_file_name, mode='w', index=False)

print(f"\nResults have been saved to: {out_file_name}")
    