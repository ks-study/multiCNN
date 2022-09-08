class EvalTaskMap():
    """
        coralだったら2値分類の評価値を記録しない
        閾値選択後の分類だったら2値分類の評価値は記載しない。
        maskだったら特殊な評価値算出を行う。

        一般の分類タスク：
            "precision":precision_label,
            "sensitivity":sensitivity_label,
            "specificity":specificity_label,
            "tpr_optimal":tpr_optimal_label,
            "fpr_optimal":fpr_optimal_label,
            "acc_optimal":acc_optimal_label,
            "auroc":auroc_label,
            "auprc":auprc_label,
            "threshold":thresholds_label

        roc関連を除く分類タスク：
            "precision":precision_label,
            "sensitivity":sensitivity_label,
            "specificity":specificity_label,

        maskタスク：
            "IoU",
            "precision",
            "sensitivity",
            "specificity", 
    """
    def __init__(self,tn,coral=False,binary=False):
        for task_name in tn.eval_task_names:
            if coral and tn.radiologic_feature_names:
                pass

        return None