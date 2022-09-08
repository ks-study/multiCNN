import copy as cp
class TaskNames:
    """
    タスク名の配列を保存するクラス
    task_names:名義的に入力として与えるタスク名の配列
        e.g. ["diagnosis","mask","充実部単純CT濃度","早期相不均一さ"]

    radiologic_feature_names:所見特徴量名のみのタスク配列
        task_namesからdiagnosis,maskを除く
        e.g. ["充実部単純CT濃度","早期相不均一さ"]

    model_task_names:モデルが出力するタスク名の配列
        task_namesから必要に応じて"diagnosis_bin"系統を追加し、"diagnosis"を除く。
        e.g. ["diagnosis_bin0","diagnosis_bin1","mask","充実部単純CT濃度","早期相不均一さ"]

    result_task_names: 結果を保存するタスク名の配列
        task_namesに"diagnosis_bin"系統を追加し、"diagnosis"はそのまま。
        e.g. ["diagnosis_bin0","diagnosis_bin1","diagnosis_bin2","mask","充実部単純CT濃度","早期相不均一さ"]

    eval_task_names: 結果の評価値を保存するタスク名の配列
        task_namesに"diagnosis_bin"系統を追加し、"diagnosis"はそのまま。
        それぞれのタスクについて閾値最適化した場合のタスク名を追加する。
        maskは除く。
        e.g. ["diagnosis_bin0","diagnosis_bin0_opt","diagnosis_bin1","diagnosis_bin1_opt","diagnosis_bin2","diagnosis_bin2_opt",
        "diagnosis","diagnosis_opt","mask","mask_opt","充実部単純CT濃度","充実部単純CT濃度_opt","早期相不均一さ","早期相不均一さ_opt"]
    
    loader_id_task_names: データローダを個別で用意するタスク名の配列
        task_namesに"diagnosis_bin"系統を追加し、"diagnosis"はそのまま。"mask"は除去。
        e.g. ["diagnosis_bin0","diagnosis_bin1","diagnosis_bin2","diagnosis","充実部単純CT濃度","早期相不均一さ"]
    
    loader_out_task_names: データローダが出力するタスク名の配列
        task_namesに"diagnosis_bin"系統を追加し、"diagnosis"はそのまま。
        e.g. ["diagnosis_bin0","diagnosis_bin1","diagnosis_bin2","diagnosis","mask","充実部単純CT濃度","早期相不均一さ"]
    
    diagnosis_bin_task_names: サブタイプ分類を2値分類に落とし込んだそれぞれのタスク名の合計
        e.g. ["diagnosis_bin0","diagnosis_bin1","diagnosis_bin2"]
    
    """
    def __init__(self,task_names,binary=False,num_subtype=1,use_bce=False,use_multi_loaders=True):
        self.task_names=task_names
        
        self.radiologic_feature_names=cp.copy(self.task_names)
        if "diagnosis" in self.radiologic_feature_names:
            self.radiologic_feature_names.remove("diagnosis")
        if "mask" in self.radiologic_feature_names:
            self.radiologic_feature_names.remove("mask")
        self.model_task_names=cp.copy(self.task_names)
        if binary and "diagnosis" in self.model_task_names:
            insert_id=self.model_task_names.index("diagnosis")
            self.model_task_names[insert_id:insert_id+1]=[f"diagnosis_bin{i_bin}" for i_bin in range(num_subtype)]
        self.result_task_names=cp.copy(self.model_task_names)
        self.eval_task_names=cp.copy(self.model_task_names)
        if binary:
            insert_id=self.eval_task_names.index("diagnosis_bin0")
            self.eval_task_names[insert_id:insert_id]=["diagnosis"]
            self.result_task_names[insert_id:insert_id]=["diagnosis"]
        # if "diagnosis" in self.eval_task_names:
        #     insert_id=self.eval_task_names.index("diagnosis")
        #     self.eval_task_names[insert_id:insert_id]=["diagnosis_opt"]
        for task_name in self.model_task_names:
            insert_id=self.eval_task_names.index(task_name)
            if task_name!="mask":
                if "diagnosis" in task_name or not use_bce:
                    self.eval_task_names[insert_id:insert_id]=[task_name+"_opt"]
        # self.loader_id_task_names=cp.copy(self.model_task_names)
        if use_multi_loaders:
            self.loader_id_task_names=[task_name for task_name in self.model_task_names]
        else:
            self.loader_id_task_names=["diagnosis"]
        if "mask" in self.loader_id_task_names:
            self.loader_id_task_names.remove("mask")
        self.loader_out_task_names=cp.copy(self.model_task_names)
        if binary:
            insert_id=self.loader_out_task_names.index("diagnosis_bin0")
            self.loader_out_task_names[insert_id:insert_id]=["diagnosis"]
            self.loader_id_task_names[insert_id:insert_id]=["diagnosis"]
        self.diagnosis_bin_task_names=[task_name for task_name in self.model_task_names if "diagnosis_bin" in task_name]