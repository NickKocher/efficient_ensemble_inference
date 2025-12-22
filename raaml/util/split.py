import numpy as np
import os
import yaml


class RAAMLDataSplit:
    
    def __init__(self, X, y, stratify = False, n_cv=5, n_budget_splits=None, load_folder=None, seed=None):
        self.X = X
        self.y = y
        self.stratify = stratify
        self.n_cv = n_cv
        self.n_budget_splits = n_budget_splits
        self.seed = seed

        if len(X) != len(y):
            raise ValueError("Inconsistent lengths between features and target.")
        
        if load_folder is None:
            self._create_splits()
        else:
            self._load_splits(load_folder)

    def _create_splits(self):
        self.train_indices, self.ensembling_holdout_indices = split_indices(len(self.X), stratify=self.y if self.stratify else None, train_size=0.75, seed=self.seed)
        self.validation_folds = create_cv_splits(len(self.train_indices), self.n_cv, stratify=self.y.iloc[self.train_indices] if self.stratify else None, seed=self.seed)

        if self.n_budget_splits is not None:
            self.n_cv_steps = np.round(np.linspace(1, self.n_cv, self.n_budget_splits)).astype(int).tolist()
            self.budget_splits = [None] * (self.n_budget_splits - 1)
            for i, n_cv in enumerate(self.n_cv_steps[:-1]):
                self.budget_splits[i] = create_cv_splits(len(self.train_indices), n_cv, stratify=self.y.iloc[self.train_indices] if self.stratify else None, seed=self.seed)

        self.ensembling_validation_folds = create_cv_splits(len(self.X), self.n_cv, stratify=self.y if self.stratify else None, seed=self.seed)


    def _load_splits(self, load_folder):
        with open(os.path.join(load_folder, "meta.yaml"), 'r') as f:
            meta = yaml.safe_load(f)

        self.n_cv = meta["n_cv"]
        self.stratify = meta["stratify"]
        self.n_budget_splits = meta["n_budget_splits"]

        if meta["n_samples_full"] != len(self.X):
            raise ValueError("Inconsistent number of samples in X.")

        self.train_indices = np.load(os.path.join(load_folder, "train_indices.npy"))
        self.ensembling_holdout_indices = np.setdiff1d(np.arange(len(self.X)), self.train_indices)
        self.validation_folds = [np.load(os.path.join(load_folder, f"val_indices_{i}.npy")) for i in range(meta["n_cv"])]

        if meta["n_budget_splits"] is not None:
            self.n_cv_steps = meta["n_cv_steps"]
            self.budget_splits = []
            for i in range(meta["n_budget_splits"] - 1):
                budget_split = [np.load(os.path.join(load_folder, f"budget_split_{i}_val_indices_{j}.npy")) for j in range(meta["n_cv_steps"][i])]
                self.budget_splits.append(budget_split)
                
        self.ensembling_validation_folds = [np.load(os.path.join(load_folder, f"ensembling_val_indices_{i}.npy")) for i in range(meta["n_cv"])]
                

    def save(self, save_dir):
        meta = {
            "n_samples_full": len(self.X),
            "stratify" : self.stratify,
            "n_cv" : self.n_cv,
            "n_budget_splits" : self.n_budget_splits,
        }
        
        if self.n_budget_splits is not None:
            meta["n_cv_steps"] = self.n_cv_steps

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(os.path.join(save_dir, "meta.yaml"), 'w') as f:
            yaml.dump(meta, f)
        
        np.save(os.path.join(save_dir, "train_indices.npy"), self.train_indices)
        for i in range(self.n_cv):
            np.save(os.path.join(save_dir, f"val_indices_{i}.npy"), self.validation_folds[i])
            
        if self.n_budget_splits is not None:
            for i, n in enumerate(self.n_cv_steps[:-1]):
                for j in range(n):
                    np.save(os.path.join(save_dir, f"budget_split_{i}_val_indices_{j}.npy"), self.budget_splits[i][j])
                    
        for i in range(self.n_cv):
            np.save(os.path.join(save_dir, f"ensembling_val_indices_{i}.npy"), self.ensembling_validation_folds[i])
               
                    
    def get_cv_train_data(self, cv_index, n_cv=None, ensembling=False):
        if ensembling:
            if n_cv is not None:
                raise ValueError("Ensembling split is only available for n_cv=max_n_cv!")
            train_indices_mask = np.ones(len(self.X), dtype=bool)
            train_indices_mask[self.ensembling_validation_folds[cv_index]] = False
            return self.X.iloc[train_indices_mask], self.y.iloc[train_indices_mask]

        X_train, y_train = self.X.iloc[self.train_indices], self.y.iloc[self.train_indices]
        
        if n_cv is None or n_cv == self.n_cv:
            mask = np.ones(len(X_train), dtype=bool)
            mask[self.validation_folds[cv_index]] = False
            return X_train.iloc[mask], y_train.iloc[mask]
        else:
            index = self.n_cv_steps.index(n_cv)
            train_indices_mask = np.ones(len(X_train), dtype=bool)
            train_indices_mask[self.budget_splits[index][cv_index]] = False
            return X_train.iloc[train_indices_mask], y_train.iloc[train_indices_mask]

    def get_cv_val_data(self, cv_index, n_cv=None, ensembling=False):

        if ensembling:
            if n_cv is not None:
                raise ValueError("Ensembling split is only available for n_cv=max_n_cv!")

            return self.X.iloc[self.ensembling_validation_folds[cv_index]], self.y.iloc[self.ensembling_validation_folds[cv_index]]

        X_train, y_train = self.X.iloc[self.train_indices], self.y.iloc[self.train_indices]
        if n_cv is None or n_cv == self.n_cv:
            return X_train.iloc[self.validation_folds[cv_index]], y_train.iloc[self.validation_folds[cv_index]]
        else:
            index = self.n_cv_steps.index(n_cv)
            return X_train.iloc[self.budget_splits[index][cv_index]], y_train.iloc[self.budget_splits[index][cv_index]]
        
    def get_full_train_data(self):
        X_train, y_train = self.X.iloc[self.train_indices], self.y.iloc[self.train_indices]
        return X_train, y_train
    
    def get_ensembling_labels(self):
        all_indices = []
        for cv_index in range(self.n_cv):
            fold_indices = self.ensembling_validation_folds[cv_index]
            all_indices.extend(fold_indices)
        all_indices = np.array(all_indices)
        return self.y.iloc[all_indices]

def create_cv_splits(len_data, n_cv, stratify=None, seed=None):
    rng = np.random.default_rng(seed)
    
    # Handle stratification for all cases
    if stratify is not None:
        unique_classes = np.unique(stratify)
        splits = [[] for _ in range(n_cv)]
        
        # Handle each class separately
        for class_ in unique_classes:
            class_indices = np.where(stratify == class_)[0]
            rng.shuffle(class_indices)
            
            # If only one sample for this class, put this sample always into the 
            # training set
            if len(class_indices) == 1:
                continue

            if n_cv == 1:
                mid = len(class_indices) // 2
                splits[0].extend(class_indices[:mid])
            else:
                full_cycles, remainder = divmod(len(class_indices), n_cv)
                
                # assign the first (full_cycles * n_cv) samples round-robin
                for idx, sample_idx in enumerate(class_indices[:full_cycles * n_cv]):
                    splits[idx % n_cv].append(sample_idx)
                    
                folds_remainder = rng.choice(n_cv, size=remainder, replace=False)
                for i in range(remainder):
                    splits[folds_remainder[i]].append(class_indices[full_cycles * n_cv + i])
        
        for split in splits:
            rng.shuffle(split)
        
        return [np.array(split) for split in splits]
    
    # Non-stratified splits
    if n_cv == 1:
        indices = np.arange(len_data)
        rng.shuffle(indices)
        mid = len_data // 2
        return [indices[:mid]]
    
    indices = np.arange(len_data)
    rng.shuffle(indices)
    return np.array_split(indices, n_cv)


def split_indices(len_data, train_size=0.8, stratify=None, seed=None):
    rng = np.random.default_rng(seed) 

    if stratify is None:
        indices = np.arange(len_data)
        rng.shuffle(indices)  
        split_point = int(len_data * train_size)
        return indices[:split_point], indices[split_point:]
    else:
        unique_classes = np.unique(stratify)
        train_indices = []
        test_indices = []

        for cls in unique_classes:
            cls_indices = np.where(stratify == cls)[0]
            if len(cls_indices) == 1:
                train_indices.append(cls_indices[0])
                continue
            rng.shuffle(cls_indices)  
            split_point = int(round(len(cls_indices) * train_size))
            train_indices.extend(cls_indices[:split_point])
            test_indices.extend(cls_indices[split_point:])

        rng.shuffle(train_indices)
        rng.shuffle(test_indices)
        
        return np.array(train_indices), np.array(test_indices)


def save_splits(dir, train_val_split, default_cv_splits, budget_splits=None):
    if not os.path.exists(dir):
        os.makedirs(dir)
    np.save(os.path.join(dir, "train_split.npy"), train_val_split[0])
    np.save(os.path.join(dir, "val_split.npy"), train_val_split[1])
    
    # Save each CV split separately
    for i, split in enumerate(default_cv_splits):
        np.save(os.path.join(dir, f"default_split_{i}.npy"), split)
    
    if budget_splits is not None:
        for budget in budget_splits:
            for i, split in enumerate(budget_splits[budget]):
                np.save(os.path.join(dir, f"budget_splits_{budget}_{i}.npy"), split)

def load_splits(dir):
    # Load train and validation splits
    train_split = np.load(os.path.join(dir, "train_split.npy"))
    val_split = np.load(os.path.join(dir, "val_split.npy"))
    train_val_split = [train_split, val_split]
    
    # Load default CV splits
    default_splits = []
    i = 0
    while os.path.exists(os.path.join(dir, f"default_split_{i}.npy")):
        default_splits.append(np.load(os.path.join(dir, f"default_split_{i}.npy")))
        i += 1
    
    # Load budget splits if they exist
    budget_splits = {}
    for file in os.listdir(dir):
        if file.startswith("budget_splits_"):
            parts = file.split("_")
            budget = int(parts[2])
            split_index = int(parts[3].split(".")[0])
            
            if budget not in budget_splits:
                budget_splits[budget] = []
            budget_splits[budget].append(np.load(os.path.join(dir, file)))
            
    return train_val_split, default_splits, budget_splits


def print_class_counts(y):
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    print("Class distribution:", class_counts)