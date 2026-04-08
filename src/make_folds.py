from typing import Dict, List

from family_map import FAMILY_NAMES, FAMILY_TO_BAGS


def make_leave_one_family_out_folds() -> List[Dict]:
    families = list(FAMILY_NAMES)
    n = len(families)

    if n < 3:
        raise ValueError("Need at least 3 families to build train/val/test folds.")

    folds = []

    for i, test_family in enumerate(families):
        val_family = families[(i + 1) % n]

        train_families = [
            fam for fam in families
            if fam != test_family and fam != val_family
        ]

        test_bags = FAMILY_TO_BAGS[test_family]
        val_bags = FAMILY_TO_BAGS[val_family]

        train_bags = []
        for fam in train_families:
            train_bags.extend(FAMILY_TO_BAGS[fam])
        train_bags = sorted(train_bags)

        fold = {
            "fold_id": i,
            "test_family": test_family,
            "val_family": val_family,
            "train_families": train_families,
            "test_bags": sorted(test_bags),
            "val_bags": sorted(val_bags),
            "train_bags": train_bags,
        }
        folds.append(fold)

    return folds


def get_fold_by_id(fold_id: int) -> Dict:
    folds = make_leave_one_family_out_folds()
    if fold_id < 0 or fold_id >= len(folds):
        raise IndexError(f"fold_id {fold_id} out of range [0, {len(folds)-1}]")
    return folds[fold_id]# training/make_folds.py

from typing import Dict, List

from family_map import FAMILY_NAMES, FAMILY_TO_BAGS


def make_leave_one_family_out_folds() -> List[Dict]:
    """
    Build grouped cross-validation folds.

    For each fold:
        - one family is used as test
        - the next family (circularly) is used as validation
        - all remaining families are used for training

        Returns:
            A list of dicts, one per fold, with:
            - fold_id
            - test_family
            - val_family
            - train_families
            - test_bags
            - val_bags
            - train_bags
        """
    families = list(FAMILY_NAMES)
    n = len(families)

    if n < 3:
        raise ValueError("Need at least 3 families to build train/val/test folds.")

    folds = []

    for i, test_family in enumerate(families):
        val_family = families[(i + 1) % n]

        train_families = [
            fam for fam in families
            if fam != test_family and fam != val_family
        ]

        test_bags = FAMILY_TO_BAGS[test_family]
        val_bags = FAMILY_TO_BAGS[val_family]

        train_bags = []
        for fam in train_families:
            train_bags.extend(FAMILY_TO_BAGS[fam])
        train_bags = sorted(train_bags)

        fold = {
            "fold_id": i,
            "test_family": test_family,
            "val_family": val_family,
            "train_families": train_families,
            "test_bags": sorted(test_bags),
            "val_bags": sorted(val_bags),
            "train_bags": train_bags,
        }
        folds.append(fold)

    return folds


def get_fold_by_id(fold_id: int) -> Dict:
    folds = make_leave_one_family_out_folds()
    if fold_id < 0 or fold_id >= len(folds):
        raise IndexError(f"fold_id {fold_id} out of range [0, {len(folds)-1}]")
    return folds[fold_id]


def print_folds():
    folds = make_leave_one_family_out_folds()
    for fold in folds:
        print(f"\nFold {fold['fold_id']}")
        print(f"  Test family: {fold['test_family']}")
        print(f"  Val family:  {fold['val_family']}")
        print(f"  Train families: {fold['train_families']}")
        print(f"  Test bags: {fold['test_bags']}")
        print(f"  Val bags:  {fold['val_bags']}")
        print(f"  Train bags: {fold['train_bags']}")


if __name__ == "__main__":
    print_folds()