def main():
    raise SystemExit(
        "Legacy finetuning.py is disabled because it used the old non-SI train/test split "
        "and tuned against the test set. Use LOSO training via `python train_loso.py` instead."
    )


if __name__ == "__main__":
    main()
