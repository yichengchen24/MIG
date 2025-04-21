def len_jsonls(files) -> int:
    """Return the total number of samples in jsonl files."""
    if not isinstance(files, list):
        files = [files]  # type: ignore
    return sum(sum(1 for _ in open(file, encoding="utf-8")) for file in files)
