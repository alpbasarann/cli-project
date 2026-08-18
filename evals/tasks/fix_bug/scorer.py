from evals.scorers import all_of, file_unchanged, pytest_passes


def score(ctx):
    return all_of(pytest_passes(), file_unchanged("test_parity.py"))(ctx)
