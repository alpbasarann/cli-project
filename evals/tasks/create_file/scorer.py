from evals.scorers import all_of, file_contains, file_exists


def score(ctx):
    return all_of(file_exists("greeting.txt"), file_contains("greeting.txt", "Hello from eval."))(ctx)
