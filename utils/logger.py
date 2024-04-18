def print_execution(fold_number, approach_name, name):
    print("\033[92m{}\033[00m"
          .format(f'Executing Fold {fold_number + 1} for {approach_name} with Model {name}'))


def print_evaluation(fold_number, approach_name, name, data):
    print("\033[94m{}\033[00m"
          .format(f'Evaluating Fold {fold_number + 1} for {approach_name} with Model {name} {data}'))



