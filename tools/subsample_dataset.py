import os
import git
import argparse
import pandas as pd

def get_git_root(path):

        git_repo = git.Repo(path, search_parent_directories=True)
        git_root = git_repo.git.rev_parse("--show-toplevel")
        return git_root


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="data/train256/", help="Source directory")
    parser.add_argument('--destination', type=str, default=None)
    parser.add_argument('--n', type=int, default=100, help="Nr of samples per shape")
    args = parser.parse_args()


    git_root = get_git_root(".")
    os.chdir(git_root)
    print(git_root)

    source = args.source
    destination = args.destination

    if destination is None:
        destination = source

    if not os.path.exists(destination):
        os.makedirs(destination)

    # update and save csv
    df = pd.read_csv(os.path.join(source, "labels.csv"))
    df_sub = df.groupby("shape_name").head(args.n)
    df_sub.to_csv(os.path.join(destination, f"labels_sub{args.n}.csv"), index=False)