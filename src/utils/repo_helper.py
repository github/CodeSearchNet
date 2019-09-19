from subprocess import check_call
from pathlib import Path
from typing import List
from json import loads
from requests import get

class Repo():
    """
    Because we don't have a good way to query the content of live repos.

    Example usage:

    # Instantiate object and retrieve code from repo: tensorflow/tensorflow
    > rc = Repo(org='tensorflow', repo_name='tensorflow', dest_path='/some/existing/folder')
    > rc.clone()  # gets the code by cloning if does not exist, or optionally pull the latest code.

    # returns list of Path objects in repo that end in '.py'
    > rc.get_filenames_with_extension('.py')
    """

    def __init__(self, org: str, repo_name: str, dest_path: str):
        self.metadata = self.__get_metadata(org, repo_name)
        assert Path(dest_path).is_dir(), f'Argument dest_path should be an existing directory: {dest_path}'
        self.dest_path = Path(dest_path)
        self.repo_save_folder = self.dest_path/self.metadata['full_name']

    def __get_metadata(self, org: str, repo_name: str) -> dict:
        "Validates github org and repo_name, and returns metadata about the repo."

        resp = get(f'https://api.github.com/repos/{org}/{repo_name}')
        resp.raise_for_status()
        info = loads(resp.text or resp.content)
        if 'clone_url' not in info:
            raise Exception(f'Cannot find repository {org}/{repo_name}')
        return info

    def clone(self, refresh: bool=True) -> Path:
        "Will clone a repo (default branch only) into the desired path, or if already exists will optionally pull latest code."
        default_branch = self.metadata['default_branch']
        clone_url = self.metadata['clone_url']

        if not self.repo_save_folder.exists():
            cmd = f'git clone --depth 1 -b {default_branch} --single-branch {clone_url} {str(self.repo_save_folder.absolute())}'
            print(f'Cloning repo:\n {cmd}')
            check_call(cmd, shell=True)

        elif refresh:
            cmd = f'git -C {str(self.repo_save_folder.absolute())} pull'
            print(f'Pulling latest code from repo:\n {cmd}')
            check_call(cmd, shell=True)

    def get_filenames_with_extension(self, extension: str='.py') -> List[Path]:
        "Return a list of filenames in the repo that end in the supplied extension."
        files = self.repo_save_folder.glob('**/*')
        return [f for f in files if f.is_file() and f.name.endswith(extension)]
