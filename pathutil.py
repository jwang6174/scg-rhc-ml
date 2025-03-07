import os
import shutil

# SCG-RHC database path.
DATABASE_PATH = os.path.join('/', 'home', 'jesse', 'scg-rhc-database')


def clear_dir(paths):
  """
  Clear directories.

  Args:
    paths (list[str]): Directory paths to clear.
  """
  for path in paths:
    if os.path.exists(path):
      shutil.rmtree(path)
      os.makedirs(path)
      print(f'Cleared {path}')