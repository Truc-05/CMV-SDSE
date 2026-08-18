Symlink or git submodule pointing at the minimal-mas-failure-modes project.

Real setup:
  rm -rf shared_testbed
  ln -s ../minimal-mas-failure-modes shared_testbed

or as a submodule:
  git submodule add <url-of-minimal-mas-failure-modes> shared_testbed

The cascade falls back to ../minimal-mas-failure-modes/data/raw_runs when this
directory holds only this placeholder, so a symlink is optional on a machine
where both projects sit side by side.
