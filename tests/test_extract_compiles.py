"""The extraction modules cannot run outside the Workbench (BigQuery), but they must at least compile,
and importing the package must not trigger them."""
import py_compile
from pathlib import Path

import delphi_aou.data.extract as ext  # noqa: F401  (package import must not run queries)


def test_extract_modules_compile():
    root = Path(ext.__file__).parent
    for name in ["core.py", "extras.py", "sdoh.py"]:
        py_compile.compile(str(root / name), doraise=True)
