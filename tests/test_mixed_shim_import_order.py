"""``sp.mixed`` stays the function whatever is imported first.

``statspai.multilevel.mixed`` is a back-compat shim module. Importing it
rebinds the package attribute ``statspai.multilevel.mixed`` to the module,
and the lazy export used to resolve ``sp.mixed`` through that attribute.
"""

import subprocess
import sys


def test_sp_mixed_is_the_function_after_the_shim_is_imported():
    code = (
        "import statspai.multilevel.mixed as shim\n"
        "import statspai as sp\n"
        "assert callable(sp.mixed) and not isinstance(sp.mixed, type(shim)), sp.mixed\n"
        "assert sp.MixedResult is shim.MixedResult\n"
        "from statspai.multilevel.mixed import mixed\n"
        "assert mixed.__name__ == 'mixed'\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
