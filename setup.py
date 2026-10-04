"""Copy the canonical interface manual into the source distribution's wheel."""
from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
    def run(self):
        super().run()
        target = Path(self.build_lib) / "comb2_templates" / "config.human"
        target.parent.mkdir(parents=True, exist_ok=True)
        self.copy_file("config.human", str(target))


setup(cmdclass={"build_py": build_py})
