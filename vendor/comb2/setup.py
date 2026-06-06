from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension("src.ComboBase", ["src/ComboBase.py"]),
    Extension("src.DataRegistry", ["src/DataRegistry.py"]),
    Extension("src.DataLoader", ["src/DataLoader.py"]),
    Extension("src.op_utils", ["src/op_utils.py"]),
    Extension("src.selection", ["src/selection.py"]),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
    package_data={
        "": [
            "*.so",
            "*.pyd",
            "*.dll",
            "*.dylib",
        ]
    },
    exclude_package_data={
        "": [
            "*.c",
            "*.py",
        ]
    },
)
