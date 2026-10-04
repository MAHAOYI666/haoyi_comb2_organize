from pathlib import Path
from types import SimpleNamespace
import ast
import importlib.util
import tomllib

import pandas as pd
import pytest
import torch

from combo2.config import load_config
from combo2.paths import alpha_path
from comb_eval.report import check_config_outputs


def config_file(tmp_path, name=None):
    path = tmp_path / f"{name or 'legacy'}.xml"
    attr = f' Name="{name}"' if name is not None else ''
    path.write_text(f'<config{attr}><constants output_root="output" /></config>', encoding='utf-8')
    return path


def test_named_experiments_isolate_all_default_outputs(tmp_path):
    configs = [load_config(str(config_file(tmp_path, name))) for name in ('first', 'second')]
    for name, config in zip(('first', 'second'), configs):
        root = tmp_path / 'output' / name
        assert config['constants']['output_root'] == str(root)
        assert alpha_path(config) == root / f'{name}.parquet'
        assert Path(config['combo']['paths']['checkpoint_root']) == root / 'checkpoints'
        assert Path(config['backtest']['output_path']) == root / 'backtest'
        assert Path(config['combo']['output']['alpha_history_path']) == root / f'{name}.alpha_history.pt'
        assert Path(config['combo']['output']['log_path']) == root / f'{name}.train.log'
        assert Path(config['monitor']['output_path']) == root / f'{name}.perf_metrics.csv'
    assert not (tmp_path / 'output').exists()  # Configuration resolution is read-only.


@pytest.mark.parametrize('name', ['', '../escape', '/absolute', 'a/b', 'a\\b', '.', '..', 'a b', 'a.', 'a' * 129])
def test_invalid_names_are_rejected(tmp_path, name):
    path = tmp_path / 'invalid.xml'
    path.write_text(f'<config Name="{name}" />', encoding='utf-8')
    with pytest.raises(ValueError, match='Name'):
        load_config(str(path))


def test_legacy_configuration_keeps_original_paths(tmp_path):
    config = load_config(str(config_file(tmp_path)))
    assert alpha_path(config) == tmp_path / 'output' / 'alpha.parquet'
    assert Path(config['combo']['output']['log_path']) == tmp_path / 'output' / 'train.log'


def test_evaluation_never_picks_another_experiments_alpha(tmp_path):
    path = config_file(tmp_path, 'first')
    other = tmp_path / 'output' / 'second' / 'alpha.parquet'
    other.parent.mkdir(parents=True)
    other.write_bytes(b'not the requested experiment')
    status = check_config_outputs(path)
    assert status.files[0].path == tmp_path / 'output' / 'first' / 'first.parquet'
    assert not status.files[0].ok


def test_named_signal_producer_and_evaluator_agree(tmp_path, monkeypatch):
    from combo2 import runtime
    path = config_file(tmp_path, 'first')
    config = load_config(str(path))
    node = SimpleNamespace(alpha_history={(20240102, 100000): torch.tensor([1., 2.])})
    monkeypatch.setattr(runtime, 'IndexMask', lambda: SimpleNamespace(code=['000001', '000002']))
    def calculate(alpha, combo):
        return pd.DataFrame({'ic': [0.5], 'count': [2]}, index=alpha.index)
    monkeypatch.setattr(runtime, 'calculate_alpha_ic', calculate)
    runtime.dump_alpha_analysis(node, None, config)
    output = pd.read_parquet(alpha_path(config))
    assert output.index.names == ['date', 'time']
    assert output.iloc[0].tolist() == [1., 2.]
    assert check_config_outputs(path).files[0].ok
    root = alpha_path(config).parent
    assert (root / 'first.alpha_history.pt').is_file()
    assert (root / 'first.daily_ic.csv').is_file()
    assert not (root / 'alpha.parquet').exists()


def test_packaging_metadata_excludes_optuna_and_agrees_with_protected_build():
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / 'pyproject.toml').read_text())
    spec = importlib.util.spec_from_file_location('combo2_build_names', root / 'packaging/build_protected_wheel.py')
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    assert 'optuna_framework' not in metadata['tool']['setuptools']['packages']
    assert 'optuna_framework' not in build.PACKAGE_SOURCES
    assert not any('optuna' in dep.lower() or 'plotly' in dep.lower() for dep in metadata['project']['dependencies'])
    assert build.resolve_dependencies()['install_requires'] == metadata['project']['dependencies']
    assert build.ENTRY_POINTS == metadata['project']['scripts']


def test_runtime_has_no_dependency_on_evaluation_or_legacy_entrypoint():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / 'combo2/runtime.py').read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imports.append(node.module or '')
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
    assert not any(name == 'runCombo' or name.startswith('comb_eval') for name in imports)
