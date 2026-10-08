"""Compare source-only training with existing pooled OOF predictions."""
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from pvt_ml.data import FEATURES, load_data, validate_data
from pvt_ml.evaluation import metrics
from pvt_ml.models import model_spec
from pvt_ml.results import environment, save_json
from pvt_ml.tuning import select_model


def _read_run(path):
    path = Path(path)
    if path.is_dir():
        return {n: (path / n).read_bytes() for n in
                ('metadata.json', 'splits.json', 'predictions.csv')}
    with ZipFile(path) as archive:
        result = {}
        for name in ('metadata.json', 'splits.json', 'predictions.csv'):
            matches = [p for p in archive.namelist()
                       if not p.startswith('__MACOSX/') and Path(p).name == name]
            if len(matches) != 1:
                raise ValueError(f'Expected exactly one {name} in {path}')
            result[name] = archive.read(matches[0])
        return result


def run_pooling_comparison(data_path, source_paths, pooled_run, output,
                           models=None):
    """Retune each family within source-only training, on matched outer tests.

    source_paths maps anonymous labels to component CSV/XLS files. Original
    grids, seed and restricted inner splits come from the existing pooled run.
    Completed model/fold tasks are reused when the input manifest matches.
    """
    files = _read_run(pooled_run)
    meta = json.loads(files['metadata.json'])
    splits = json.loads(files['splits.json'])
    pooled = pd.read_csv(io.BytesIO(files['predictions.csv']))
    if meta.get('status') != 'complete':
        raise ValueError('The pooled run must be complete')
    data_path = Path(data_path)
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != meta['source']['file_sha256']:
        raise ValueError('Data bytes differ from the original pooled-run input')
    data, target = validate_data(load_data(data_path), meta['target'])
    models = tuple(meta['models'] if models is None else models)
    if not models or len(set(models)) != len(models) or set(models) - set(meta['models']):
        raise ValueError('Models must be unique families evaluated in the pooled run')
    # Map source membership using all four inputs AND Pb. Bob is available
    # in the combined file even when the component files contain only Pb.
    identity, _ = validate_data(load_data(data_path), 'Pb')
    key_cols = FEATURES + ['Pb']
    membership = {}
    source_hashes = {}
    for label, path in source_paths.items():
        if not isinstance(label, str) or not label or not all(c.isalnum() or c == '_' for c in label):
            raise ValueError('Source labels must contain only letters, digits and underscores')
        path = Path(path)
        source_hashes[label] = hashlib.sha256(path.read_bytes()).hexdigest()
        component, _ = validate_data(load_data(path), 'Pb')
        for row in component[key_cols].itertuples(index=False, name=None):
            if row in membership:
                raise ValueError('Ambiguous/repeated record in component datasets')
            membership[row] = label
    keys = list(identity[key_cols].itertuples(index=False, name=None))
    if len(set(keys)) != len(keys) or set(keys) != set(membership):
        raise ValueError('Component records must exactly partition combined data')
    groups = np.array([membership[row] for row in keys])
    tests = []
    for sp in splits:
        tr, te = set(sp['train_positions']), set(sp['test_positions'])
        if tr & te or tr | te != set(range(len(data))):
            raise ValueError('Invalid outer split')
        tests.extend(te)
        for inner in sp['inner_splits']:
            a, b = set(inner['train_positions']), set(inner['validation_positions'])
            if a & b or a | b != tr:
                raise ValueError('Invalid inner split')
    if sorted(tests) != list(range(len(data))):
        raise ValueError('Outer tests must cover every row exactly once')
    expected_fold = {p: sp['fold'] for sp in splits for p in sp['test_positions']}
    for model in models:
        p = pooled[pooled.model == model]
        if len(p) != len(data) or p.row_position.nunique() != len(data):
            raise ValueError('Incomplete pooled predictions')
        if not np.allclose(p.actual, data[target].to_numpy()[p.row_position]):
            raise ValueError('Pooled labels do not match input')
        if any(expected_fold[int(r.row_position)] != r.fold for r in p.itertuples()):
            raise ValueError('Pooled prediction fold mismatch')
        metrics(p.actual, p.predicted)
    signature = {
        'target': target, 'data_file_sha256': meta['source']['file_sha256'],
        'component_sha256': source_hashes, 'models': list(models),
        'pooled_input_sha256': {n: hashlib.sha256(b).hexdigest() for n, b in files.items()},
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'seed': meta['seed'], 'search_spaces': {m: meta['search_spaces'][m] for m in models},
        'environment': environment(),
    }
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'comparison_manifest.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old['inputs'] != signature:
            raise ValueError('Existing output belongs to different inputs/environment; use a fresh folder')
    elif any(out.iterdir()):
        raise ValueError('Output folder must be empty for a new run')
    save_json(manifest, {'inputs': signature, 'status': 'running'})
    pd.DataFrame({'row_position': range(len(data)), 'source_dataset': groups}).to_csv(
        out / 'source_membership.csv', index=False)
    individual = []
    try:
        for label in source_paths:
            for name in models:
                estimator, _ = model_spec(name, meta['seed'])
                grid = meta['search_spaces'][name]
                for sp in splits:
                    fold = sp['fold']
                    task = out / label / name / f'fold{fold}'
                    done = task / 'complete.json'
                    if done.exists():
                        p = pd.read_csv(task / 'predictions.csv')
                        print(f'{target} {label} {name} fold {fold}: reused', flush=True)
                    else:
                        tr = np.array([i for i in sp['train_positions'] if groups[i] == label])
                        te = np.array([i for i in sp['test_positions'] if groups[i] == label])
                        local = {int(pos): i for i, pos in enumerate(tr)}
                        inner = []
                        inner_records = []
                        for s in sp['inner_splits']:
                            a = [i for i in s['train_positions'] if groups[i] == label]
                            b = [i for i in s['validation_positions'] if groups[i] == label]
                            if len(a) < 2 or len(b) < 2:
                                raise ValueError('Too few source observations in an inner split')
                            inner.append((np.array([local[i] for i in a]),
                                          np.array([local[i] for i in b])))
                            inner_records.append({'train_positions': a, 'validation_positions': b})
                        if len(te) < 2:
                            raise ValueError('Too few source observations in an outer test')
                        print(f'{target} {label} {name} fold {fold}: training ({len(tr)} rows)', flush=True)
                        task.mkdir(parents=True, exist_ok=True)
                        save_json(task / 'splits.json', {
                            'train_positions': tr.tolist(), 'test_positions': te.tolist(),
                            'inner_splits': inner_records})
                        fitted, params, score, search, audits = select_model(
                            estimator, grid, data[FEATURES].iloc[tr], data[target].iloc[tr], inner, tr)
                        prediction = fitted.predict(data[FEATURES].iloc[te])
                        metrics(data[target].iloc[te], prediction)
                        p = pd.DataFrame({'model': name, 'fold': fold, 'row_position': te,
                                          'actual': data[target].iloc[te].to_numpy(), 'predicted': prediction})
                        p.to_csv(task / 'predictions.csv', index=False)
                        search.to_csv(task / 'search.csv', index=False)
                        save_json(task / 'fit_audit.json', audits)
                        save_json(task / 'selected_parameters.json', {
                            'params': params, 'inner_selection_rmse': score,
                            'resolved_parameters': {k: repr(v) for k, v in fitted.get_params(deep=True).items()}})
                        save_json(done, {'status': 'complete'})
                    p['source_dataset'] = label
                    p['training'] = 'individual'
                    individual.append(p)
        pooled = pooled[pooled.model.isin(models)].copy()
        pooled['source_dataset'] = groups[pooled.row_position]
        pooled['training'] = 'pooled'
        predictions = pd.concat([pooled, *individual], ignore_index=True)
        rows = []
        fold_rows = []
        for (label, name, training), p in predictions.groupby(['source_dataset', 'model', 'training']):
            rows.append({'source_dataset': label, 'model': name, 'training': training,
                         'n': len(p), **metrics(p.actual, p.predicted)})
            for fold, q in p.groupby('fold'):
                fold_rows.append({'source_dataset': label, 'model': name, 'training': training,
                                  'fold': fold, 'n': len(q), **metrics(q.actual, q.predicted)})
        table = pd.DataFrame(rows)
        wide = table.pivot(index=['source_dataset', 'model'], columns='training', values='rmse')
        wide['pooled_rmse_improvement_percent'] = 100 * (wide.individual - wide.pooled) / wide.individual
        predictions.to_csv(out / 'predictions.csv', index=False)
        table.to_csv(out / 'metrics_by_source.csv', index=False)
        pd.DataFrame(fold_rows).to_csv(out / 'fold_metrics_by_source.csv', index=False)
        wide.to_csv(out / 'pooling_comparison.csv')
        save_json(manifest, {'inputs': signature, 'status': 'complete'})
        return wide
    except Exception as error:
        save_json(manifest, {'inputs': signature, 'status': 'failed',
                             'error': f'{type(error).__name__}: {error}'})
        raise
