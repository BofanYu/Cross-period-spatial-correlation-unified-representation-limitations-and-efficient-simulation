"""Execute the article notebooks in reading order and save their cell outputs."""
from pathlib import Path
import argparse
import os
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    'model_fit/model_fit.ipynb',
    'psd/psd.ipynb',
    'predictive_accuracy/full.ipynb',
    'predictive_accuracy/complete.ipynb',
    'structural_flexibility/structural_flexibility.ipynb',
    'structural_flexibility/cpc.ipynb',
    'data_requirements/main.ipynb',
    'data_requirements/supp.ipynb',
    'computation/fit.ipynb',
    'computation/sampling.ipynb',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', nargs='+', choices=NOTEBOOKS,
                        help='Run selected notebooks; default runs all ten.')
    args = parser.parse_args()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    for relative in args.notebook or NOTEBOOKS:
        path = ROOT / 'scripts/analysis' / relative
        print(f'Running {relative}', flush=True)
        book = nbformat.read(path, as_version=4)
        NotebookClient(book, timeout=None, kernel_name='python3',
                       resources={'metadata': {'path': str(path.parent)}}).execute()
        nbformat.write(book, path)
    print('Notebook outputs updated. Figures: results/figures/')


if __name__ == '__main__':
    main()
