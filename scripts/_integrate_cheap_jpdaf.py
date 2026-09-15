from pathlib import Path
import subprocess
import sys


def replace_once(path, old, new):
    path = Path(path)
    text = path.read_text()
    if text.count(old) != 1:
        raise RuntimeError(f'Expected one integration anchor in {path}: {old!r}')
    path.write_text(text.replace(old, new, 1))


renderers = {
    'docs/backend-api-matrix.md': 'scripts/render_backend_api_matrix.py',
    'docs/public-api-registry.md': 'scripts/check_public_api_registry.py',
}
old_tables = {doc: subprocess.check_output([sys.executable, script], text=True)
              for doc, script in renderers.items()}

path = Path('src/pyrecest/filters/joint_probabilistic_data_association_filter.py')
text = path.read_text()
start = text.index('        track_order = sorted(')
end = text.index('        self.latest_association_probabilities = association_probabilities', start)
solver = text[start:end]
replacement = '''        association_probabilities, map_association = self._compute_association_probabilities(
            log_likelihoods,
            eligible_measurements,
            detection_probability,
            clutter_intensity,
        )

'''
text = text[:start] + replacement + text[end:]
method = '''    def _compute_association_probabilities(
        self,
        log_likelihoods,
        eligible_measurements,
        detection_probability,
        clutter_intensity,
    ):
        """Solve the gated association problem by exact joint-event enumeration."""
        n_targets, n_meas = log_likelihoods.shape
'''
method += solver + '        return association_probabilities, map_association\n\n'
anchor = '    def find_association(\n'
assert text.count(anchor) == 1
text = text.replace(anchor, method + anchor, 1)
path.write_text(text)

names = ('CheapJointProbabilisticDataAssociationFilter', 'CheapJPDAF', 'CJPDAF')
exports = ''.join(f'    "{name}": ".cheap_joint_probabilistic_data_association_filter",\n' for name in names)
anchor = '    "JPDAF": ".joint_probabilistic_data_association_filter",\n'
replace_once('src/pyrecest/filters/__init__.py', anchor, anchor + exports)

capabilities = ''.join(
    f'''    "{name}": {{
        "numpy": "supported",
        "pytorch": "unsupported",
        "jax": "unsupported",
        "notes": "Normalized cheap JPDA for linear-Gaussian models; no joint-event enumeration.",
    }},
''' for name in names)
replace_once('src/pyrecest/_backend/capabilities.py', 'API_BACKEND_CAPABILITIES: Final = {\n',
             'API_BACKEND_CAPABILITIES: Final = {\n' + capabilities)
registry = ''.join(
    f'''    "{name}": {{
        "module": "pyrecest.filters",
        "category": "experimental",
        "backend_contract": "{name}",
        "notes": "Fitzgerald-style cheap JPDA with complementary missed-detection mass; NumPy only.",
    }},
''' for name in names)
replace_once('src/pyrecest/api_registry.py', 'PUBLIC_API_REGISTRY: Final = {\n',
             'PUBLIC_API_REGISTRY: Final = {\n' + registry)
for doc, script in renderers.items():
    new_table = subprocess.check_output([sys.executable, script], text=True)
    replace_once(doc, old_tables[doc], new_table)

with Path('docs/api-overview.md').open('a') as stream:
    stream.write('\n## Cheap joint probabilistic data association\n\n'
                 '`CheapJPDAF` / `CJPDAF` provides normalized soft association for\n'
                 'linear-Gaussian tracking without joint-event enumeration. It reuses\n'
                 'the exact `JPDAF` Gaussian update and is explicitly NumPy-only.\n'
                 'See [Cheap JPDAF](cheap-jpdaf.md) for usage, normalization,\n'
                 'complexity, and the non-MAP greedy diagnostic.\n')
with Path('docs/backend-compatibility.md').open('a') as stream:
    stream.write('\n### Cheap JPDAF\n\n'
                 '`CheapJPDAF`, `CJPDAF`, and\n'
                 '`CheapJointProbabilisticDataAssociationFilter` are NumPy-only.\n'
                 'Association and measurement updates reject other backends explicitly.\n'
                 'See [Cheap JPDAF](cheap-jpdaf.md) for the approximation contract.\n')
path = Path('tests/filters/test_cheap_joint_probabilistic_data_association_filter.py')
text = path.read_text()
path.write_text('# pylint: disable=protected-access,no-name-in-module,no-member\n' + text)
replace_once(
    path,
    '    tracker.filter_state = []\n',
    '    # Isolate association caches from the inherited empty-bank history logger.\n'
    '    tracker.log_prior_estimates = False\n'
    '    tracker.filter_state = []\n',
)
