import json
import re
import os

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load all locale files
locales = {}
for lang in ['zh', 'en', 'ja']:
    path = os.path.join(script_dir, 'locales', f'{lang}.json')
    with open(path, 'r', encoding='utf-8') as f:
        locales[lang] = json.load(f)

# Parse app.py to find all t() calls
app_path = os.path.join(script_dir, 'dashboard', 'app.py')
with open(app_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find all t('xxx') patterns
pattern = r"t\(['\"]([^'\"]+)['\"]\)"
all_keys = set(re.findall(pattern, content))

# Check each key
missing = {lang: [] for lang in locales}
for key in all_keys:
    parts = key.split('.')
    for lang, data in locales.items():
        value = data
        try:
            for part in parts:
                value = value[part]
            if not value or value == key:
                missing[lang].append(key)
        except KeyError:
            missing[lang].append(key)

# Report
for lang in locales:
    if missing[lang]:
        print(f'Missing/empty in {lang}.json:')
        for k in sorted(set(missing[lang])):
            print(f'  - {k}')
    else:
        print(f'{lang}.json: All keys present')
