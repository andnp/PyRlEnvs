import glob
import os
from io import TextIOWrapper
import re
from typing import Dict, List

MODULES = ['domains', 'GridWorld', 'mazes']

doc_str: str = """# PyRlEnvs
"""

def getName(line: str):
    line = line.strip()
    line = line.replace('def ', '')
    line = line.replace('class ', '')
    line = re.sub(r'\W*\(.*\).*:*', '', line)
    line = line.replace(':', '')
    return line

def scanFile(f: TextIOWrapper):
    in_doc = False
    tabs = 0
    get_method = False
    buffer: List[str] = []
    total: Dict[str, List[str]] = {}
    for line in f.readlines():
        if get_method:
            if '@' in line:
                continue
            get_method = False
            name = getName(line)

            if name == '' and len(total) == 0:
                name = '-file-start'

            total[name] = buffer
            tabs = 0
            buffer = []
            continue

        if not in_doc and '"""doc' in line:
            in_doc = True
            # count the number of whitespaces to offset all lines in docs by
            tabs = len(re.match(r'\W*', line)[0]) - 3
            continue

        if not in_doc:
            continue

        if '"""' in line:
            in_doc = False
            get_method = True
            continue

        line = line[tabs:]
        buffer.append(line)

    return total

def grabModuleDocs(module: str):
    init = open(f'PyRlEnvs/{module}/__init__.py', 'r')
    init_str = ''
    start_read = False
    for line in init.readlines():
        if '"""doc' in line:
            start_read = True
            continue

        if '"""' in line:
            start_read = False
            continue

        if start_read:
            init_str += line

    init.close()

    return init_str

py_paths = glob.glob('PyRlEnvs/**/*.py', recursive=True)
py_paths = filter(lambda path: '__init__.py' not in path, py_paths)

split_paths: Dict[str, List[str]] = {}
for module in MODULES:
    split_paths[module] = []

for path in py_paths:
    parts = path.split('/')
    module = parts[1]
    arr = split_paths.get(module, [])
    arr.append(path)

for module in MODULES:
    doc_str += f"## {module}\n"

    init = open(f'PyRlEnvs/{module}/__init__.py', 'r')
    init_str = ''
    start_read = False
    for line in init.readlines():
        if '"""doc' in line:
            start_read = True
            continue

        if '"""' in line:
            start_read = False
            continue

        if start_read:
            init_str += line

    doc_str += init_str
    init.close()

    for path in split_paths[module]:
        f = open(path, 'r')
        docs = scanFile(f)
        for method in docs:
            if method == '-file-start':
                filename = os.path.basename(path).replace('.py', '')
                doc_str += f"## {filename}\n\n"
            else:
                doc_str += f"**{method}**:\n\n"

            doc_str += ''.join(docs[method]) + '\n\n'
        f.close()

print(doc_str)
# with open('README.md', 'w') as f:
#     f.write(doc_str)
