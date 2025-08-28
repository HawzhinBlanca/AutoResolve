import os, re, sys, json, hashlib, time, ast
from collections import defaultdict, Counter
from datetime import datetime, timedelta

ROOT = '/Users/hawzhin/AutoResolve'
NOW = time.time()
THIRTY_DAYS = 30*24*3600

MEDIA_EXTS = {'.mp4','.mov','.wav','.npz','.png','.jpg','.jpeg','.gif','.tiff','.bmp'}
TEXT_EXTS = {'.py','.swift','.md','.ini','.json','.sh','.txt','.yaml','.yml','.toml','.cfg','.swift','.gitignore'}
CONFIG_EXTS = {'.ini','.yaml','.yml','.toml','.cfg','.json'}
DOC_EXTS = {'.md','.rst'}

SAFE_NUMS = {-1,0,1}

# Utilities

def read_text(path, max_bytes=2_000_000):
    try:
        size = os.path.getsize(path)
        if size > max_bytes:
            return None
        with open(path,'r',encoding='utf-8',errors='ignore') as f:
            return f.read()
    except Exception:
        return None


def line_count(path):
    try:
        with open(path,'r',encoding='utf-8',errors='ignore') as f:
            return sum(1 for _ in f)
    except Exception:
        return None


def file_hash(path, normalize_whitespace=False):
    try:
        if normalize_whitespace and os.path.splitext(path)[1].lower() in TEXT_EXTS:
            txt = read_text(path)
            if txt is None:
                return None
            txt = re.sub(r"\s+"," ", txt)
            return hashlib.md5(txt.encode('utf-8')).hexdigest()
        else:
            h = hashlib.md5()
            with open(path,'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    h.update(chunk)
            return h.hexdigest()
    except Exception:
        return None


def tokenize_code(text):
    if not text:
        return []
    # crude tokenization
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\S", text)
    return [t for t in tokens if t.strip()]


def jaccard(a, b):
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter/union if union else 0.0

# Phase 1: FS scan
files = []
empty_dirs = []
deep_dirs = []
wrong_loc_media = []
nonstandard_names = []

for dp, dns, fns in os.walk(ROOT):
    # skip .git internals from heavy processing but still record
    rel = os.path.relpath(dp, ROOT)
    depth = 0 if rel == '.' else rel.count(os.sep) + 1
    if depth > 5:
        deep_dirs.append(dp)
    if not fns and not dns:
        empty_dirs.append(dp)
    for fn in fns:
        p = os.path.join(dp, fn)
        try:
            st = os.stat(p)
            size = st.st_size
            ext = os.path.splitext(fn)[1].lower()
            is_media = ext in MEDIA_EXTS
            is_text = ext in TEXT_EXTS
            lc = line_count(p) if is_text and size < 2_000_000 else None
            if is_media and not any(seg in dp for seg in (
                'assets','artifacts','models','AutoResolveUI/DerivedData','DerivedData','AutoResolveUI/Sources/AutoResolveUI')):
                wrong_loc_media.append(p)
            if re.search(r"[A-Z ]", fn) and ext in TEXT_EXTS:
                nonstandard_names.append(p)
            files.append({
                'path': p,
                'rel': os.path.relpath(p, ROOT),
                'size': size,
                'ext': ext,
                'dir': dp,
                'depth': depth,
                'is_media': is_media,
                'is_text': is_text,
                'lines': lc,
                'mtime': st.st_mtime,
            })
        except Exception:
            continue

# Classification aggregates
classification = {
    'SOURCE_CODE': defaultdict(list),
    'CONFIGS': defaultdict(list),
    'ASSETS': defaultdict(list),
    'DOCUMENTATION': defaultdict(list),
    'JUNK': defaultdict(list),
}

for f in files:
    ext = f['ext']
    p = f['path']
    if ext in {'.py','.swift','.sh'}:
        classification['SOURCE_CODE'][ext].append(p)
    elif ext in CONFIG_EXTS:
        classification['CONFIGS'][ext].append(p)
    elif ext in MEDIA_EXTS:
        classification['ASSETS'][ext].append(p)
    elif ext in DOC_EXTS:
        classification['DOCUMENTATION'][ext].append(p)
    # Junk heuristics
    if any(seg in p for seg in ('DerivedData','ModuleCache.noindex','CompilationCache.noindex')) or \
       os.path.basename(p) in {'.DS_Store'}:
        classification['JUNK']['junk'].append(p)

# Phase 2: code intelligence (Python + Swift)

def analyze_python(path, text):
    result = {
        'purpose': None,
        'imports': [],
        'unused_imports': [],
        'exports': [],
        'complexity': 0,
        'smells': [],
        'critical_functions': {},
        'classes': {},
        'print_statements': 0,
        'error_swallows': [],
        'god_objects': [],
    }
    if text is None:
        return result
    try:
        tree = ast.parse(text)
    except Exception:
        return result

    # Purpose: module docstring first line
    doc = ast.get_docstring(tree)
    if doc:
        result['purpose'] = doc.strip().splitlines()[0][:200]

    # Imports and usage
    imported_names = {}
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
                imported_names[alias.asname or alias.name.split('.')[-1]] = alias.name
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ''
            for alias in node.names:
                fullname = f"{mod}.{alias.name}" if mod else alias.name
                imports.append(fullname)
                imported_names[alias.asname or alias.name] = fullname

    # usage
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            used.add(node.value.id)
    unused = [k for k in imported_names.keys() if k not in used]
    result['imports'] = imports
    result['unused_imports'] = unused

    # Exports
    exports = []
    classes = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            exports.append(node.name)
        elif isinstance(node, ast.ClassDef):
            exports.append(node.name)
            # count methods
            methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
            classes[node.name] = {
                'method_count': len(methods),
                'line_count': getattr(node, 'end_lineno', None) and getattr(node, 'end_lineno') - node.lineno + 1,
            }
    result['exports'] = exports
    result['classes'] = classes

    # Complexity: count decision points
    complexity = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.BoolOp, ast.IfExp)):
            complexity += 1
        if isinstance(node, ast.Try):
            complexity += len(node.handlers)
    result['complexity'] = complexity

    # Smells and critical functions
    smells = []
    critical = {}
    def node_len(n):
        try:
            return getattr(n,'end_lineno', n.lineno) - n.lineno + 1
        except Exception:
            return None

    max_nesting = 0
    def walk_with_depth(n, d=0):
        nonlocal max_nesting
        max_nesting = max(max_nesting, d)
        for c in ast.iter_child_nodes(n):
            nd = d + (1 if isinstance(n, (ast.If, ast.For, ast.While, ast.Try, ast.With)) else 0)
            walk_with_depth(c, nd)
    walk_with_depth(tree, 0)
    if max_nesting >= 4:
        smells.append(f'deep_nesting:{max_nesting}')

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            length = node_len(node) or 0
            if length >= 80:
                smells.append(f'long_function:{node.name}:{length}')
            # magic numbers
            magic_count = 0
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, (int, float)) and sub.value not in SAFE_NUMS:
                    magic_count += 1
            if magic_count >= 5:
                smells.append(f'magic_numbers:{node.name}:{magic_count}')
            critical[node.name] = {'what': (ast.get_docstring(node) or '').strip().splitlines()[0] if ast.get_docstring(node) else ''}

    # print statements
    prints = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'print':
            prints += 1
    result['print_statements'] = prints

    # error swallowing
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            body_src = ''.join(ast.get_source_segment(text, b) or '' for b in node.body)
            if not body_src.strip() or re.search(r"\bpass\b", body_src):
                result['error_swallows'].append({'lineno': node.lineno, 'type': getattr(node.type, 'id', str(node.type)) if node.type else None})

    # god objects: large classes
    for cname, meta in classes.items():
        if (meta['line_count'] or 0) > 500 or meta['method_count'] > 20:
            result['god_objects'].append(cname)

    return result


def analyze_swift(path, text):
    result = {
        'purpose': None,
        'imports': [],
        'exports': [],
        'complexity': 0,
        'smells': [],
        'critical_functions': {},
        'print_statements': 0,
        'error_swallows': [],
        'god_objects': [],
    }
    if not text:
        return result
    # Purpose: first comment line
    m = re.search(r"/\*[\s\S]*?\*/|//.*", text)
    if m:
        first = m.group(0).strip().splitlines()[0]
        result['purpose'] = first[:200]
    # imports
    result['imports'] = re.findall(r"^\s*import\s+([A-Za-z0-9_]+)", text, re.M)
    # exports: public types/functions
    pub_types = re.findall(r"^\s*public\s+(class|struct|enum)\s+([A-Za-z0-9_]+)", text, re.M)
    pub_funcs = re.findall(r"^\s*public\s+func\s+([A-Za-z0-9_]+)\s*\(", text, re.M)
    result['exports'] = [n for _, n in pub_types] + pub_funcs
    # complexity via keyword counts
    complexity = 0
    for kw in [' if ', ' for ', ' while ', ' guard ', ' switch ', ' case ']:
        complexity += text.count(kw)
    result['complexity'] = complexity
    # smells
    if text.count(' class ') + text.count(' struct ') > 10 and len(text.splitlines()) > 800:
        result['smells'].append('god_file_suspect')
    # prints
    result['print_statements'] = len(re.findall(r"\bprint\s*\(", text))
    return result

source_analysis = {}
module_graph = defaultdict(set)  # python import graph by module file rel
symbols_defined = defaultdict(set)
symbols_referenced = defaultdict(set)

# collect tests
python_tests = []
for f in files:
    if f['ext'] == '.py' and ('tests' in f['rel'] or os.path.basename(f['rel']).startswith('test_')):
        python_tests.append(f['path'])

# Analyze
for f in files:
    p = f['path']
    if f['ext'] == '.py':
        txt = read_text(p)
        result = analyze_python(p, txt)
        source_analysis[p] = result
        # module graph
        try:
            tree = ast.parse(txt or '')
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_graph[p].add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_graph[p].add(node.module.split('.')[0])
            # symbols
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    symbols_defined[p].add(node.name)
                elif isinstance(node, ast.ClassDef):
                    symbols_defined[p].add(node.name)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    symbols_referenced[p].add(node.id)
        except Exception:
            pass
    elif f['ext'] == '.swift':
        txt = read_text(p)
        result = analyze_swift(p, txt)
        source_analysis[p] = result

# Test coverage mapping (Python heuristic)
function_tested = defaultdict(bool)
for tpath in python_tests:
    ttxt = read_text(tpath) or ''
    called = set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", ttxt))
    for p, defs in symbols_defined.items():
        for fn in defs:
            if fn in called:
                function_tested[(p, fn)] = True

# Exact duplicates
hash_to_files = defaultdict(list)
for f in files:
    h = file_hash(f['path'])
    if h:
        hash_to_files[h].append(f['path'])
exact_duplicates = [v for v in hash_to_files.values() if len(v) > 1]

# Near duplicates (source only, cheap Jaccard)
near_dups = []
source_paths = [p for p in source_analysis.keys()]
shingles = {}
for p in source_paths:
    txt = read_text(p)
    tokens = tokenize_code(txt)
    shingles[p] = set(tokens)
for i in range(len(source_paths)):
    for j in range(i+1, len(source_paths)):
        a,b = source_paths[i], source_paths[j]
        sim = jaccard(shingles[a], shingles[b])
        if sim >= 0.8:
            near_dups.append({'files':[a,b],'similarity':round(sim*100,2)})

# Dead code: definitions never referenced and not called in own module
dead_code = []
for p, defs in symbols_defined.items():
    refs_elsewhere = set()
    for q, refs in symbols_referenced.items():
        if q != p:
            refs_elsewhere |= refs
    for sym in defs:
        if sym not in symbols_referenced[p] and sym not in refs_elsewhere:
            dead_code.append({'path': p, 'symbol': sym})

# Redundant abstractions: single-call wrappers
redundant_wrappers = []
for p in source_analysis:
    if not p.endswith('.py'):
        continue
    txt = read_text(p) or ''
    try:
        tree = ast.parse(txt)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
                if len(calls) == 1 and len(list(ast.iter_child_nodes(node))) < 5:
                    redundant_wrappers.append({'path': p, 'function': node.name})
    except Exception:
        pass

# Secrets detection
secret_patterns = [
    (r"AKIA[0-9A-Z]{16}", 'aws_access_key'),
    (r"(?i)secret[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9/_\-]{16,}['\"]?", 'generic_secret'),
    (r"(?i)api[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9/_\-]{16,}['\"]?", 'api_key'),
    (r"(?i)password\s*[:=]\s*['\"][^'\"]{6,}['\"]", 'password_literal'),
]
secrets = []
for f in files:
    if f['is_text'] and f['size'] < 2_000_000:
        txt = read_text(f['path']) or ''
        for pat, kind in secret_patterns:
            for m in re.finditer(pat, txt):
                lineno = txt.count('\n', 0, m.start()) + 1
                secrets.append({'path': f['path'], 'line': lineno, 'kind': kind, 'excerpt': txt[m.start():m.start()+50]})

# TODO/FIXME/HACK age (by file mtime heuristic)
annotations = []
for f in files:
    if f['is_text'] and f['size'] < 2_000_000:
        txt = read_text(f['path']) or ''
        if re.search(r"\b(TODO|FIXME|HACK)\b", txt):
            age_days = (NOW - f['mtime'])/86400.0
            annotations.append({'path': f['path'], 'age_days': round(age_days,1)})

# Console prints in production (Python/Swift) excluding tests/scripts
prod_prints = []
for p, analysis in source_analysis.items():
    if ('tests' in p) or os.path.basename(p).startswith('test_'):
        continue
    if analysis.get('print_statements'):
        prod_prints.append({'path': p, 'count': analysis['print_statements']})

# Circular dependencies (python by top-level module name)
# Map file to module (top-level dir or package)
py_modules = {}
for f in files:
    if f['ext'] == '.py':
        rel = f['rel']
        parts = rel.split(os.sep)
        if len(parts) > 1:
            py_modules[rel] = parts[0]
        else:
            py_modules[rel] = ''

graph = defaultdict(set)
for p, deps in module_graph.items():
    a = py_modules.get(os.path.relpath(p, ROOT), '')
    for d in deps:
        graph[a].add(d)

# detect cycles via DFS
cycles = []
visited = set()
stack = []

def dfs(n):
    if n in stack:
        i = stack.index(n)
        cyc = stack[i:] + [n]
        cycles.append(cyc)
        return
    if n in visited:
        return
    visited.add(n)
    stack.append(n)
    for m in graph.get(n, []):
        dfs(m)
    stack.pop()

for n in list(graph.keys()):
    dfs(n)

# Layering heuristics for Python by directory tiers
layering_violations = []
# Configure simple layers: src/* should not import tests/*, UI should not import backend, etc.
for p, deps in module_graph.items():
    rel = os.path.relpath(p, ROOT)
    if rel.startswith('autoresolve/src'):
        for dep in deps:
            if 'tests' in dep:
                layering_violations.append({'path': p, 'violates': 'src->tests'})

# Build per-file summary for PHASE 2 output
file_summaries = []
for f in files:
    if f['ext'] not in ('.py','.swift'):
        continue
    p = f['path']
    a = source_analysis.get(p, {})
    imports = a.get('imports', [])
    unused = a.get('unused_imports', [])
    exports = a.get('exports', [])
    # critical functions: use exports
    crit = []
    if a.get('critical_functions'):
        for name, meta in a['critical_functions'].items():
            if name in exports or f['ext'] == '.py':
                crit.append({'name': name, 'what': meta.get('what','')})
    # tech debt score
    debt = 'Low'
    reasons = []
    comp = a.get('complexity', 0)
    if comp >= 30 or any('long_function' in s for s in a.get('smells', [])):
        debt = 'High'
    elif comp >= 12 or a.get('smells'):
        debt = 'Medium'
    file_summaries.append({
        'file': p,
        'purpose': a.get('purpose'),
        'imports_total': len(imports),
        'unused_imports': unused,
        'critical_functions': crit,
        'complexity': comp,
        'smells': a.get('smells', []),
        'tech_debt': debt,
        'exports': exports,
    })

# Orphaned files heuristic: not imported by any python file and not executed scripts/tests
imported_rel_paths = set()
# Rough: gather mentions of file basenames in imports
for p, deps in module_graph.items():
    for d in deps:
        imported_rel_paths.add(d)

orphaned = []
for f in files:
    if f['ext'] == '.py':
        rel = os.path.relpath(f['path'], ROOT)
        mod = os.path.splitext(os.path.basename(rel))[0]
        if mod == '__init__':
            continue
        if mod not in imported_rel_paths and not ('tests' in rel or os.path.basename(rel).startswith('test_')):
            orphaned.append(f['path'])

# Final report
report = {
    'generated_at': datetime.utcnow().isoformat() + 'Z',
    'root': ROOT,
    'fs': {
        'files': files,
        'empty_dirs': empty_dirs,
        'deep_dirs': deep_dirs,
        'wrong_loc_media': wrong_loc_media,
        'nonstandard_names': nonstandard_names,
        'classification': {
            k: {ext: v for ext, v in d.items()} for k, d in classification.items()
        },
    },
    'code': {
        'per_file': file_summaries,
        'dead_code': dead_code,
        'redundant_wrappers': redundant_wrappers,
        'exact_duplicates': exact_duplicates,
        'near_duplicates': near_dups,
        'prod_prints': prod_prints,
        'secrets': secrets,
        'annotations': annotations,
        'cycles': cycles,
        'layering_violations': layering_violations,
        'function_tested': list(map(lambda kv: {'path': kv[0], 'function': kv[1]}, [k for k,v in function_tested.items() if v]))
    }
}

OUT = os.path.join(ROOT, 'artifacts', 'autopsy_report.json')
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT,'w') as f:
    json.dump(report, f, indent=2)
print(OUT)
