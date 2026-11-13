import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path('${SERVER_HOME}/result/external_validation_asia_africa_v1')
ROOT.mkdir(parents=True, exist_ok=True)
meta = json.load(urllib.request.urlopen('https://api.figshare.com/v2/articles/28330067', timeout=90))
(ROOT / 'chetna_metadata.json').write_text(json.dumps(meta, indent=2))

def download(f):
    dest = ROOT / ('chetna_' + str(f['id']) + '.nc')
    if not dest.exists() or dest.stat().st_size != f['size']:
        urllib.request.urlretrieve(f['download_url'], dest)
    digest = hashlib.md5(dest.read_bytes()).hexdigest()
    expected = f.get('computed_md5')
    if expected and digest != expected:
        raise ValueError('Checksum mismatch: ' + str(dest))
    print(json.dumps({'id': f['id'], 'file': str(dest), 'size': dest.stat().st_size, 'md5': digest}), flush=True)

files = [f for f in meta['files'] if f['name'] == 'co2_emission.nc']
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(download, files))
