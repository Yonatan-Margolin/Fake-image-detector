from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path
from zipfile import ZipFile, is_zipfile

DATASET = 'reubensuju/celeb-df-v2'
ROOT    = Path(r'F:\datasets\CelebDFv2')   # change if needed
ROOT.mkdir(parents=True, exist_ok=True)

api = KaggleApi(); api.authenticate()

# get all files once (no pagination in the API)
resp  = api.dataset_list_files(DATASET)
files = [f.name for f in resp.files if f.name.endswith('.mp4')]
print(f'Total .mp4 files: {len(files)}')

# (optional) grab a small subset first to test:
# files = files[:50]

for i, name in enumerate(files, 1):
    dest_dir = ROOT / Path(name).parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / Path(name).name
    if target.exists():
        print(f'[{i}/{len(files)}] skip {name} (exists)')
        continue
    print(f'[{i}/{len(files)}] downloading {name}')
    zpath = Path(api.dataset_download_file(DATASET, name, path=str(dest_dir), force=False, quiet=False))
    if zpath.exists() and is_zipfile(zpath):
        with ZipFile(zpath) as z: z.extractall(dest_dir)
        try: zpath.unlink()
        except: pass

print('Done.')
