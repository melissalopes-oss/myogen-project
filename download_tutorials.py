import urllib.request
import zipfile
import os

url = "https://github.com/NsquaredLab/MyoGen/archive/refs/heads/main.zip"
zip_path = "myogen.zip"
urllib.request.urlretrieve(url, zip_path)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Find the main folder name, usually "MyoGen-main"
    tutorials_prefix = None
    for name in zip_ref.namelist():
        if "examples/" in name:
            tutorials_prefix = name.split("examples/")[0] + "examples/"
            break
        if "tutorials/" in name:
            tutorials_prefix = name.split("tutorials/")[0] + "tutorials/"
            break
            
    if tutorials_prefix:
        for name in zip_ref.namelist():
            if name.startswith(tutorials_prefix):
                # Extract to 'exemplos' folder
                rel_path = name[len(tutorials_prefix):]
                if not rel_path:
                    continue
                target_path = os.path.join("exemplos", rel_path)
                if name.endswith('/'):
                    os.makedirs(target_path, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    with zip_ref.open(name) as source, open(target_path, "wb") as target:
                        target.write(source.read())
        print("Tutorials extracted successfully to 'exemplos'")
    else:
        print("Tutorials folder not found in the zip")
        
os.remove(zip_path)
