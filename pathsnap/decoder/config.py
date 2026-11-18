import json
import os
from pathlib import Path

def selected_settings():
    return json.loads(Path(os.getenv('PATHSNAP_DECODER','artifacts/decoder.json')).read_text())
