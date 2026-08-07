import requests
import urllib3
from huggingface_hub import snapshot_download

# 1. Suppress the annoying insecure request warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 2. Save the original request method
original_request = requests.Session.request

# 3. Define a new request method that forces verify=False
def patched_request(self, method, url, **kwargs):
    kwargs['verify'] = False
    return original_request(self, method, url, **kwargs)

# 4. Apply the patch globally
requests.Session.request = patched_request

print("Bypassing firewall and resuming download...")

# 5. Run the exact same download command
snapshot_download(
    repo_id="jingchao-peng/HDRTDataset", 
    repo_type="dataset", 
    local_dir="data/HDRT"
)
