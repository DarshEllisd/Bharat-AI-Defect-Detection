"""Which IP family does Python (pip, hf, urllib) pick by default, and how fast is it?

On this connection IPv4 to Fastly/CloudFront crawls (~30 KB/s) while IPv6 is fast,
so Python must prefer IPv6 for the build's downloads to be quick.
"""
import socket
import time
import urllib.request

URLS = {
    "pypi (Fastly)": "https://files.pythonhosted.org/packages/source/n/numpy/numpy-1.26.4.tar.gz",
    "pytorch (CloudFront)": "https://download.pytorch.org/whl/cpu/torch-2.5.1%2Bcpu-cp312-cp312-linux_x86_64.whl",
    "huggingface": "https://huggingface.co/executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET/resolve/main/tokenizer.model",
}

for name, url in URLS.items():
    host = url.split("/")[2]
    first = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)[0]
    family = "IPv6" if first[0] == socket.AF_INET6 else "IPv4"
    start, got = time.time(), 0
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            while time.time() - start < 8:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                got += len(chunk)
    except Exception as e:
        print(f"{name:22s} first={family}  error: {e}")
        continue
    rate = got / max(time.time() - start, 1e-6)
    print(f"{name:22s} first={family}  {got / 1e6:6.1f} MB in {time.time() - start:4.1f}s  = {rate / 1e6:5.2f} MB/s")
