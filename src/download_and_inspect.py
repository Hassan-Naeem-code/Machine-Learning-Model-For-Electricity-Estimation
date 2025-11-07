#!/usr/bin/env python3
"""
Download dataset index page and extract file links for inspection.
"""
import os
import re
import requests
from bs4 import BeautifulSoup

URL = "http://frankfurt-school-2024-autumn-data.s3-website.eu-central-1.amazonaws.com/"
OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTDIR = os.path.abspath(OUTDIR)

os.makedirs(OUTDIR, exist_ok=True)

def download_index():
    r = requests.get(URL, timeout=30)
    r.raise_for_status()
    path = os.path.join(OUTDIR, 'index.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(r.text)
    print('Saved', path)
    return path


def extract_links(index_path):
    with open(index_path, 'r', encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    for a in soup.find_all('a'):
        href = a.get('href')
        if href and href.endswith('.parquet'):
            links.append(href)
    print('Found', len(links), 'parquet links (sample 10):')
    for L in links[:10]:
        print('-', L)
    return links


if __name__ == '__main__':
    idx = download_index()
    extract_links(idx)
