#!/usr/bin/env python3
"""Fetch arXiv metadata and print BibTeX for selected IDs."""
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

IDS = [
    "2310.08560",  # MemGPT
    "2405.14831",  # HippoRAG
    "2502.14802",  # From RAG to Memory
    "2310.11511",  # Self-RAG
    "1907.05242",  # Large Memory Layers with Product Keys
    "2109.04212",  # Efficient kNN-LM
    "2301.02828",  # Why do kNN-LM Work?
    "2205.12674",  # Training LMs with Memory Augmentation
    "2210.16433",  # Knowledge-in-Context
    "2311.08590",  # PEMA
    "2307.03172",  # Lost in the Middle
    "2305.17691",  # Plug-and-Play Knowledge Injection
    "2402.13904",  # Calibrating LLMs with Sample Consistency
    "2410.22954",  # RAG with Estimation of Source Reliability
    "2412.21199",  # HumanEval Pro
    "2411.13504",  # Disentangling Memory and Reasoning
    "2503.07903",  # Memory-Augmented LMs Reasoning-in-a-Haystack
    "2407.01437",  # Needle in the Haystack for Memory Based LLMs
    "2609.03426",  # Lngram v2
    "2606.08347",  # (shortlisted recent; may exist)
    "2510.22344",  # FAIR-RAG
    "2604.20854",  # ERA
    "2608.25487",  # ReliableRAG
    "2603.19935",  # Memori
    "2509.18713",  # MemOrb
    "2501.13956",  # Zep
]

url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
    {"id_list": ",".join(IDS), "max_results": 100}
)
data = urllib.request.urlopen(url, timeout=90).read()
root = ET.fromstring(data)
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

for e in root.findall("a:entry", NS):
    title = " ".join(e.find("a:title", NS).text.split())
    authors = [a.find("a:name", NS).text for a in e.findall("a:author", NS)]
    year = e.find("a:published", NS).text[:4]
    idurl = e.find("a:id", NS).text.split("/abs/")[-1]
    key = "arxiv" + idurl.split("v")[0].replace(".", "")
    print("@article{" + key + ",")
    print("  title={" + title + "},")
    print("  author={" + " and ".join(authors) + "},")
    print("  year={" + year + "},")
    print("  url={https://arxiv.org/abs/" + idurl + "}")
    print("}")
