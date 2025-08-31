#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
School directory email finder — consolidated rewrite

What’s new (rolled up):
1) Directory discovery checks *all links* on the base site for staff/directory/HS hints before trying common paths.
2) Requests allow up to 30s per fetch with retries/backoff.
3) Incremental CSV writing + resume: each row is appended and flushed immediately; reruns skip already-written rows.
4) Principal search inside a directory follows: (a) last name → (b) first name → (c) title 'principal' (excluding 'assistant principal').
"""

from __future__ import annotations
import re
import os
import csv
import time
import random
import base64
import argparse
from dataclasses import dataclass
from typing import Optional, Iterable, Tuple, List, Dict

import requests
from bs4 import BeautifulSoup, Tag
from urllib.parse import urljoin, urlparse, parse_qs, unquote, quote, urlencode, urlunparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

# -----------------------------
# Utilities
# -----------------------------

def norm_space(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def split_name(raw: str) -> Tuple[str, str]:
    """Return (first, last) best‑effort from an Administrator field.
    Strips titles and suffixes. If only one token, last=that token.
    """
    s = norm_space(raw)
    s = re.sub(r"\b(Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Principal|Assistant\s+Principal|Head\s+of\s+School|Dean)\b\.?:?\s*", "", s, flags=re.I)
    s = re.sub(r",?\s*(Jr\.?|Sr\.?|III|II|IV|PhD|EdD|MEd|MBA|MA|MS)$", "", s, flags=re.I)
    parts = [p for p in s.split(" ") if p]
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], parts[0]
    return parts[0], parts[-1]


def looks_like_directory_link(a: Tag) -> bool:
    txt = norm_space(a.get_text(" ")).lower()
    href = (a.get("href") or "").lower()
    needles = [
        "directory", "staff", "faculty", "contacts", "contact-us", "administration",
        "our-staff", "staff-directory", "faculty-staff", "employee-directory",
    ]
    return any(n in txt or n in href for n in needles)


def looks_like_hs_link(a: Tag) -> bool:
    txt = norm_space(a.get_text(" ")).lower()
    href = (a.get("href") or "").lower()
    hs_needles = [
        "/high-school", "/high_school", "/highschool", "/hs/", "/hs", " high school", "highschool",
        "/schools/high-", "/secondary", " senior high",
    ]
    return any(n in href or n in txt for n in hs_needles)


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(UA)
    retries = Retry(total=4, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"]) 
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s


def fetch(session: requests.Session, url: str, timeout: float = 90.0) -> Optional[str]:
    print(f"    [FETCH] GET {url}")
    try:
        r = session.get(url, timeout=timeout)
        if 200 <= r.status_code < 300:
            print(f"    [SUCCESS] {r.status_code} {url}")
            return r.text
        else:
            print(f"    [FAIL] non-2xx {r.status_code} {url}")
    except requests.RequestException as e:
        print(f"    [FAIL] Request failed for {url}: {e}")
        return None
    return None


def extract_all_links(html: str, base_url: str) -> List[Tag]:
    """Return a list of <a> tags with absolute href in a['__abs_href']."""
    soup = BeautifulSoup(html, "html.parser")
    anchors: List[Tag] = []
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        try:
            abs_url = urljoin(base_url, href)
            a.attrs["__abs_href"] = abs_url
            anchors.append(a)
        except Exception:
            continue
    return anchors

# -----------------------------
# Email extraction helpers
# -----------------------------

def decode_cf_email(hex_str: str) -> str:
    r = int(hex_str[:2], 16)
    out = []
    for i in range(2, len(hex_str), 2):
        out.append(chr(int(hex_str[i:i+2], 16) ^ r))
    return "".join(out)


def _b64_try_all(s: str) -> Optional[str]:
    cand = s
    for fn in (base64.b64decode, base64.urlsafe_b64decode):
        t = cand
        pad = len(t) % 4
        if pad:
            t += "=" * (4 - pad)
        try:
            return fn(t).decode("utf-8")
        except Exception:
            continue
    return None


def collect_emails_from_soup(soup: BeautifulSoup) -> List[str]:
    emails = set()

    for a in soup.select('a[href^="mailto:"]'):
        href = a.get("href", "")
        emails.add(href.replace("mailto:", "").strip())

    for a in soup.select('a[href*="/cdn-cgi/l/email-protection#"]'):
        href = a.get("href", "")
        if "#" in href:
            hexpart = href.split("#", 1)[-1]
            if re.fullmatch(r"[0-9a-fA-F]+", hexpart or ""):
                try:
                    dec = decode_cf_email(hexpart)
                    if dec: emails.add(dec)
                except Exception:
                    pass

    for sp in soup.select('span.__cf_email__'):
        hexpart = sp.get("data-cfemail", "")
        if hexpart and re.fullmatch(r"[0-9a-fA-F]+", hexpart or ""):
            try:
                dec = decode_cf_email(hexpart)
                if dec: emails.add(dec)
            except Exception:
                pass

    for a in soup.select('a[href*="sendMail.cfm"]'):
        href = a.get("href", "")
        qs = parse_qs(urlparse(href).query)
        if "e" in qs and qs["e"]:
            dec = _b64_try_all(unquote(qs["e"][0]))
            if dec: emails.add(dec)
        onclick = a.get("onclick", "")
        m = re.search(r"sendEmail\('([^']+)'\)", onclick or "")
        if m:
            dec = _b64_try_all(m.group(1))
            if dec: emails.add(dec)

    text = soup.get_text(" ")
    for m in re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text):
        emails.add(m)

    cleaned = set()
    for e in emails:
        e2 = e.strip().strip(".;,:").lower()
        if "@" in e2 and not e2.startswith("//"):
            cleaned.add(e2)
    return sorted(cleaned)

# -----------------------------
# Form provider detection
# -----------------------------
PROVIDERS = {
    "Contact Form 7":      {"css": [".wpcf7-form", ".wpcf7"], "re": []},
    "WPForms":             {"css": [".wpforms-form", ".wpforms-container"], "re": [r"wpforms"]},
    "Gravity Forms":       {"css": [".gform_wrapper", "form[id^='gform_']"], "re": [r"gravityforms"]},
    "Ninja Forms":         {"css": [".nf-form-cont", ".ninja-forms-form"], "re": [r"ninja-forms"]},
    "Formidable Forms":    {"css": [".frm_fields_container", ".frm_forms"], "re": [r"formidableforms"]},
    "Caldera Forms":       {"css": [".caldera-grid", ".caldera-forms-form"], "re": [r"caldera-forms"]},
    "Drupal Webform":      {"css": [".webform-submission-form", "form.webform-client-form"], "re": [r"\bwebform\b"]},
    "Joomla RSForm":       {"css": [".rsform", "form[id^='userForm']"], "re": [r"com_rsform"]},
    "Google Forms":        {"css": ["iframe[src*='docs.google.com/forms']"], "re": [r"docs\\.google\\.com/forms"]},
    "Microsoft Forms":     {"css": ["iframe[src*='forms.office.com']"], "re": [r"forms\\.office\\.com"]},
    "Typeform":            {"css": ["iframe[src*='typeform.com']", "div[data-tf-widget]"], "re": [r"typeform\\.com"]},
    "Jotform":             {"css": ["script[src*='jotform']", "iframe[src*='jotform']"], "re": [r"jotform"]},
    "Wufoo":               {"css": ["form[id^='wufoo']", "script[src*='wufoo.com']"], "re": [r"wufoo\\.com"]},
    "Formstack":           {"css": [".fsForm", "form[action*='formstack']"], "re": [r"formstack"]},
    "HubSpot Forms":       {"css": ["script[src*='js.hsforms.net']", "iframe[src*='hsforms']"], "re": [r"hsforms\\.net", r"hubspot"]},
    "ColdFusion SendMail": {"css": ["form[action*='sendMail.cfm']"], "re": [r"sendMail\\.cfm"]},
    "Generic contact.php": {"css": ["form[action*='contact.php']"], "re": [r"contact\\.php"]},
    "ASP.NET Contact":     {"css": ["form[action*='/Contact']", "form[action*='SendEmail']"], "re": [r"\\.asmx|/Contact/Send|/api/mail", r"__VIEWSTATE"]},
}


def detect_form_providers(html: str) -> List[Dict[str, List[str]]]:
    soup = BeautifulSoup(html, "html.parser")
    hits = []
    for name, rules in PROVIDERS.items():
        found_css = []
        for sel in rules["css"]:
            if soup.select_one(sel):
                found_css.append(sel)
        found_re = []
        if rules["re"]:
            blobs = []
            for tag in soup.find_all(["script", "link", "iframe"]):
                for attr in ("src", "href"):
                    if tag.has_attr(attr):
                        blobs.append(str(tag.get(attr)))
            blobs.append(html)
            big = "\n".join(blobs)
            for pat in rules["re"]:
                if re.search(pat, big, flags=re.I):
                    found_re.append(pat)
        if found_css or found_re:
            hits.append({"provider": name, "evidence_css": found_css, "evidence_regex": found_re})
    return hits

# -----------------------------
# Site discovery
# -----------------------------

def choose_hs_site(session: requests.Session, base_url: str) -> str:
    print(f"  [HS-SITE] Checking for a specific high school site from base: {base_url}")
    html = fetch(session, base_url)
    if not html:
        print(f"  [HS-SITE] Failed to fetch base URL, using base URL: {base_url}")
        return base_url
    anchors = extract_all_links(html, base_url)
    for a in anchors:
        try:
            if looks_like_hs_link(a):
                hs_url = a.attrs.get("__abs_href") or a.get("href")
                if hs_url:
                    print(f"  [HS-SITE] Found HS link: {hs_url}")
                    return hs_url
        except Exception:
            continue
    print(f"  [HS-SITE] No HS link found, using base URL: {base_url}")
    return base_url


def find_directory_url(session: requests.Session, site_url: str) -> Optional[str]:
    """Look for a directory/staff/faculty link. More permissive acceptance:
    - Scan ALL anchors on the homepage.
    - Try common paths; if the path itself contains staff/directory hints and returns 200, accept it even if the page text is JS-rendered/empty.
    - Otherwise require simple keyword hints in the text.
    """
    print(f"  [DIR] Searching for a directory page on: {site_url}")
    html = fetch(session, site_url)
    if not html:
        print("  [DIR] Could not fetch site URL to find directory.")
        return None

    # Step A: examine *all* links on the base page for directory candidates
    print("  [DIR] Step A: Scanning ALL links for directory-like targets...")
    anchors = extract_all_links(html, site_url)
    for a in anchors:
        if looks_like_directory_link(a):
            href = a.attrs.get("__abs_href") or a.get("href")
            if href:
                print(f"    [DIR] Found candidate via anchor scan: {href}")
                return href

    # Step B: try common paths relative to site_url (more permissive)
    print("  [DIR] Step B: Trying common directory paths...")
    common_paths = [
        "/staff", "/directory", "/staff-directory", "/faculty", "/faculty-staff", "/contacts", "/administration",
        "/about/staff", "/about/directory", "/our-staff",
    ]
    url_hints = ("staff", "directory", "faculty", "contacts", "administration")
    fallback_200: Optional[str] = None
    for p in common_paths:
        test_url = urljoin(site_url, p)
        print(f"    [DIR] Trying path: {test_url}")
        h = fetch(session, test_url)
        if not h:
            continue
        # If URL itself contains obvious hint and returned 200, accept immediately
        if any(k in test_url.lower() for k in url_hints):
            print(f"  [SUCCESS] Accepting directory by URL hint + 200: {test_url}")
            return test_url
        # Otherwise, look for light text signals (handles non-JS pages)
        s2 = BeautifulSoup(h, "html.parser")
        txt = s2.get_text(" ").lower()
        if re.search(r"directory|staff|faculty|administration|teacher|employee", txt):
            print(f"  [SUCCESS] Found directory by page text: {test_url}")
            return test_url
        # keep the first 200 as a fallback in case we found nothing else
        if not fallback_200:
            fallback_200 = test_url

    if fallback_200:
        print(f"  [FALLBACK] Using first 200 OK path as directory: {fallback_200}")
        return fallback_200

    print("  [DIR] No directory found.")
    return None

# -----------------------------
# Site search (name in site search bar)
# -----------------------------

SEARCH_PARAM_KEYS = [
    "q","s","keys","key","keyword","search","searchword","query","term","k","searchtext","SearchText","text"
]


def discover_search_endpoints(session: requests.Session, site_url: str) -> List[str]:
    """Find likely on-site search endpoints from the homepage.
    Looks for links with 'search' and forms that look like search.
    Also adds a few common guesses.
    """
    html = fetch(session, site_url)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")

    endpoints: set[str] = set()

    # 1) Links that look like search pages
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        if "search" in href.lower():
            endpoints.add(urljoin(site_url, href))

    # 2) Forms that look like search
    for form in soup.find_all("form"):
        action = form.get("action") or site_url
        action_abs = urljoin(site_url, action)
        looks_search = False
        for inp in form.find_all("input"):
            n = (inp.get("name") or "").lower()
            t = (inp.get("type") or "").lower()
            ident = (inp.get("id") or "").lower()
            if t in ("search", "text") and ("search" in ident or n in [k.lower() for k in SEARCH_PARAM_KEYS]):
                looks_search = True
        if looks_search:
            endpoints.add(action_abs)

    # 3) Common guesses (CMSes vary)
    for p in [
        "/search", "/Search", "/site-search", "/Site-Search", "/search/site", "/search/results",
        "/District/Search/",  # Finalsite-style
    ]:
        endpoints.add(urljoin(site_url, p))

    return list(endpoints)


def build_search_urls_for_endpoint(endpoint: str, query: str) -> List[str]:
    """Generate concrete URLs to try for a given endpoint and query.
    Handles querystring variants and a path-encoded fallback (/Search/<base64(urlsafe) payload>)."""
    urls: List[str] = []
    low = endpoint.lower()

    # Finalsite-like path (/District/Search/<b64>) fallback
    if "/search" in low and "." not in low and "?" not in endpoint:
        payload = f"t={query}"
        b64 = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")
        urls.append(endpoint.rstrip("/") + "/" + b64)

    # If endpoint already has a querystring
    if "?" in endpoint:
        parsed = urlparse(endpoint)
        qs = parse_qs(parsed.query)
        # If it already contains a known search key, replace it
        found_key = None
        for k in SEARCH_PARAM_KEYS:
            if k in qs:
                found_key = k
                break
        if found_key:
            qs[found_key] = [query]
            new_qs = urlencode({k: v[0] if isinstance(v, list) else v for k, v in qs.items()})
            urls.append(urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_qs, parsed.fragment)))
        else:
            sep = "&" if parsed.query else "?"
            for k in ["q", "s", "keys", "search", "searchword", "query"]:
                urls.append(endpoint + f"{sep}{k}={quote(query)}")
    else:
        # No querystring; try common param names
        for k in ["q", "s", "keys", "search", "searchword", "query", "term"]:
            urls.append(endpoint + f"?{k}={quote(query)}")

    return urls


def search_site_for_person(session: requests.Session, site_url: str, first: str, last: str) -> List[str]:
    """Use the site's own search to look for the principal by name.
    Tries (1) last name, then (2) first+last, then (3) first name.
    Returns up to a few candidate profile/result URLs.
    """
    name_attempts = []
    if last:
        name_attempts.append(last)
    full = norm_space(f"{first} {last}")
    if full:
        name_attempts.append(full)
    if first:
        name_attempts.append(first)

    endpoints = discover_search_endpoints(session, site_url)
    tried: set[str] = set()

    for q in name_attempts:
        print(f"  [SEARCH] Searching site for '{q}' ...")
        probes = endpoints[:]
        # If no endpoints discovered, also try some root guesses directly
        if not probes:
            probes = [
                urljoin(site_url, "/search"), urljoin(site_url, "/Search"), urljoin(site_url, "/site-search"),
                urljoin(site_url, "/Site-Search"), urljoin(site_url, "/search/site"), site_url,
            ]
        for ep in probes:
            for url in build_search_urls_for_endpoint(ep, q):
                if url in tried:
                    continue
                tried.add(url)
                html = fetch(session, url)
                if not html:
                    continue
                soup = BeautifulSoup(html, "html.parser")
                tokens = [t for t in q.lower().split() if t]
                cands: List[str] = []
                for a in soup.find_all("a"):
                    href = a.get("href")
                    if not href:
                        continue
                    text = norm_space(a.get_text(" ")).lower()
                    href_low = href.lower()
                    if all(t in text or t in href_low for t in tokens):
                        absu = urljoin(site_url, href)
                        # prefer same-domain results
                        if urlparse(absu).netloc == urlparse(site_url).netloc:
                            cands.append(absu)
                if cands:
                    uniq = []
                    seen = set()
                    for c in cands:
                        if c not in seen:
                            uniq.append(c)
                            seen.add(c)
                    print(f"  [SEARCH] Found {len(uniq)} candidate link(s) for '{q}'.")
                    return uniq[:6]
    print("  [SEARCH] No results from site search.")
    return []

# -----------------------------
# Person finding on directory pages
# -----------------------------

def match_person_nodes_strategy(soup: BeautifulSoup, first: str, last: str) -> List[Tag]:
    """Search order: (a) last name → (b) first name → (c) title 'principal' (exclude 'assistant principal' heuristically later).
    Returns up to 6 blocks.
    """
    def climb(tag: Tag, limit: int = 3, max_chars: int = 5000) -> Tag:
        cur = tag
        for _ in range(limit):
            if cur and cur.parent and len(norm_space(cur.get_text(" "))) < max_chars:
                cur = cur.parent
        return cur or tag

    orders: List[List[str]] = []
    if last:
        orders.append([last.lower()])
    if first:
        orders.append([first.lower()])
    orders.append(["principal"])  # fallback

    for terms in orders:
        hits: List[Tag] = []
        for tag in soup.find_all(True):
            txt = norm_space(tag.get_text(" ")).lower()
            if not txt:
                continue
            if all(t in txt for t in terms):
                # skip obvious assistant principal when searching title fallback
                if terms == ["principal"] and "assistant principal" in txt:
                    continue
                hits.append(climb(tag))
        if hits:
            out, seen = [], set()
            for h in hits:
                key = id(h)
                if key not in seen:
                    out.append(h)
                    seen.add(key)
            return out[:6]
    return []


def verify_email_for_name(email: str, first: str, last: str) -> bool:
    e = email.lower()
    last_ok = last and last.lower() in e
    first_ok = first and first.lower() in e
    return bool(last_ok or first_ok)


def find_assistant_blocks(soup: BeautifulSoup) -> List[Tag]:
    blocks: List[Tag] = []
    for tag in soup.find_all(True):
        txt = norm_space(tag.get_text(" ")).lower()
        if "assistant principal" in txt:
            cur = tag
            for _ in range(3):
                if cur and cur.parent and len(norm_space(cur.get_text(" "))) < 6000:
                    cur = cur.parent
            blocks.append(cur or tag)
    out, seen = [], set()
    for b in blocks:
        key = id(b)
        if key not in seen:
            out.append(b)
            seen.add(key)
    return out[:10]


def pull_emails_near_block(block: Tag) -> List[str]:
    soup = BeautifulSoup(str(block), "html.parser")
    return collect_emails_from_soup(soup)


def pull_names_near_block(block: Tag) -> List[str]:
    text = norm_space(block.get_text(" "))
    names = set()
    for m in re.finditer(r"Assistant Principal[:\-\s]*([A-Z][a-z]+\s+[A-Z][a-z'\-]+)", text):
        names.add(m.group(1))
    for m in re.finditer(r"\b([A-Z][a-z]+\s+[A-Z][a-z'\-]+)\b", text):
        names.add(m.group(1))
    return sorted(names)

# -----------------------------
# Per-school processing
# -----------------------------

@dataclass
class PersonResult:
    role: str  # "Principal" or "Assistant Principal"
    name: str
    email: Optional[str]
    contact_form: int  # 0 or 1


def search_site_for_name(session: requests.Session, site_url: str, first: str, last: str) -> Optional[str]:
    """Search the site for the principal's name and return a profile link if found."""
    query = "+".join([t for t in [first, last] if t])
    if not query:
        return None
    search_url = urljoin(site_url, f"/District/Search/dD1{query}")
    print(f"  [SEARCH] Trying search URL: {search_url}")
    html = fetch(session, search_url)
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a"):
        txt = norm_space(a.get_text(" ")).lower()
        if last.lower() in txt or first.lower() in txt:
            href = a.get("href")
            if href:
                return urljoin(site_url, href)
    return None


def process_school(session: requests.Session, base_website: str, admin_name: str, sleep_range=(0.8, 1.8)) -> Tuple[Optional[PersonResult], List[PersonResult]]:
    print(f" [PROCESS] {base_website} | Admin: {admin_name}")
    first, last = split_name(admin_name)
    if not base_website:
        print("  [SKIP] No website")
        return None, []

    site = choose_hs_site(session, base_website)
    dir_url = find_directory_url(session, site)

    principal_email: Optional[str] = None
    principal_form = 0

    if dir_url:
        html = fetch(session, dir_url)
        if html:
            soup = BeautifulSoup(html, "html.parser")
            blocks = match_person_nodes_strategy(soup, first, last)
            for blk in blocks:
                emails = pull_emails_near_block(blk)
                valid = [e for e in emails if verify_email_for_name(e, first, last)]
                if valid:
                    principal_email = valid[0]
                    principal_form = 1 if detect_form_providers(str(blk)) else 0
                    break
                for a in blk.find_all("a"):
                    href = a.get("href")
                    if href:
                        page = urljoin(dir_url, href)
                        pe = extract_emails_from_page(session, page)
                        valid2 = [e for e in pe.emails if verify_email_for_name(e, first, last)]
                        if valid2:
                            principal_email = valid2[0]
                            principal_form = 1 if pe.form_hits else 0
                            break
                if principal_email:
                    break
    else:
        # No directory; try search fallback
        profile = search_site_for_name(session, site, first, last)
        if profile:
            pe = extract_emails_from_page(session, profile)
            valid = [e for e in pe.emails if verify_email_for_name(e, first, last)]
            if valid:
                principal_email = valid[0]
                principal_form = 1 if pe.form_hits else 0

    # Fallback: if no principal email yet, try site search on the main site
    if not principal_email:
        print("  [PRINCIPAL] Falling back to site search...")
        links = search_site_for_person(session, site, first, last)
        for page in links:
            pe = extract_emails_from_page(session, page)
            valid = [e for e in pe.emails if verify_email_for_name(e, first, last)]
            if valid:
                principal_email = valid[0]
                principal_form = 1 if pe.form_hits else 0
                break

    principal_res: Optional[PersonResult] = None
    if principal_email:
        principal_res = PersonResult(role="Principal", name=f"{first} {last}".strip(), email=principal_email, contact_form=principal_form)

    assistants: List[PersonResult] = []
    if dir_url:
        html = fetch(session, dir_url)
        if html:
            soup = BeautifulSoup(html, "html.parser")
            ablocks = find_assistant_blocks(soup)
            for blk in ablocks:
                ems = pull_emails_near_block(blk)
                names = pull_names_near_block(blk)
                if not ems:
                    for a in blk.find_all("a"):
                        href = a.get("href")
                        if href:
                            page = urljoin(dir_url, href)
                            pe = extract_emails_from_page(session, page)
                            if pe.emails:
                                ems.extend(pe.emails)
                ems = sorted(set(ems))
                if not names:
                    names = ["" ] * len(ems)
                for i, e in enumerate(ems):
                    if "@" not in e:
                        continue
                    nm = names[i] if i < len(names) else names[-1] if names else ""
                    cf = 1 if detect_form_providers(str(blk)) else 0
                    assistants.append(PersonResult(role="Assistant Principal", name=nm, email=e, contact_form=cf))

    time.sleep(random.uniform(*sleep_range))
    return principal_res, assistants

# -----------------------------
# CSV helpers (incremental write + resume)
# -----------------------------

def row_key(row: dict, website_col: str, admin_col: str) -> tuple:
    return (
        norm_space(row.get(website_col, "")).lower(),
        norm_space(row.get(admin_col, "")).lower(),
    )


def load_existing_keys(path: str, website_col: str, admin_col: str) -> set:
    keys = set()
    if not os.path.exists(path):
        return keys
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                keys.add(row_key(r, website_col, admin_col))
    except Exception:
        pass
    return keys


def ensure_writer(path: str, fieldnames: List[str]):
    file_exists = os.path.exists(path)
    need_header = True
    if file_exists:
        try:
            need_header = os.path.getsize(path) == 0
        except OSError:
            need_header = True
    fh = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    if need_header:
        writer.writeheader()
        fh.flush()
    return fh, writer

# -----------------------------
# main
# -----------------------------

def find_col(cols: List[str], wanted: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for w in wanted:
        if w.lower() in low:
            return low[w.lower()]
    for c in cols:
        if any(w.lower() in c.lower() for w in wanted):
            return c
    return None


def read_input_rows(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: (v or "") for k, v in r.items()})
    return rows


def main(input_csv: str, output_csv: str, sleep_lo: float = 0.8, sleep_hi: float = 1.8) -> None:
    print(f"[MAIN] Input: '{input_csv}' → Output: '{output_csv}'")
    session = make_session()
    base_rows = read_input_rows(input_csv)
    if not base_rows:
        raise SystemExit("No input rows.")

    cols = list(base_rows[0].keys())
    website_col = find_col(cols, ["website", "school_website", "url"]) 
    admin_col   = find_col(cols, ["Administrator", "Principal", "administrator", "principal_name"]) 
    if not website_col or not admin_col:
        raise SystemExit("Input must have 'website' and 'Administrator' (or similar) columns.")

    fieldnames = list(cols)
    for extra in ["email", "contact_form"]:
        if extra not in fieldnames:
            fieldnames.append(extra)

    processed = load_existing_keys(output_csv, website_col, admin_col)
    if processed:
        print(f"[RESUME] Loaded {len(processed)} existing rows; duplicates will be skipped.")

    fh, writer = ensure_writer(output_csv, fieldnames)
    try:
        total = len(base_rows)
        for i, r in enumerate(base_rows):
            print("-" * 60)
            print(f"[MAIN] Row {i+1}/{total}")

            base_key = row_key(r, website_col, admin_col)
            if base_key in processed:
                print(f"[RESUME] Skip already written principal row for key={base_key}")
                continue

            website = norm_space(r.get(website_col, ""))
            admin   = norm_space(r.get(admin_col, ""))

            principal_res, assistant_res = process_school(
                session,
                website,
                admin,
                sleep_range=(sleep_lo, sleep_hi),
            )

            base_out = dict(r)
            base_out["email"] = principal_res.email if principal_res else ""
            base_out["contact_form"] = int(bool(principal_res.contact_form)) if principal_res else 0
            writer.writerow(base_out)
            fh.flush()
            processed.add(base_key)
            print("[MAIN] Wrote principal row.")

            for ap in assistant_res:
                ap_row = dict(r)
                ap_row[admin_col] = ap.name or "Assistant Principal"
                ap_row["email"] = ap.email or ""
                ap_row["contact_form"] = int(bool(ap.contact_form))
                ap_key = row_key(ap_row, website_col, admin_col)
                if ap_key in processed:
                    print(f"[RESUME] Skip already written assistant row for key={ap_key}")
                    continue
                writer.writerow(ap_row)
                fh.flush()
                processed.add(ap_key)
                print("[MAIN] Wrote assistant row.")

        print("-" * 60)
        print("[MAIN] Done.")
    finally:
        try:
            fh.close()
        except Exception:
            pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Find school/principal + assistant principal emails from staff directories.")
    ap.add_argument("--input-csv", "-i", default="/Users/kyungphillee/Desktop/Normal Lab/principal names/clean_input.csv", dest="input_csv", help="Path to input CSV")
    ap.add_argument("--output-csv", "-o", default="/Users/kyungphillee/Desktop/Normal Lab/principal names/clean_output.csv", dest="output_csv", help="Path to write the enriched CSV")
    ap.add_argument("--sleep-lo", type=float, default=0.8, dest="sleep_lo", help="Min sleep between schools")
    ap.add_argument("--sleep-hi", type=float, default=1.8, dest="sleep_hi", help="Max sleep between schools")
    args = ap.parse_args()

    main(args.input_csv, args.output_csv, sleep_lo=args.sleep_lo, sleep_hi=args.sleep_hi)
