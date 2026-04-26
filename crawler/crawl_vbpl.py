#!/usr/bin/env python
"""
Crawler for https://vbpl.vn/.

The current VBPL site is a Next.js application. The detail pages expose their
document content after client-side hydration, so this crawler combines:

1. `requests` + `BeautifulSoup` to collect public document URLs from sitemap and
   optional browser index pages.
2. Selenium + headless Chrome to render each detail page and save a static HTML
   snapshot of the hydrated DOM for later dataset generation.

The crawler is restartable. It keeps a JSONL manifest in `data/raw_html/` and
skips URLs that were already saved successfully.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support.ui import WebDriverWait
from urllib3.util.retry import Retry

BASE_URL = "https://vbpl.vn"
DEFAULT_SCOPE = "trung-uong"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36 "
    "VBPLCrawler/1.0"
)
DOCUMENT_URL_RE = re.compile(r"^https://vbpl\.vn/van-ban/chi-tiet/.+$", re.IGNORECASE)
DOC_FILE_RE = re.compile(r"^doc_(\d{6})\.html$")
SNAPSHOT_STYLE = """
html { scroll-behavior: auto !important; }
body {
    background: #ffffff !important;
    color: #111827 !important;
}
img, iframe, video, canvas, svg {
    max-width: 100%;
}
table {
    max-width: 100%;
    border-collapse: collapse;
}
.table-scroll-wrapper {
    overflow: visible !important;
}
"""

MARK_CONTENT_SCRIPT = r"""
return (() => {
  const absolute = (value) => {
    try {
      return new URL(value, document.baseURI).href;
    } catch (error) {
      return value;
    }
  };

  document.querySelectorAll("[href]").forEach((node) => {
    const href = node.getAttribute("href");
    if (href) {
      node.setAttribute("href", absolute(href));
    }
  });

  document.querySelectorAll("[src]").forEach((node) => {
    const src = node.getAttribute("src");
    if (src) {
      node.setAttribute("src", absolute(src));
    }
  });

  const visibleTextLength = (element) => {
    const text = (element.innerText || "").trim();
    return text.length;
  };

  const candidates = Array.from(document.querySelectorAll("main, article, section, div"));
  const scored = candidates
    .map((element) => {
      const rect = element.getBoundingClientRect();
      const area = Math.max(rect.width, 0) * Math.max(rect.height, 0);
      const paragraphs = element.querySelectorAll("p, [class*='prov-content']").length;
      const headings = element.querySelectorAll("h1, h2, h3, [class*='prov-article'], [class*='prov-chapter']").length;
      const textLength = visibleTextLength(element);
      return {
        element,
        score: area * 0.01 + paragraphs * 800 + headings * 500 + textLength,
      };
    })
    .sort((left, right) => right.score - left.score);

  let root =
    document.querySelector("[data-vbpl-content-root='true']") ||
    document.querySelector(".preview-content") ||
    document.querySelector("[class*='preview-content']") ||
    scored[0]?.element ||
    document.querySelector("main") ||
    document.body;

  if (root) {
    root.setAttribute("data-vbpl-content-root", "true");
  }

  const headingCandidates = Array.from(
    (root || document).querySelectorAll(
      "[data-vbpl-title='true'], h1, h2, h3, [class*='prov-title'], [class*='doc-title']"
    )
  );

  const visibleHeadingCandidates = headingCandidates.filter((element) => {
    const rect = element.getBoundingClientRect();
    const style = window.getComputedStyle(element);
    return (
      rect.width >= 20 &&
      rect.height >= 10 &&
      style.display !== "none" &&
      style.visibility !== "hidden"
    );
  });

  let titleElement = (root || document).querySelector("[data-vbpl-title='true']");
  if (!titleElement && visibleHeadingCandidates.length > 0) {
    const ranked = visibleHeadingCandidates
      .map((element, index) => {
        const rect = element.getBoundingClientRect();
        const fontSize = parseFloat(window.getComputedStyle(element).fontSize || "16");
        const topBonus = Math.max(0, 1200 - Math.max(rect.top, 0)) / 12;
        const score = fontSize * 12 + topBonus + (index === 0 ? 80 : 0);
        return { element, score };
      })
      .sort((left, right) => right.score - left.score);
    titleElement = ranked[0]?.element || null;
  }

  if (titleElement) {
    titleElement.setAttribute("data-vbpl-title", "true");
  }

  return {
    has_root: Boolean(root),
    paragraph_count: (root || document).querySelectorAll("p, [class*='prov-content']").length,
    title: titleElement ? (titleElement.innerText || "").trim() : "",
  };
})();
"""

CLICK_FULL_TEXT_TAB_SCRIPT = r"""
return (() => {
  const candidates = Array.from(document.querySelectorAll("[role='tab'], button, a"));
  const target = candidates.find((element) => {
    const text = (element.textContent || "").trim().toLowerCase();
    return text === "nội dung" || text === "noi dung" || text === "full text";
  });
  if (target) {
    target.click();
    return true;
  }
  return false;
})();
"""

INDEX_ROUTES = {
    "trung-uong": [urljoin(BASE_URL, "/van-ban/trung-uong")],
    "dia-phuong": [urljoin(BASE_URL, "/van-ban/dia-phuong")],
    "all": [
        urljoin(BASE_URL, "/van-ban/trung-uong"),
        urljoin(BASE_URL, "/van-ban/dia-phuong"),
    ],
}


def configure_logging(log_file: Path, verbose: bool = False) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("vbpl_crawler")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    normalized = urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, "", ""))
    return normalized.rstrip("/")


def is_document_url(url: str) -> bool:
    return bool(DOCUMENT_URL_RE.match(normalize_url(url)))


def build_session(user_agent: str) -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
        }
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def load_existing_manifest(manifest_path: Path, raw_html_dir: Path) -> Dict[str, str]:
    existing: Dict[str, str] = {}
    if not manifest_path.exists():
        return existing

    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        url = normalize_url(str(record.get("url", "")))
        file_name = str(record.get("file_name", ""))
        if not url or not file_name:
            continue
        if (raw_html_dir / file_name).exists():
            existing[url] = file_name
    return existing


def next_document_index(raw_html_dir: Path) -> int:
    highest = 0
    for path in raw_html_dir.glob("doc_*.html"):
        match = DOC_FILE_RE.match(path.name)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


@dataclass
class CrawlConfig:
    target_count: int
    scope: str
    seed_mode: str
    min_delay: float
    max_delay: float
    wait_timeout: int
    page_load_timeout: int
    max_index_pages: int
    headless: bool
    user_agent: str
    force_refresh: bool
    verbose: bool


class VbplCrawler:
    def __init__(self, config: CrawlConfig) -> None:
        self.config = config
        self.project_root = Path(__file__).resolve().parents[1]
        self.raw_html_dir = self.project_root / "data" / "raw_html"
        self.raw_html_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.raw_html_dir / "manifest.jsonl"
        self.state_path = self.raw_html_dir / "crawl_state.json"
        self.logger = configure_logging(self.project_root / "crawler" / "crawl_vbpl.log", config.verbose)
        self.session = build_session(config.user_agent)
        self.robots, self.sitemap_urls = self._load_robots_and_sitemaps()
        self.downloaded = {} if config.force_refresh else load_existing_manifest(self.manifest_path, self.raw_html_dir)
        self.next_index = next_document_index(self.raw_html_dir)
        self.driver: Optional[WebDriver] = None
        self.saved_this_run = 0
        self.errors = 0

    def _load_robots_and_sitemaps(self) -> tuple[RobotFileParser, List[str]]:
        robots_url = urljoin(BASE_URL, "/robots.txt")
        response = self.session.get(robots_url, timeout=30)
        response.raise_for_status()
        lines = response.text.splitlines()

        parser = RobotFileParser()
        parser.parse(lines)

        sitemaps: List[str] = []
        for line in lines:
            if line.lower().startswith("sitemap:"):
                sitemap_url = line.split(":", 1)[1].strip()
                if sitemap_url:
                    sitemaps.append(sitemap_url)

        if not sitemaps:
            sitemaps = [urljoin(BASE_URL, "/sitemap.xml")]

        self.logger.info("Loaded robots.txt from %s", robots_url)
        self.logger.info("Discovered %d sitemap entry point(s) from robots.txt", len(sitemaps))
        return parser, sitemaps

    def _assert_allowed(self, url: str) -> bool:
        allowed = self.robots.can_fetch(self.config.user_agent, url)
        if not allowed:
            self.logger.warning("Skipping disallowed URL according to robots.txt: %s", url)
        return allowed

    def _sleep(self) -> None:
        delay = random.uniform(self.config.min_delay, self.config.max_delay)
        time.sleep(delay)

    def _build_driver(self) -> WebDriver:
        if self.driver is not None:
            return self.driver

        options = Options()
        if self.config.headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1440,1800")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")
        options.add_argument("--lang=vi-VN")
        options.add_argument(f"--user-agent={self.config.user_agent}")

        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(self.config.page_load_timeout)
        self.driver = driver
        return driver

    def _fetch_xml(self, url: str) -> BeautifulSoup:
        if not self._assert_allowed(url):
            raise RuntimeError(f"robots.txt disallows {url}")
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        self._sleep()
        return BeautifulSoup(response.text, "xml")

    def collect_urls_from_sitemaps(self) -> List[str]:
        target_pool = max(self.config.target_count * 2, self.config.target_count + 100)
        queue: List[str] = list(self.sitemap_urls)
        visited_sitemaps: Set[str] = set()
        collected: List[str] = []
        seen_urls: Set[str] = set()

        while queue and len(collected) < target_pool:
            sitemap_url = queue.pop(0)
            sitemap_url = normalize_url(sitemap_url)
            if sitemap_url in visited_sitemaps:
                continue
            visited_sitemaps.add(sitemap_url)

            try:
                soup = self._fetch_xml(sitemap_url)
            except Exception as exc:
                self.logger.exception("Failed to fetch sitemap %s: %s", sitemap_url, exc)
                self.errors += 1
                continue

            sitemap_nodes = soup.find_all("sitemap")
            if sitemap_nodes:
                for node in sitemap_nodes:
                    loc = node.find("loc")
                    if not loc or not loc.text:
                        continue
                    child_url = normalize_url(loc.text)
                    if self.config.scope == "trung-uong" and "trung-uong" not in child_url and child_url != normalize_url(urljoin(BASE_URL, "/sitemap.xml")):
                        continue
                    if self.config.scope == "dia-phuong" and "dia-phuong" not in child_url and child_url != normalize_url(urljoin(BASE_URL, "/sitemap.xml")):
                        continue
                    queue.append(child_url)
                continue

            url_nodes = soup.find_all("url")
            for node in url_nodes:
                loc = node.find("loc")
                if not loc or not loc.text:
                    continue
                candidate = normalize_url(loc.text)
                if not is_document_url(candidate):
                    continue
                if candidate in seen_urls:
                    continue
                seen_urls.add(candidate)
                collected.append(candidate)
                if len(collected) >= target_pool:
                    break

        self.logger.info("Collected %d candidate document URL(s) from sitemap(s)", len(collected))
        return collected

    def _accept_consent_if_present(self, driver: WebDriver) -> None:
        xpaths = [
            "//button[contains(., 'Đồng ý')]",
            "//button[contains(., 'Chấp nhận')]",
            "//button[contains(., 'Accept')]",
            "//a[contains(., 'Đồng ý')]",
        ]
        for xpath in xpaths:
            try:
                button = driver.find_element(By.XPATH, xpath)
                driver.execute_script("arguments[0].click();", button)
                self.logger.info("Accepted consent banner automatically.")
                time.sleep(1.0)
                return
            except Exception:
                continue

    def _wait_for_browser_ready(self, driver: WebDriver) -> None:
        WebDriverWait(driver, self.config.wait_timeout).until(
            lambda browser: browser.execute_script("return document.readyState") == "complete"
        )

    def collect_urls_from_index_pages(self) -> List[str]:
        driver = self._build_driver()
        target_pool = max(self.config.target_count * 2, self.config.target_count + 100)
        collected: List[str] = []
        seen: Set[str] = set()

        for index_url in INDEX_ROUTES[self.config.scope]:
            if not self._assert_allowed(index_url):
                continue

            self.logger.info("Collecting URLs from browser index pages starting at %s", index_url)
            try:
                driver.get(index_url)
                self._wait_for_browser_ready(driver)
                self._accept_consent_if_present(driver)
            except Exception as exc:
                self.logger.exception("Failed to open index page %s: %s", index_url, exc)
                self.errors += 1
                continue

            last_signature: Optional[tuple[str, ...]] = None

            for page_number in range(1, self.config.max_index_pages + 1):
                try:
                    WebDriverWait(driver, self.config.wait_timeout).until(
                        lambda browser: browser.find_elements(By.CSS_SELECTOR, "a[href*='/van-ban/chi-tiet/']")
                    )
                except TimeoutException:
                    self.logger.warning("Timed out waiting for document links on page %d of %s", page_number, index_url)
                    break

                page_soup = BeautifulSoup(driver.page_source, "html.parser")
                page_urls = []
                for link in page_soup.select("a[href]"):
                    candidate = normalize_url(urljoin(BASE_URL, link.get("href", "")))
                    if is_document_url(candidate):
                        page_urls.append(candidate)

                signature = tuple(page_urls[:8])
                if not page_urls or signature == last_signature:
                    self.logger.info("No new pagination signature found on %s page %d", index_url, page_number)
                    break
                last_signature = signature

                new_on_page = 0
                for candidate in page_urls:
                    if candidate not in seen:
                        seen.add(candidate)
                        collected.append(candidate)
                        new_on_page += 1

                self.logger.info(
                    "Index page %d at %s yielded %d document URL(s), %d new",
                    page_number,
                    index_url,
                    len(page_urls),
                    new_on_page,
                )

                if len(collected) >= target_pool:
                    break

                next_button = None
                xpath_candidates = [
                    "//button[contains(normalize-space(.), 'Sau') and not(@disabled)]",
                    "//a[contains(normalize-space(.), 'Sau')]",
                    "//*[@role='button' and contains(normalize-space(.), 'Sau')]",
                    "//button[@aria-label='Sau' and not(@disabled)]",
                ]
                for xpath in xpath_candidates:
                    try:
                        next_button = driver.find_element(By.XPATH, xpath)
                        if next_button.is_enabled():
                            break
                    except Exception:
                        next_button = None

                if next_button is None:
                    self.logger.info("Reached the last discoverable index page for %s", index_url)
                    break

                try:
                    driver.execute_script("arguments[0].click();", next_button)
                    time.sleep(1.0)
                    self._wait_for_browser_ready(driver)
                    self._sleep()
                except Exception as exc:
                    self.logger.warning("Failed to advance pagination on %s page %d: %s", index_url, page_number, exc)
                    break

            if len(collected) >= target_pool:
                break

        self.logger.info("Collected %d candidate document URL(s) from index page(s)", len(collected))
        return collected

    def _sanitize_snapshot_html(self, raw_html: str, original_url: str) -> str:
        soup = BeautifulSoup(raw_html, "html.parser")

        for tag in soup.find_all(["script", "noscript", "iframe"]):
            tag.decompose()

        if soup.html is None:
            wrapper = BeautifulSoup("<html><head></head><body></body></html>", "html.parser")
            wrapper.body.append(soup)
            soup = wrapper

        if soup.head is None:
            soup.html.insert(0, soup.new_tag("head"))

        base_tag = soup.head.find("base")
        if base_tag is None:
            base_tag = soup.new_tag("base", href=f"{BASE_URL}/")
            soup.head.insert(0, base_tag)
        else:
            base_tag["href"] = f"{BASE_URL}/"

        meta_charset = soup.head.find("meta", attrs={"charset": True})
        if meta_charset is None:
            soup.head.insert(0, soup.new_tag("meta", charset="utf-8"))

        existing_url_meta = soup.head.find("meta", attrs={"name": "vbpl-original-url"})
        if existing_url_meta is None:
            soup.head.append(soup.new_tag("meta", attrs={"name": "vbpl-original-url", "content": original_url}))

        existing_timestamp_meta = soup.head.find("meta", attrs={"name": "vbpl-crawled-at"})
        timestamp = datetime.now(timezone.utc).isoformat()
        if existing_timestamp_meta is None:
            soup.head.append(soup.new_tag("meta", attrs={"name": "vbpl-crawled-at", "content": timestamp}))
        else:
            existing_timestamp_meta["content"] = timestamp

        style_tag = soup.head.find("style", attrs={"id": "vbpl-static-snapshot-style"})
        if style_tag is None:
            style_tag = soup.new_tag("style", id="vbpl-static-snapshot-style")
            style_tag.string = SNAPSHOT_STYLE
            soup.head.append(style_tag)

        for tag_name, attribute in (("link", "href"), ("img", "src"), ("a", "href")):
            for node in soup.find_all(tag_name):
                value = node.get(attribute)
                if value:
                    node[attribute] = urljoin(BASE_URL, value)

        return "<!DOCTYPE html>\n" + str(soup)

    def _extract_snapshot(self, url: str) -> tuple[str, str]:
        driver = self._build_driver()
        driver.get(url)
        self._wait_for_browser_ready(driver)
        self._accept_consent_if_present(driver)

        try:
            driver.execute_script(CLICK_FULL_TEXT_TAB_SCRIPT)
        except Exception:
            pass

        try:
            WebDriverWait(driver, self.config.wait_timeout).until(
                lambda browser: browser.execute_script(
                    """
                    const root = document.querySelector('.preview-content')
                        || document.querySelector("[class*='preview-content']")
                        || document.querySelector('main')
                        || document.body;
                    return root && root.innerText && root.innerText.trim().length > 250;
                    """
                )
            )
        except TimeoutException:
            self.logger.warning("Timed out waiting for full text on %s. Saving best-effort snapshot.", url)

        metadata = driver.execute_script(MARK_CONTENT_SCRIPT) or {}
        html = driver.execute_script("return document.documentElement.outerHTML;")
        static_html = self._sanitize_snapshot_html(html, url)

        title = str(metadata.get("title") or "").strip()
        if not title:
            soup = BeautifulSoup(static_html, "html.parser")
            title_node = soup.select_one("[data-vbpl-title='true']") or soup.find("title")
            title = title_node.get_text(" ", strip=True) if title_node else ""

        return static_html, title

    def _append_manifest_record(self, record: dict) -> None:
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_state(self, candidate_total: int) -> None:
        state = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "scope": self.config.scope,
            "seed_mode": self.config.seed_mode,
            "target_count": self.config.target_count,
            "downloaded_total": len(self.downloaded),
            "saved_this_run": self.saved_this_run,
            "candidate_total": candidate_total,
            "errors": self.errors,
        }
        self.state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    def _candidate_urls(self) -> List[str]:
        collected: List[str] = []
        seen: Set[str] = set()

        if self.config.seed_mode in {"sitemap", "both"}:
            for url in self.collect_urls_from_sitemaps():
                if url not in seen:
                    seen.add(url)
                    collected.append(url)

        if self.config.seed_mode in {"index", "both"}:
            for url in self.collect_urls_from_index_pages():
                if url not in seen:
                    seen.add(url)
                    collected.append(url)

        return collected

    def run(self) -> int:
        remaining_needed = max(0, self.config.target_count - len(self.downloaded))
        if remaining_needed == 0 and not self.config.force_refresh:
            self.logger.info(
                "Target already satisfied. %d document(s) are already present in %s.",
                len(self.downloaded),
                self.raw_html_dir,
            )
            return 0

        candidate_urls = self._candidate_urls()
        self._write_state(len(candidate_urls))

        if not candidate_urls:
            self.logger.error("No candidate document URLs were discovered.")
            return 1

        self.logger.info(
            "Starting crawl with target=%d, already downloaded=%d, remaining=%d",
            self.config.target_count,
            len(self.downloaded),
            remaining_needed,
        )

        try:
            for url in candidate_urls:
                if not self.config.force_refresh and len(self.downloaded) >= self.config.target_count:
                    break

                normalized = normalize_url(url)
                if not self.config.force_refresh and normalized in self.downloaded:
                    continue

                if not self._assert_allowed(normalized):
                    continue

                file_name = f"doc_{self.next_index:06d}.html"
                file_path = self.raw_html_dir / file_name

                try:
                    html, title = self._extract_snapshot(normalized)
                    file_path.write_text(html, encoding="utf-8")
                except Exception as exc:
                    self.errors += 1
                    self.logger.exception("Failed to crawl %s: %s", normalized, exc)
                    self._write_state(len(candidate_urls))
                    self._sleep()
                    continue

                record = {
                    "url": normalized,
                    "file_name": file_name,
                    "title": title,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                }
                self._append_manifest_record(record)
                self.downloaded[normalized] = file_name
                self.saved_this_run += 1
                self.next_index += 1
                self._write_state(len(candidate_urls))

                self.logger.info(
                    "Saved %s -> %s (%d/%d total stored)",
                    normalized,
                    file_name,
                    len(self.downloaded),
                    self.config.target_count,
                )
                self._sleep()

        finally:
            if self.driver is not None:
                try:
                    self.driver.quit()
                except WebDriverException:
                    pass

        self.logger.info(
            "Crawl finished. Saved %d new document(s), total stored=%d, errors=%d",
            self.saved_this_run,
            len(self.downloaded),
            self.errors,
        )
        return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> CrawlConfig:
    parser = argparse.ArgumentParser(description="Crawl legal documents from vbpl.vn")
    parser.add_argument("--target-count", type=int, default=1000, help="Number of documents to keep in data/raw_html")
    parser.add_argument(
        "--scope",
        choices=("trung-uong", "dia-phuong", "all"),
        default=DEFAULT_SCOPE,
        help="Which public document scope to crawl",
    )
    parser.add_argument(
        "--seed-mode",
        choices=("sitemap", "index", "both"),
        default="sitemap",
        help="Where to discover document URLs from",
    )
    parser.add_argument("--min-delay", type=float, default=1.0, help="Minimum polite delay between requests")
    parser.add_argument("--max-delay", type=float, default=2.0, help="Maximum polite delay between requests")
    parser.add_argument("--wait-timeout", type=int, default=25, help="Selenium wait timeout in seconds")
    parser.add_argument("--page-load-timeout", type=int, default=45, help="Selenium page load timeout in seconds")
    parser.add_argument("--max-index-pages", type=int, default=50, help="Max browser index pages to traverse")
    parser.add_argument(
        "--headless",
        dest="headless",
        action="store_true",
        help="Run Chrome in headless mode (default)",
    )
    parser.add_argument(
        "--no-headless",
        dest="headless",
        action="store_false",
        help="Run Chrome with a visible window for manual consent/CAPTCHA handling",
    )
    parser.set_defaults(headless=True)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="User-Agent header for requests and browser")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-download documents even if their URLs already exist in manifest.jsonl",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose console logging")

    args = parser.parse_args(argv)

    if args.min_delay <= 0 or args.max_delay <= 0:
        parser.error("Delay values must be positive.")
    if args.min_delay > args.max_delay:
        parser.error("--min-delay cannot be greater than --max-delay.")
    if args.target_count <= 0:
        parser.error("--target-count must be greater than zero.")

    return CrawlConfig(
        target_count=args.target_count,
        scope=args.scope,
        seed_mode=args.seed_mode,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
        wait_timeout=args.wait_timeout,
        page_load_timeout=args.page_load_timeout,
        max_index_pages=args.max_index_pages,
        headless=args.headless,
        user_agent=args.user_agent,
        force_refresh=args.force_refresh,
        verbose=args.verbose,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    config = parse_args(argv)
    crawler = VbplCrawler(config)
    return crawler.run()


if __name__ == "__main__":
    raise SystemExit(main())
