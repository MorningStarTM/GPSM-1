"""Download ONE AMASS sub-dataset (SMPL-X, neutral) and publish it under the
same AMASS Kaggle dataset as every other one downloaded this way -- so "KIT"
and "ACCAD" and whatever comes next all live under one Kaggle dataset (e.g.
amass-kit-smplx-neutral), not as separate Kaggle datasets each.

Call main() once per sub-dataset, with that dataset's own display name and
download URL. Nothing else is re-downloaded.

HOW TO GET A DATASET'S DOWNLOAD URL
--------------------------------------
The download buttons on the AMASS page are not plain <a href> links
(right-click -> "Copy link address" finds nothing). Capture the real
request from DevTools instead:
    1. Log in at https://amass.is.tue.mpg.de/download.php
    2. DevTools (F12) -> Network tab -> "Keep log" on -> clear the log
    3. Click that dataset's row's "SMPL-X N" button
    4. Right-click the request that appears -> Copy -> Copy as cURL
    5. Take just the URL out of that curl command (the one starting
       https://download.is.tue.mpg.de/download.php?...) and pass it to
       main() below.

A wrong or expired URL is not a silent failure: `download_dataset_archive`
refuses to accept an HTML error page pretending to be an archive.

SETUP ON KAGGLE
----------------
1. Turn Internet ON for the notebook (Settings > Internet).
2. Add four notebook Secrets (Add-ons > Secrets) -- NOT hardcoded here, and
   not the same thing as a downloaded kaggle.json file:
     AMASS_EMAIL, AMASS_PASSWORD   your login at amass.is.tue.mpg.de
     KAGGLE_USERNAME                your Kaggle username
     KAGGLE_KEY                     the "key" field from Account > Create
                                     New API Token's downloaded kaggle.json
   Saving a secret to your account is not enough -- each one also has to be
   toggled ON for *this* notebook in that same Add-ons > Secrets panel.
   If you ever paste a real password or API key into a chat or a terminal,
   treat it as burned: regenerate the Kaggle token and change the AMASS
   password rather than reuse them.
3. Run this in a normal code cell -- NOT `!python scripts/...` (a shell
   subprocess can't reliably reach the Secrets connection, and can't be
   typed into if it falls back to asking):
     from scripts.download_kit_to_kaggle import main
     main("ACCAD", "https://download.is.tue.mpg.de/download.php?...")
   Call it again with a different name/url for the next sub-dataset.

Result: a private Kaggle dataset at <your-username>/<KAGGLE_DATASET_SLUG>,
with one subfolder per sub-dataset downloaded this way so far, ready for:
    from src.gpsm.motionprep.kaggle import prepare
    prepare("/kaggle/input/<KAGGLE_DATASET_SLUG>")
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

# =============================================================================
# CONFIG -- fill these in before running
# =============================================================================

DOWNLOAD_FORM_DATA: dict = {}  # every captured request so far has been a plain GET, no form body

#: The Kaggle dataset this becomes: <your-username>/<this-slug>. Lowercase,
#: hyphens only. Keep this the same as your existing AMASS KIT dataset's
#: slug if the goal is to add more sub-datasets to it, not create a new one.
KAGGLE_DATASET_SLUG = "amass-kit-smplx-neutral"

#: Where things are written inside the notebook. One archive + one
#: extracted subfolder per dataset, all collected under EXTRACT_ROOT so the
#: final published dataset has one top-level folder per sub-dataset.
WORK_DIR = Path("/kaggle/working/amass_download")
EXTRACT_ROOT = WORK_DIR / "extracted"

# =============================================================================

AMASS_LOGIN_URL = "https://amass.is.tue.mpg.de/login.php"


def _secret(name: str, env_fallback: str = "") -> str:
    """Read a Kaggle notebook Secret, falling back to an env var, then (only
    off Kaggle, e.g. local testing) to an interactive prompt.

    On Kaggle a failure here almost always means one of two things, and the
    error says so rather than silently dropping into a prompt that a
    `!python script.py` subprocess cannot actually be typed into:
      1. the secret exists in your account but was never toggled ON for
         *this* notebook (Add-ons > Secrets lists your saved secrets with a
         per-notebook attach switch -- having one saved is not enough), or
      2. the script was run as a shell subprocess (`!python ...`) instead
         of in-kernel, and the Secrets connection is not reliably available
         to a child subprocess.
    """
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception as error:
        if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:  # i.e. this is Kaggle
            raise RuntimeError(
                f"Could not read secret '{name}' ({error}). On Kaggle:\n"
                f"  1. Add-ons > Secrets -> make sure '{name}' is attached "
                f"(toggled ON) for THIS notebook, not just saved to your "
                f"account.\n"
                f"  2. Run this in a normal code cell, not `!python ...`:\n"
                f"       from scripts.download_kit_to_kaggle import main\n"
                f"       main(\"ACCAD\", \"https://download.is.tue.mpg.de/...\")"
            ) from error
    if env_fallback and os.environ.get(env_fallback):
        return os.environ[env_fallback]
    import getpass
    return getpass.getpass(f"{name}: ") if "PASSWORD" in name or "KEY" in name else input(f"{name}: ")


def login_to_amass(email: str, password: str):
    """Return a requests.Session already authenticated against AMASS.

    Field names (`username`, `password`, `commit`) were read directly from
    the live login form's HTML, not guessed.
    """
    import requests

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    response = session.post(
        AMASS_LOGIN_URL,
        data={"username": email, "password": password, "commit": "Log in"},
        allow_redirects=True,
    )
    # A failed login re-renders the same login form; a successful one takes
    # you somewhere else. This is a soft check, not a guarantee, but it turns
    # a wrong password into a clear error here instead of a corrupt download
    # later. The site also renders its own reason (e.g. "Username / Password
    # incorrect") in an `alert-danger` div -- surface that verbatim instead
    # of a generic guess, since it is the actual answer, not a guess at one.
    if 'name="password"' in response.text and "login" in response.url.lower():
        import re
        site_reason = re.search(
            r"alert-danger[^>]*>(?:\s*<button[^>]*>.*?</button>\s*)?(.*?)</div>",
            response.text, re.S,
        )
        detail = (
            re.sub(r"<[^>]+>", "", site_reason.group(1)).strip()
            if site_reason else "no error message was rendered by the site"
        )
        raise RuntimeError(
            f"AMASS login failed. The site says: {detail!r}. If you rotated "
            f"AMASS_PASSWORD recently (recommended after typing it in a "
            f"chat/terminal), make sure the Kaggle Secret was updated to "
            f"match -- a stale secret is the most common cause of this. "
            f"Also check for a trailing space or newline pasted into the "
            f"AMASS_EMAIL / AMASS_PASSWORD secrets."
        )
    print("Logged in to AMASS.")

    # download.is.tue.mpg.de is a different host from amass.is.tue.mpg.de.
    # A captured DevTools request confirmed the browser sends this same
    # PHPSESSID there, but `requests` follows cookie-domain scoping strictly
    # and will not forward a cookie set for one host to another on its own
    # -- so it is copied across explicitly rather than assumed.
    phpsessid = session.cookies.get("PHPSESSID", domain="amass.is.tue.mpg.de")
    if phpsessid:
        session.cookies.set("PHPSESSID", phpsessid, domain="download.is.tue.mpg.de")
    return session


def download_dataset_archive(
    session, url: str, out_path: Path, form_data: dict | None = None
) -> None:
    """Download one dataset's archive with the authenticated session, and
    refuse to accept an HTML page pretending to be one (the classic
    silent-failure mode for gated downloads: auth fails, server serves you
    a login page, and a script that doesn't check ends up "successfully"
    saving 2KB of HTML as if it were a real dataset -- including a wrong
    guess at a dataset's internal `name`, since that also 404s to an HTML
    page rather than an archive).

    ``form_data``: pass a non-empty dict if DevTools showed the button as a
    POST with a form body; leave it empty for a plain GET link (every
    dataset captured so far has been a plain GET).
    """
    if not url:
        raise ValueError("No download URL given.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from: {url}" + (" (POST)" if form_data else " (GET)"))
    # A captured DevTools request showed the browser sending Referer:
    # https://amass.is.tue.mpg.de/ -- these gated download endpoints
    # commonly check it as a weak anti-hotlinking measure, so match it
    # rather than let the request look like it came from nowhere.
    headers = {"Referer": "https://amass.is.tue.mpg.de/"}
    requester = (
        (lambda **kw: session.post(url, data=form_data, headers=headers, **kw))
        if form_data
        else (lambda **kw: session.get(url, headers=headers, **kw))
    )
    with requester(stream=True, allow_redirects=True) as response:
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type:
            import re
            snippet = re.sub(r"<[^>]+>", " ", response.text)
            snippet = re.sub(r"\s+", " ", snippet).strip()[:300]
            raise RuntimeError(
                f"Got HTML back instead of an archive (Content-Type: "
                f"{content_type}). This host (download.is.tue.mpg.de) is "
                f"separate from the one you log in to (amass.is.tue.mpg.de) "
                f"-- if the PHPSESSID cookie / Referer header this script "
                f"sends ever stop being enough, that link may have changed. "
                f"What the page actually said: {snippet!r}"
            )

        total = int(response.headers.get("Content-Length", 0))
        written = 0
        with open(out_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):  # 1 MB
                handle.write(chunk)
                written += len(chunk)
                if total:
                    print(f"\r  {written / 1e6:7.1f} / {total / 1e6:.1f} MB", end="")
        print()

    _assert_looks_like_an_archive(out_path)
    print(f"Saved {out_path.stat().st_size / 1e6:.1f} MB to {out_path}")


def _assert_looks_like_an_archive(path: Path) -> None:
    """Sniff the first bytes rather than trust the file extension -- catches
    the same silent-HTML-download failure the Content-Type check might miss
    (some servers mislabel it as application/octet-stream)."""
    with open(path, "rb") as handle:
        head = handle.read(8)
    magic_ok = (
        head[:2] == b"BZ"          # .tar.bz2
        or head[:2] == b"\x1f\x8b"  # .tar.gz
        or head[:2] == b"PK"        # .zip
        or head[:4] == b"7z\xbc\xaf"
    )
    if not magic_ok or head.lstrip().startswith(b"<"):
        raise RuntimeError(
            f"Downloaded file does not look like an archive (first bytes: "
            f"{head!r}). Most likely an expired link or a failed login "
            f"produced an HTML page instead."
        )


def extract_archive(archive_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting to {out_dir} ...")
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(out_dir)
    else:
        with tarfile.open(archive_path, "r:*") as tf:  # auto-detects gz/bz2/xz
            tf.extractall(out_dir)

    npz_count = sum(1 for _ in out_dir.rglob("*.npz"))
    print(f"Extracted. {npz_count} .npz files found.")
    if npz_count == 0:
        raise RuntimeError(
            "Extraction produced no .npz files -- the archive may hold a "
            "different layout than expected. Check the extracted contents "
            f"manually under {out_dir}."
        )
    return out_dir


def publish_to_kaggle(
    source_dir: Path, dataset_slug: str, kaggle_username: str, kaggle_key: str
) -> None:
    """Push `source_dir` as a private Kaggle dataset via the Kaggle CLI.

    The `kaggle` package reads credentials from the environment at import
    time, so KAGGLE_USERNAME/KAGGLE_KEY must be set *before* it is imported
    or invoked -- hence setting os.environ here rather than writing
    ~/.kaggle/kaggle.json (either works; this avoids a file with a
    credential sitting on disk).

    `kaggle_username` is passed in explicitly rather than guessed from a
    Kaggle-internal environment variable, since which env var (if any)
    reliably carries it could not be confirmed -- a wrong guess would fail
    silently by publishing to the wrong dataset id, which is worse than
    asking for one more secret.
    """
    os.environ["KAGGLE_USERNAME"] = kaggle_username
    os.environ["KAGGLE_KEY"] = kaggle_key

    username = kaggle_username
    metadata = {
        "title": "AMASS (SMPL-X, neutral)",  # not "... KIT ..." anymore -- this now holds several sub-datasets
        "id": f"{username}/{dataset_slug}",
        "licenses": [{"name": "other"}],  # AMASS license, not CC0
    }
    (source_dir / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Publishing {source_dir} as Kaggle dataset {metadata['id']} ...")
    create = subprocess.run(
        ["kaggle", "datasets", "create", "-p", str(source_dir), "-r", "zip"],
        capture_output=True, text=True,
    )
    print(create.stdout)
    if create.returncode != 0:
        if "already exists" in (create.stdout + create.stderr).lower():
            print("Dataset already exists -- pushing a new version instead.")
            version = subprocess.run(
                ["kaggle", "datasets", "version", "-p", str(source_dir),
                 "-m", "update", "-r", "zip"],
                capture_output=True, text=True,
            )
            print(version.stdout)
            if version.returncode != 0:
                print(version.stderr, file=sys.stderr)
                raise RuntimeError("Kaggle dataset version push failed.")
        else:
            print(create.stderr, file=sys.stderr)
            raise RuntimeError("Kaggle dataset create failed.")

    print(f"\nDone. Use it in later notebooks with:")
    print(f'  prepare("/kaggle/input/{dataset_slug}")')


def main(display_name: str, url: str) -> None:
    """Download ONE AMASS sub-dataset and publish it under the AMASS Kaggle
    dataset. Only `display_name`/`url` are fetched -- nothing else is
    re-downloaded.

    Args:
        display_name: Becomes that sub-dataset's subfolder name in the
            published Kaggle dataset (e.g. "ACCAD").
        url: The exact URL captured from DevTools for that dataset's
            "SMPL-X N" button (see the module docstring for how).
    """
    email = _secret("AMASS_EMAIL", "AMASS_EMAIL")
    password = _secret("AMASS_PASSWORD", "AMASS_PASSWORD")
    kaggle_username = _secret("KAGGLE_USERNAME", "KAGGLE_USERNAME")
    kaggle_key = _secret("KAGGLE_KEY", "KAGGLE_KEY")

    session = login_to_amass(email, password)

    archive_path = WORK_DIR / f"{display_name}.archive"
    out_dir = EXTRACT_ROOT / display_name
    download_dataset_archive(session, url, archive_path, DOWNLOAD_FORM_DATA)
    extract_archive(archive_path, out_dir)

    publish_to_kaggle(EXTRACT_ROOT, KAGGLE_DATASET_SLUG, kaggle_username, kaggle_key)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <display_name> <url>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
