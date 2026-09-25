"""Download AMASS KIT (SMPL-X, neutral) and re-publish it as a Kaggle dataset.

Meant to be run once, inside a Kaggle notebook, so the KIT motions live as a
Kaggle dataset that ``src/gpsm/motionprep/kaggle.py``'s ``prepare()`` can then
point at in every later notebook, without re-downloading from AMASS each time.

WHERE KIT_DOWNLOAD_URL BELOW CAME FROM
----------------------------------------
The download buttons on that page are not plain <a href> links (right-click
-> "Copy link address" finds nothing), so it was captured from DevTools
instead: Network tab -> "Keep log" on -> click the KIT row's "SMPL-X N"
button -> right-click the resulting request -> Copy -> Copy as cURL. That
showed a plain GET to download.is.tue.mpg.de with the file path baked into
the URL as a query parameter, authenticated purely by a session cookie --
which is why login_to_amass() below still matters: it establishes that same
kind of session itself, with your own credentials, rather than reusing
anyone's copied-from-a-browser cookie (those expire, and are not something
to keep lying around in a script anyway).

If AMASS ever restructures this page and KIT_DOWNLOAD_URL stops working,
repeat the DevTools capture above for whatever dataset/gender you need and
replace the constant -- the rest of the script does not change.

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
3. KIT_DOWNLOAD_URL is already filled in below; only change
   KAGGLE_DATASET_SLUG if you want a different dataset name.
4. Run this in a normal code cell -- NOT `!python scripts/...` (a shell
   subprocess can't reliably reach the Secrets connection, and can't be
   typed into if it falls back to asking):
     from scripts.download_kit_to_kaggle import main
     main()

Result: a private Kaggle dataset at <your-username>/<KAGGLE_DATASET_SLUG>,
ready for:
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

#: Captured from DevTools (Network tab -> "Copy as cURL") on the KIT row's
#: "SMPL-X N" button: a plain GET, auth carried entirely by the PHPSESSID
#: cookie -- which is exactly what login_to_amass() below obtains itself,
#: so no form data is needed (this script does not reuse your browser's
#: session/cookie; it logs in fresh with your AMASS_EMAIL/AMASS_PASSWORD).
#: sfile confirms this is MoSh++'s own neutral-gender SMPL-X fit of KIT --
#: exactly the "AMASS instead of refitting c3d ourselves" data this script
#: exists to fetch.
KIT_DOWNLOAD_URL = (
    "https://download.is.tue.mpg.de/download.php"
    "?domain=amass&resume=1"
    "&sfile=amass_per_dataset/smplx/neutral/mosh_results/KIT.tar.bz2"
)
KIT_DOWNLOAD_FORM_DATA: dict = {}

#: The Kaggle dataset this becomes: <your-username>/<this-slug>. Lowercase,
#: hyphens only.
KAGGLE_DATASET_SLUG = "amass-kit-smplx-neutral"

#: Where things are written inside the notebook.
WORK_DIR = Path("/kaggle/working/kit_download")
ARCHIVE_PATH = WORK_DIR / "kit_smplx_n.archive"
EXTRACT_DIR = WORK_DIR / "kit_smplx_n"

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
                f"       main()"
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
    return session


def download_kit_archive(
    session, url: str, out_path: Path, form_data: dict | None = None
) -> None:
    """Download the KIT archive with the authenticated session, and refuse
    to accept an HTML page pretending to be one (the classic silent-failure
    mode for gated downloads: auth fails, server serves you a login page,
    and a script that doesn't check ends up "successfully" saving 2KB of
    HTML as if it were a 660MB dataset).

    ``form_data``: pass a non-empty dict if DevTools showed the button as a
    POST with a form body (see the KIT_DOWNLOAD_FORM_DATA comment above);
    leave it empty for a plain GET link.
    """
    if not url:
        raise ValueError(
            "KIT_DOWNLOAD_URL is empty. See the comment above "
            "KIT_DOWNLOAD_URL for how to get it from DevTools -- the "
            "download buttons on that page are not plain links."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from: {url}" + (" (POST)" if form_data else " (GET)"))
    requester = (
        (lambda **kw: session.post(url, data=form_data, **kw))
        if form_data
        else (lambda **kw: session.get(url, **kw))
    )
    with requester(stream=True, allow_redirects=True) as response:
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type:
            raise RuntimeError(
                f"Got HTML back instead of an archive (Content-Type: "
                f"{content_type}). This almost always means the session "
                f"was not authenticated for this file, or the link expired "
                f"-- re-copy KIT_DOWNLOAD_URL from a fresh logged-in page."
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
        "title": "AMASS KIT (SMPL-X, neutral)",
        "id": f"{username}/{dataset_slug}",
        "licenses": [{"name": "other"}],  # AMASS/KIT license, not CC0
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


def main() -> None:
    email = _secret("AMASS_EMAIL", "AMASS_EMAIL")
    password = _secret("AMASS_PASSWORD", "AMASS_PASSWORD")
    kaggle_username = _secret("KAGGLE_USERNAME", "KAGGLE_USERNAME")
    kaggle_key = _secret("KAGGLE_KEY", "KAGGLE_KEY")

    session = login_to_amass(email, password)
    download_kit_archive(session, KIT_DOWNLOAD_URL, ARCHIVE_PATH, KIT_DOWNLOAD_FORM_DATA)
    extract_archive(ARCHIVE_PATH, EXTRACT_DIR)
    publish_to_kaggle(EXTRACT_DIR, KAGGLE_DATASET_SLUG, kaggle_username, kaggle_key)


if __name__ == "__main__":
    main()
