# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""权重下载工具(train11111new 比赛版,从 utils/downloads.py 精简)。

保留 is_url / curl_download / safe_download / attempt_download,
删除 gsutil / url_getsize 等比赛用不到的功能。
"""

import logging
import subprocess
import urllib
from pathlib import Path

import requests
import torch


# 判断字符串是否为合法 URL(可选联网校验)
def is_url(url, check=True):
    """Determines if a string is a valid URL and optionally checks its existence online."""
    try:
        url = str(url)
        result = urllib.parse.urlparse(url)
        assert all([result.scheme, result.netloc])  # check if is url
        return (urllib.request.urlopen(url, timeout=5).getcode() == 200) if check else True  # check if exists online
    except Exception:
        return False


# 使用 curl 下载文件(支持断点续传与静默模式)
def curl_download(url, filename, *, silent: bool = False) -> bool:
    """Download a file from a url to a filename using curl."""
    silent_option = "sS" if silent else ""  # silent
    proc = subprocess.run(
        [
            "curl",
            "-#",
            f"-{silent_option}L",
            url,
            "--output",
            filename,
            "--retry",
            "9",
            "-C",
            "-",
        ],
        check=False,
    )
    return proc.returncode == 0


# 安全下载文件(校验最小字节数,失败时用 curl 重试并清理不完整文件)
def safe_download(file, url, url2=None, min_bytes=1e0, error_msg=""):
    """Downloads a file from 'url' or 'url2' to 'file', ensuring size > 'min_bytes'; removes incomplete downloads."""
    from .general import LOGGER

    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    try:  # url1
        LOGGER.info(f"Downloading {url} to {file}...")
        torch.hub.download_url_to_file(url, str(file), progress=LOGGER.level <= logging.INFO)
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg  # check
    except Exception as e:  # url2
        if file.exists():
            file.unlink()  # remove partial downloads
        LOGGER.warning(f"{e}\nRe-attempting {url2 or url} to {file}...")
        # curl download, retry and resume on fail
        curl_download(url2 or url, file)
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:  # check
            if file.exists():
                file.unlink()  # remove partial downloads
            LOGGER.error(f"{assert_msg}\n{error_msg}")
        LOGGER.info("")


# 尝试下载权重:本地不存在时,从 URL 或 GitHub release 资产下载
# Keep local (do not dedup): pinned to YOLOv3 release assets
def attempt_download(file, repo="ultralytics/yolov3", release="v9.6.0"):
    """Download a file from a URL or a GitHub release asset if it is not already present locally."""
    from .general import LOGGER

    def github_assets(repository, version="latest"):
        """Returns GitHub tag and assets for a given repository and version from the GitHub API."""
        if version != "latest":
            version = f"tags/{version}"  # i.e. tags/v7.0
        response = requests.get(f"https://api.github.com/repos/{repository}/releases/{version}", timeout=10).json()
        return response["tag_name"], [x["name"] for x in response["assets"]]  # tag, assets

    file = Path(str(file).strip().replace("'", ""))
    if not file.exists():
        # URL specified
        name = Path(urllib.parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
        if str(file).startswith(("http:/", "https:/")):  # download
            url = str(file).replace(":/", "://")  # Pathlib turns :// -> :/
            file = name.split("?")[0]  # parse authentication https://url.com/file.txt?auth...
            if Path(file).is_file():
                LOGGER.info(f"Found {url} locally at {file}")  # file already exists
            else:
                safe_download(file=file, url=url, min_bytes=1e5)
            return file

        # GitHub assets
        assets = ["yolov3.pt", "yolov3-spp.pt", "yolov3-tiny.pt"]  # default
        try:
            tag, assets = github_assets(repo, release)
        except Exception:
            try:
                tag, assets = github_assets(repo)  # latest release
            except Exception:
                try:
                    tag = subprocess.check_output("git tag", shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
                except Exception:
                    tag = release

        if name in assets:
            file.parent.mkdir(parents=True, exist_ok=True)  # make parent dir (if required)
            safe_download(
                file,
                url=f"https://github.com/{repo}/releases/download/{tag}/{name}",
                min_bytes=1e5,
                error_msg=f"{file} missing, try downloading from https://github.com/{repo}/releases/{tag}",
            )

    return str(file)
