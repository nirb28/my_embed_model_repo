# Utility Scripts

This directory contains various utility scripts for the my_model_repo project.

## GitHub Repository Downloader

The `github_downloader.py` script allows you to download any GitHub repository as a zip file and extract it to a destination location. It supports proxy configuration for network connections.

### Usage

```bash
python github_downloader.py --repo owner/repo --branch branch_name --dest destination_folder
```

### Command-line Options

- `--repo` or `-r`: GitHub repository URL or owner/repo format (required)
- `--branch` or `-b`: Branch, tag, or commit reference to download (default: main)
- `--dest` or `-d`: Destination directory to extract files to (default: current directory)
- `--keep-zip` or `-k`: Keep the downloaded zip file after extraction

#### Proxy Options

- `--proxy`: HTTP/HTTPS proxy URL (e.g., 'http://proxy.example.com:8080')
- `--proxy-user`: Username for proxy authentication
- `--proxy-password`: Password for proxy authentication
- `--no-proxy-verify`: Disable SSL certificate verification when using proxy

### Examples

```bash
# Basic usage (downloads main branch to current directory)
python github_downloader.py --repo username/repository

# Specify branch and destination
python github_downloader.py --repo https://github.com/username/repository --branch develop --dest ./downloads

# Keep the zip file after extraction
python utils/github_downloader.py --repo https://github.com/nirb28/my_model_repo --keep-zip --dest .

# Using a proxy server
python github_downloader.py --repo username/repository --proxy http://proxy.example.com:8080

# Using a proxy server with authentication
python github_downloader.py --repo username/repository --proxy http://proxy.example.com:8080 --proxy-user myuser --proxy-password mypass

```

### Full Path Example

If you're running from the project root:

```bash
python utils/github_downloader.py --repo username/repository --dest ./downloaded_repos
```
