import re
import gdown


class GDriveInstaller:
    """
    A reusable Google Drive file/folder installer.
    Supports:
    - File download
    - Folder download
    - Full URL or ID
    """

    @staticmethod
    def extract_id(url: str) -> str:
        """Extract Google Drive file/folder ID from a URL or return the ID if already one."""
        
        # Matches: /d/FILE_ID/
        match = re.search(r"/d/([^/]+)", url)
        if match:
            return match.group(1)

        # Matches: /folders/FOLDER_ID
        match = re.search(r"folders/([^/?]+)", url)
        if match:
            return match.group(1)

        # Already an ID
        return url

    @staticmethod
    def download_file(file_id: str, output_path: str):
        """Download a single file from Google Drive."""
        
        file_id = GDriveInstaller.extract_id(file_id)
        url = f"https://drive.google.com/uc?id={file_id}"
        
        print(f"[GDrive] Downloading FILE: {file_id} -> {output_path}")
        gdown.download(url, output_path, quiet=False)

    @staticmethod
    def download_folder(folder_id: str, output_dir: str):
        """Download an entire folder from Google Drive."""
        
        folder_id = GDriveInstaller.extract_id(folder_id)
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        
        print(f"[GDrive] Downloading FOLDER: {folder_id} -> {output_dir}")
        gdown.download_folder(
            url=url,
            output=output_dir,
            quiet=False,
            use_cookies=False
        )
        
        
    """Example Usage:
    GDriveInstaller.download_file("https://drive.google.com/file/d/FILE_ID/view?usp=sharing", "local_file.txt")
    GDriveInstaller.download_folder("https://drive.google.com/drive/folders/FOLDER_ID?usp=sharing", "local_folder")
    """
