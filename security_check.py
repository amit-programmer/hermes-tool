# security_check.py
"""
Security validation module for uploaded documents and media files.
Verifies file size limits, MIME magic byte signatures, and prevents dangerous double extensions.
"""
import os

# Maximum file size allowed: 10MB (optimal for ZeroGPU memory constraints)
MAX_SIZE = 10 * 1024 * 1024

try:
    import magic
    HAS_MAGIC = True
except Exception:
    try:
        import magic_bin as magic
        HAS_MAGIC = True
    except Exception:
        magic = None
        HAS_MAGIC = False

def check_file(file_path: str, file_name: str):
    """
    Validates an uploaded file for security hazards:
    1. Size Check - Restricts files > 10MB to protect memory.
    2. Magic Signature Check - Detects executables disguised as documents via MIME type.
    3. Double Extension Check - Catches dangerous filenames like 'report.pdf.exe'.
    
    Returns:
        (is_safe: bool, message: str)
    """
    # 1. Size Check
    if not os.path.exists(file_path):
        return False, "File not found on server"

    file_size = os.path.getsize(file_path)
    if file_size > MAX_SIZE:
        return False, f"File 10MB se badi hai ({file_size / (1024*1024):.1f}MB)."

    if file_size == 0:
        return False, "Uploaded file is completely empty (0 bytes)."

    # 2. Dangerous Double Extension Check
    lower_name = file_name.lower()
    dangerous_exts = (".exe", ".bat", ".cmd", ".sh", ".vbs", ".js", ".ps1", ".com", ".scr", ".pif", ".bin")
    if lower_name.count(".") > 1 and any(lower_name.endswith(d) for d in dangerous_exts):
        return False, "Double extension wali dangerous file hai"

    # 3. Magic Byte & MIME Validation
    if HAS_MAGIC and magic is not None and hasattr(magic, "from_file"):
        try:
            mime = magic.from_file(file_path, mime=True).lower()

            if lower_name.endswith(".pdf") and "pdf" not in mime:
                return False, f"Fake PDF hai, MIME mismatch: {mime}"

            if lower_name.endswith(".docx") and not any(k in mime for k in ("zip", "word", "officedocument", "octet-stream")):
                return False, f"Fake Word file hai, MIME mismatch: {mime}"

            if lower_name.endswith((".xlsx", ".xls")) and not any(k in mime for k in ("zip", "excel", "officedocument", "octet-stream", "csv")):
                return False, f"Fake Excel file hai, MIME mismatch: {mime}"

            if lower_name.endswith((".pptx", ".ppt")) and not any(k in mime for k in ("zip", "powerpoint", "officedocument", "presentation")):
                return False, f"Fake PowerPoint file hai, MIME mismatch: {mime}"

            if any(k in mime for k in ("application/x-executable", "application/x-dosexec", "application/x-sharedlib")):
                return False, "Executable binary disguised as document detected."
        except Exception:
            pass

    return True, "Safe hai"
