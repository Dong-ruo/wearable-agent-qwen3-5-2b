import sys


def utf8_console():
    """Keep Chinese CLI output and redirected JSON consistent on Windows."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
