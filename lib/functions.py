import re


def filter_active(configurations):
    return [config for config in configurations if config.get("active", False)]


def clean_string(s):
    clean_str = re.sub(r'_', ' ', s)
    clean_str = re.sub(r'_[\d]+', '', clean_str)
    clean_str = clean_str.title()
    return clean_str
