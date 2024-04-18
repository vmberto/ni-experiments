def filter_active(configurations):
    return [config for config in configurations if config.get("active", False)]
