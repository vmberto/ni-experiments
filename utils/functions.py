import re


def extract_preprocessing_layer_names(objects):
    preprocessing_layer_names = []

    for obj in objects:
        if isinstance(obj, str):
            match = re.search(r"<class '.*\.(\w+)'>", obj)
            if match:
                preprocessing_layer_names.append(match.group(1))
        else:
            class_name = obj.__class__.__name__
            preprocessing_layer_names.append(class_name)

    return ", ".join(preprocessing_layer_names)
