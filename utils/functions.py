import re


def extract_preprocessing_layer_names(objects, params=None):
    preprocessing_layer_names = []

    for i, obj in enumerate(objects):
        if isinstance(obj, str):
            # If the object is a string, try to extract the class name from it
            match = re.search(r"<class '.*\.(\w+)'>", obj)
            if match:
                preprocessing_layer_names.append(match.group(1))
        else:
            # If the object is not a string, it's an instance of a class
            class_name = obj.__class__.__name__
            if params and params[i] is not None:
                preprocessing_layer_names.append(f"{class_name} {params[i]}")
            else:
                preprocessing_layer_names.append(class_name)

    return ", ".join(preprocessing_layer_names)
