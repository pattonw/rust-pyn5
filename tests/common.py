import json


def blocks_in(dpath):
    return {
        str(path.relative_to(dpath))
        for path in dpath.glob('**/*')
        if path.suffix != ".json"
    }


def attrs_in(dpath):
    return json.loads((dpath / "attributes.json").read_text())
