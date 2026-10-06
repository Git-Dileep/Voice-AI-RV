records = {
    "001": {
        "value": "demo"
    }
}


def update_record(record_id: str, value: str):
    if record_id not in records:
        return {
            "ok": False,
            "error": "Record not found"
        }

    records[record_id]["value"] = value

    return {
        "ok": True,
        "record_id": record_id,
        "value": value
    }


def get_record(record_id: str):
    if record_id not in records:
        return {
            "ok": False,
            "error": "Record not found"
        }

    return {
        "ok": True,
        "record_id": record_id,
        "value": records[record_id]["value"]
    }