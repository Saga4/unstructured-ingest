import base64
import hashlib
import json
import zlib


def id_to_hash(element: dict, sequence_number: int) -> str:
    """Calculates and assigns a deterministic hash as an ID.

    The hash ID is based on element's text, sequence number on page,
    page number and its filename.

    Args:
        sequence_number: index on page

    Returns: new ID value
    """
    filename = element["metadata"].get("filename")
    text = element["text"]
    page_number = element["metadata"].get("page_number")
    data = f"{filename}{text}{page_number}{sequence_number}"
    element["element_id"] = hashlib.sha256(data.encode()).hexdigest()[:32]
    return element["element_id"]


def assign_and_map_hash_ids(elements: list[dict]) -> list[dict]:
    # -- generate sequence number for each element on a page --
    # Single pass to compute sequence numbers and mapping
    old_to_new_mapping = {}
    last_page = None
    seq_on_page_counter = -1
    for element in elements:
        page = element["metadata"].get("page_number")
        if page != last_page:
            seq_on_page_counter = 0
            last_page = page
        else:
            seq_on_page_counter += 1
        old_id = element["element_id"]
        new_id = id_to_hash(element=element, sequence_number=seq_on_page_counter)
        old_to_new_mapping[old_id] = new_id

    # -- map old parent IDs to new ones --
    for e in elements:
        parent_id = e["metadata"].get("parent_id")
        if not parent_id:
            continue
        e["metadata"]["parent_id"] = old_to_new_mapping[parent_id]

    return elements


def elements_from_base64_gzipped_json(raw_s: str) -> list[dict]:
    decoded_b64_bytes = base64.b64decode(raw_s)
    elements_json_bytes = zlib.decompress(decoded_b64_bytes)
    elements_json_str = elements_json_bytes.decode("utf-8")
    element_dicts = json.loads(elements_json_str)
    return element_dicts
