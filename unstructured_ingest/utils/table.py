from typing import TYPE_CHECKING, Any

from unstructured_ingest.utils.data_prep import flatten_dict
from unstructured_ingest.utils.dep_check import requires_dependencies

if TYPE_CHECKING:
    from pandas import DataFrame


@requires_dependencies(["pandas"])
def get_default_pandas_dtypes() -> dict[str, Any]:
    import pandas as pd

    # Cache the StringDtype instance
    str_dtype = pd.StringDtype()  # type: ignore

    # Dictionary with str_dtype reused for all string columns,
    # reducing repeated function calls/object creations
    return {
        "text": str_dtype,
        "type": str_dtype,
        "element_id": str_dtype,
        "filename": str_dtype,  # Optional[str]
        "filetype": str_dtype,  # Optional[str]
        "file_directory": str_dtype,  # Optional[str]
        "last_modified": str_dtype,  # Optional[str]
        "attached_to_filename": str_dtype,  # Optional[str]
        "parent_id": str_dtype,  # Optional[str]
        "category_depth": "Int64",  # Optional[int]
        "image_path": str_dtype,  # Optional[str]
        "languages": object,  # Optional[list[str]]
        "page_number": "Int64",  # Optional[int]
        "page_name": str_dtype,  # Optional[str]
        "url": str_dtype,  # Optional[str]
        "link_urls": str_dtype,  # Optional[str]
        "link_texts": object,  # Optional[list[str]]
        "links": object,
        "sent_from": object,  # Optional[list[str]],
        "sent_to": object,  # Optional[list[str]]
        "subject": str_dtype,  # Optional[str]
        "section": str_dtype,  # Optional[str]
        "header_footer_type": str_dtype,  # Optional[str]
        "emphasized_text_contents": object,  # Optional[list[str]]
        "emphasized_text_tags": object,  # Optional[list[str]]
        "text_as_html": str_dtype,  # Optional[str]
        "regex_metadata": object,
        "max_characters": "Int64",  # Optional[int]
        "is_continuation": "boolean",  # Optional[bool]
        "detection_class_prob": float,  # Optional[float],
        "sender": str_dtype,
        "coordinates_points": object,
        "coordinates_system": str_dtype,
        "coordinates_layout_width": float,
        "coordinates_layout_height": float,
        "data_source_url": str_dtype,  # Optional[str]
        "data_source_version": str_dtype,  # Optional[str]
        "data_source_record_locator": object,
        "data_source_date_created": str_dtype,  # Optional[str]
        "data_source_date_modified": str_dtype,  # Optional[str]
        "data_source_date_processed": str_dtype,  # Optional[str]
        "data_source_permissions_data": object,
        "embeddings": object,
        "regex_metadata_key": object,
    }


def convert_to_pandas_dataframe(
    elements_dict: list[dict[str, Any]],
    drop_empty_cols: bool = False,
) -> "DataFrame":
    import pandas as pd

    # Flatten metadata if it hasn't already been flattened
    for d in elements_dict:
        if metadata := d.pop("metadata", None):
            d.update(flatten_dict(metadata, keys_to_omit=["data_source_record_locator"]))

    df = pd.DataFrame.from_dict(
        elements_dict,
    )
    dt = {k: v for k, v in get_default_pandas_dtypes().items() if k in df.columns}
    df = df.astype(dt)
    if drop_empty_cols:
        df.dropna(axis=1, how="all", inplace=True)
    return df
