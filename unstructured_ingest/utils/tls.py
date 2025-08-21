import os
import ssl
from functools import lru_cache

import certifi


def ssl_context_with_optional_ca_override():
    """
    # https://www.python-httpx.org/advanced/ssl/#working-with-ssl_cert_file-and-ssl_cert_dir
    # We choose REQUESTS_CA_BUNDLE because that works with many other Python packages.
    """
    ca_bundle = os.environ.get("REQUESTS_CA_BUNDLE")
    cafile = ca_bundle if ca_bundle is not None else certifi.where()
    return _cached_ssl_context(cafile, ca_bundle)


@lru_cache(maxsize=4)
def _cached_ssl_context(cafile, capath):
    return ssl.create_default_context(
        cafile=cafile,
        capath=capath,
    )
