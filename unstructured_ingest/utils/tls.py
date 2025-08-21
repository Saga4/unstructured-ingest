import os
import ssl

import certifi


def ssl_context_with_optional_ca_override():
    """
    # https://www.python-httpx.org/advanced/ssl/#working-with-ssl_cert_file-and-ssl_cert_dir
    # We choose REQUESTS_CA_BUNDLE because that works with many other Python packages.
    """
    ca_bundle = os.environ.get("REQUESTS_CA_BUNDLE")
    cafile = ca_bundle if ca_bundle is not None else certifi.where()

    # Check cache first
    context = _ssl_context_cache.get(cafile)
    if context is not None:
        return context

    context = ssl.create_default_context(
        cafile=cafile,
        capath=ca_bundle,
    )
    _ssl_context_cache[cafile] = context
    return context


_ssl_context_cache = {}
