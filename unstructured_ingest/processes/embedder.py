from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import BaseModel, Field, SecretStr

from unstructured_ingest.interfaces.process import BaseProcess
from unstructured_ingest.utils.data_prep import get_json_data

if TYPE_CHECKING:
    from unstructured_ingest.embed.interfaces import BaseEmbeddingEncoder


class EmbedderConfig(BaseModel):
    embedding_provider: Optional[
        Literal[
            "openai",
            "azure-openai",
            "huggingface",
            "bedrock",
            "vertexai",
            "voyageai",
            "octoai",
            "mixedbread-ai",
            "togetherai",
        ]
    ] = Field(default=None, description="Type of the embedding class to be used.")
    embedding_api_key: Optional[SecretStr] = Field(
        default=None,
        description="API key for the embedding model, for the case an API key is needed.",
    )
    embedding_model_name: Optional[str] = Field(
        default=None,
        description="Embedding model name, if needed. "
        "Chooses a particular LLM between different options, to embed with it.",
    )
    embedding_aws_access_key_id: Optional[str] = Field(
        default=None, description="AWS access key used for AWS-based embedders, such as bedrock"
    )
    embedding_aws_secret_access_key: Optional[SecretStr] = Field(
        default=None, description="AWS secret key used for AWS-based embedders, such as bedrock"
    )
    embedding_aws_region: Optional[str] = Field(
        default="us-west-2", description="AWS region used for AWS-based embedders, such as bedrock"
    )
    embedding_azure_endpoint: Optional[str] = Field(
        default=None,
        description="Your Azure endpoint, including the resource, "
        "e.g. `https://example-resource.azure.openai.com/`",
    )
    embedding_azure_api_version: Optional[str] = Field(
        description="Azure API version", default=None
    )

    def get_huggingface_embedder(self, embedding_kwargs: dict) -> "BaseEmbeddingEncoder":
        HuggingFaceEmbeddingConfig, HuggingFaceEmbeddingEncoder = _import_encoder_and_config("huggingface")
        return HuggingFaceEmbeddingEncoder(
            config=HuggingFaceEmbeddingConfig.model_validate(embedding_kwargs)
        )

    def get_openai_embedder(self, embedding_kwargs: dict) -> "BaseEmbeddingEncoder":
        OpenAIEmbeddingConfig, OpenAIEmbeddingEncoder = _import_encoder_and_config("openai")
        return OpenAIEmbeddingEncoder(config=OpenAIEmbeddingConfig.model_validate(embedding_kwargs))

    def get_azure_openai_embedder(self, embedding_kwargs: dict) -> "BaseEmbeddingEncoder":
        AzureOpenAIEmbeddingConfig, AzureOpenAIEmbeddingEncoder = _import_encoder_and_config("azure-openai")

        config_kwargs = {
            "api_key": self.embedding_api_key,
            "azure_endpoint": self.embedding_azure_endpoint,
        }
        if api_version := self.embedding_azure_api_version:
            config_kwargs["api_version"] = api_version
        if model_name := self.embedding_model_name:
            config_kwargs["model_name"] = model_name

        return AzureOpenAIEmbeddingEncoder(
            config=AzureOpenAIEmbeddingConfig.model_validate(config_kwargs)
        )

    def get_octoai_embedder(self, embedding_kwargs: dict) -> "BaseEmbeddingEncoder":
        OctoAiEmbeddingConfig, OctoAIEmbeddingEncoder = _import_encoder_and_config("octoai")
        return OctoAIEmbeddingEncoder(config=OctoAiEmbeddingConfig.model_validate(embedding_kwargs))

    def get_bedrock_embedder(self, embedding_kwargs: dict) -> "BaseEmbeddingEncoder":
        BedrockEmbeddingConfig, BedrockEmbeddingEncoder = _import_encoder_and_config("bedrock")

        # Rebuilt as before; do not mutate input dicts
        merged_kwargs = embedding_kwargs | {
            "aws_access_key_id": self.embedding_aws_access_key_id,
            "aws_secret_access_key": self.embedding_aws_secret_access_key.get_secret_value(),
            "region_name": self.embedding_aws_region,
        }

        return BedrockEmbeddingEncoder(
            config=BedrockEmbeddingConfig.model_validate(merged_kwargs)
        )

    def get_vertexai_embedder(self, embedding_kwargs: dict) -> "BaseEmbeddingEncoder":
        VertexAIEmbeddingConfig, VertexAIEmbeddingEncoder = _import_encoder_and_config("vertexai")
        return VertexAIEmbeddingEncoder(
            config=VertexAIEmbeddingConfig.model_validate(embedding_kwargs)
        )

    def get_voyageai_embedder(self, embedding_kwargs: dict) -> "BaseEmbeddingEncoder":
        VoyageAIEmbeddingConfig, VoyageAIEmbeddingEncoder = _import_encoder_and_config("voyageai")
        return VoyageAIEmbeddingEncoder(
            config=VoyageAIEmbeddingConfig.model_validate(embedding_kwargs)
        )

    def get_mixedbread_embedder(self, embedding_kwargs: dict) -> "BaseEmbeddingEncoder":
        MixedbreadAIEmbeddingConfig, MixedbreadAIEmbeddingEncoder = _import_encoder_and_config("mixedbread-ai")
        return MixedbreadAIEmbeddingEncoder(
            config=MixedbreadAIEmbeddingConfig.model_validate(embedding_kwargs)
        )

    def get_togetherai_embedder(self, embedding_kwargs: dict) -> "BaseEmbeddingEncoder":
        TogetherAIEmbeddingConfig, TogetherAIEmbeddingEncoder = _import_encoder_and_config("togetherai")
        return TogetherAIEmbeddingEncoder(
            config=TogetherAIEmbeddingConfig.model_validate(embedding_kwargs)
        )

    def get_embedder(self) -> "BaseEmbeddingEncoder":
        kwargs: dict[str, Any] = {}
        if self.embedding_api_key:
            kwargs["api_key"] = self.embedding_api_key.get_secret_value()
        if self.embedding_model_name:
            kwargs["model_name"] = self.embedding_model_name

        provider = self.embedding_provider
        # Branch prediction: most common providers checked first for optimization
        if provider == "openai":
            return self.get_openai_embedder(embedding_kwargs=kwargs)

        if provider == "huggingface":
            return self.get_huggingface_embedder(embedding_kwargs=kwargs)

        if provider == "octoai":
            return self.get_octoai_embedder(embedding_kwargs=kwargs)

        if provider == "bedrock":
            return self.get_bedrock_embedder(embedding_kwargs=kwargs)

        if provider == "vertexai":
            return self.get_vertexai_embedder(embedding_kwargs=kwargs)

        if provider == "voyageai":
            return self.get_voyageai_embedder(embedding_kwargs=kwargs)
        if provider == "mixedbread-ai":
            return self.get_mixedbread_embedder(embedding_kwargs=kwargs)
        if provider == "togetherai":
            return self.get_togetherai_embedder(embedding_kwargs=kwargs)
        if provider == "azure-openai":
            return self.get_azure_openai_embedder(embedding_kwargs=kwargs)

        raise ValueError(f"{provider} not a recognized encoder")


@dataclass
class Embedder(BaseProcess, ABC):
    config: EmbedderConfig

    def init(self, **kwargs: Any) -> None:
        self.config.get_embedder().initialize()

    def precheck(self) -> None:
        embedder = self.config.get_embedder()
        embedder.precheck()

    def run(self, elements_filepath: Path, **kwargs: Any) -> list[dict]:
        # TODO update base embedder classes to support async
        embedder = self.config.get_embedder()
        elements = get_json_data(path=elements_filepath)
        if not elements:
            return []
        embedded_elements = embedder.embed_documents(elements=elements)
        return embedded_elements

def _import_encoder_and_config(provider: str):
    if provider in _EMBED_IMPORTS:
        return _EMBED_IMPORTS[provider]

    if provider == "huggingface":
        from unstructured_ingest.embed.huggingface import (
            HuggingFaceEmbeddingConfig, HuggingFaceEmbeddingEncoder)
        _EMBED_IMPORTS[provider] = (HuggingFaceEmbeddingConfig, HuggingFaceEmbeddingEncoder)
    elif provider == "openai":
        from unstructured_ingest.embed.openai import (OpenAIEmbeddingConfig,
                                                      OpenAIEmbeddingEncoder)
        _EMBED_IMPORTS[provider] = (OpenAIEmbeddingConfig, OpenAIEmbeddingEncoder)
    elif provider == "azure-openai":
        from unstructured_ingest.embed.azure_openai import (
            AzureOpenAIEmbeddingConfig, AzureOpenAIEmbeddingEncoder)
        _EMBED_IMPORTS[provider] = (AzureOpenAIEmbeddingConfig, AzureOpenAIEmbeddingEncoder)
    elif provider == "octoai":
        from unstructured_ingest.embed.octoai import (OctoAiEmbeddingConfig,
                                                      OctoAIEmbeddingEncoder)
        _EMBED_IMPORTS[provider] = (OctoAiEmbeddingConfig, OctoAIEmbeddingEncoder)
    elif provider == "bedrock":
        from unstructured_ingest.embed.bedrock import (BedrockEmbeddingConfig,
                                                       BedrockEmbeddingEncoder)
        _EMBED_IMPORTS[provider] = (BedrockEmbeddingConfig, BedrockEmbeddingEncoder)
    elif provider == "vertexai":
        from unstructured_ingest.embed.vertexai import (
            VertexAIEmbeddingConfig, VertexAIEmbeddingEncoder)
        _EMBED_IMPORTS[provider] = (VertexAIEmbeddingConfig, VertexAIEmbeddingEncoder)
    elif provider == "voyageai":
        from unstructured_ingest.embed.voyageai import (
            VoyageAIEmbeddingConfig, VoyageAIEmbeddingEncoder)
        _EMBED_IMPORTS[provider] = (VoyageAIEmbeddingConfig, VoyageAIEmbeddingEncoder)
    elif provider == "mixedbread-ai":
        from unstructured_ingest.embed.mixedbreadai import (
            MixedbreadAIEmbeddingConfig, MixedbreadAIEmbeddingEncoder)
        _EMBED_IMPORTS[provider] = (MixedbreadAIEmbeddingConfig, MixedbreadAIEmbeddingEncoder)
    elif provider == "togetherai":
        from unstructured_ingest.embed.togetherai import (
            TogetherAIEmbeddingConfig, TogetherAIEmbeddingEncoder)
        _EMBED_IMPORTS[provider] = (TogetherAIEmbeddingConfig, TogetherAIEmbeddingEncoder)
    else:
        raise ValueError(f"{provider} not a recognized encoder")
    return _EMBED_IMPORTS[provider]

_EMBED_IMPORTS = {}
