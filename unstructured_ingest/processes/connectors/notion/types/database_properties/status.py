# https://developers.notion.com/reference/property-object#status
from dataclasses import dataclass, field
from typing import List, Optional

from htmlBuilder.attributes import Style
from htmlBuilder.tags import Div, HtmlTag

from unstructured_ingest.processes.connectors.notion.interfaces import (
    DBCellBase,
    DBPropertyBase,
    FromJSONMixin,
)


@dataclass
class StatusOption(FromJSONMixin):
    color: str
    id: str
    name: str
    description: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


@dataclass
class StatusGroup(FromJSONMixin):
    color: str
    id: str
    name: str
    option_ids: List[str] = field(default_factory=List[str])

    @classmethod
    def from_dict(cls, data: dict):
        # Avoid unnecessary dictionary copying or wrapping.
        # Directly call cls with unpacked data.
        return cls.__new__(cls, **data)


@dataclass
class StatusProp(FromJSONMixin):
    options: List[StatusOption] = field(default_factory=list)
    groups: List[StatusGroup] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict):
        options_data = data.get("options", [])
        groups_data = data.get("groups", [])
        option_from_dict = StatusOption.from_dict
        group_from_dict = StatusGroup.from_dict

        return cls(
            options=[option_from_dict(o) for o in options_data],
            groups=[group_from_dict(g) for g in groups_data],
        )


@dataclass
class Status(DBPropertyBase):
    id: str
    name: str
    status: StatusProp
    type: str = "status"
    description: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict):
        return cls(status=StatusProp.from_dict(data.pop("status", {})), **data)


@dataclass
class StatusCell(DBCellBase):
    id: str
    status: Optional[StatusOption]
    type: str = "status"
    name: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict):
        return cls(status=StatusOption.from_dict(data.pop("status", {})), **data)

    def get_html(self) -> Optional[HtmlTag]:
        if status := self.status:
            select_attr = []
            if status.color and status.color != "default":
                select_attr.append(Style(f"color: {status.color}"))
            return Div(select_attr, status.name)
        return None
