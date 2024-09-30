from typing import List, Optional

from pydantic import BaseModel, Field


class ColLineage(BaseModel):
    name: str
    unique_id: str = Field(..., alias="uniqueId")
    node_unique_id: str = Field(..., alias="nodeUniqueId")
    description: Optional[str]
    description_origin_column_name: Optional[str] = Field(
        ..., alias="descriptionOriginColumnName"
    )
    description_origin_resource_unique_id: Optional[str] = Field(
        ..., alias="descriptionOriginResourceUniqueId"
    )
    transformation_type: str = Field(..., alias="transformationType")
    relationship: str
    parent_columns: List[str] = Field(..., alias="parentColumns")
    child_columns: List[str] = Field(..., alias="childColumns")


class Column(BaseModel):
    lineage: List[ColLineage]


class Data(BaseModel):
    column: Column


class Model(BaseModel):
    data: Data
