"""Tortoise models with the field and registration names the service already uses."""

from uuid import uuid4

from tortoise import fields
from tortoise.models import Model

CharField = fields.CharField
CharEnumField = fields.CharEnumField
IntField = fields.IntField
FloatField = fields.FloatField
BooleanField = fields.BooleanField
TextField = fields.TextField
JSONField = fields.JSONField
DatetimeField = fields.DatetimeField
UUIDField = fields.UUIDField


from benchmark.libs.models.mixin import ObserverMixin
from benchmark.libs.models.observer import ModelObserver

class BaseModel(Model):
    id = UUIDField(pk=True, default=uuid4)
    created_at = DatetimeField(auto_now_add=True, null=True)
    updated_at = DatetimeField(auto_now=True, null=True)

    class Meta:
        abstract = True


class BaseModelWithSoftDelete(BaseModel):
    is_deleted = BooleanField(default=False)
    deleted_at = DatetimeField(null=True)

    class Meta:
        abstract = True
