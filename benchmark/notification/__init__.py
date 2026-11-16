"""Notification publish plugins. A business replaces the default publisher with its own external platform."""

from benchmark.notification.aggregate import (
    AggregatedNotification,
    NotificationCase,
    aggregate_by_root_cause,
    publish_root_cause_groups,
)
from benchmark.notification.registry import (
    LocalNotificationPublisher,
    register_notification_publisher,
    resolve_notification_publisher,
)

__all__ = [
    "AggregatedNotification",
    "LocalNotificationPublisher",
    "NotificationCase",
    "aggregate_by_root_cause",
    "publish_root_cause_groups",
    "register_notification_publisher",
    "resolve_notification_publisher",
]
