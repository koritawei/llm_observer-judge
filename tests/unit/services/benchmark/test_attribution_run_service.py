"""Unit tests for BenchmarkService attribution methods — resolve_item_id_by_job_id / update_attribution_job_id.

Covers two-field callback matching (source_trace_id + attribution_job_id).
The mock ORM does not touch a real DB (aligned with the tests/unit/dataset_eval/test_run_service.py convention).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.runs.run_service import BenchmarkService

_RUN_SERVICE_MODULE = "benchmark.runs.run_helpers"


@pytest.fixture()
def service() -> BenchmarkService:
    return BenchmarkService()


class TestResolveItemIdByJobId:
    """resolve_item_id_by_job_id — two-field matching that distinguishes callback type by kind."""

    # ── judge callback (kind="judge", default)──────────────────────────

    @pytest.mark.asyncio
    async def test_judge_resolves_by_source_trace_id(
        self,
        service: BenchmarkService,
    ) -> None:
        """Judge callback: look up source_trace_id first (the judge job_id), and return on a hit."""
        mock_item = MagicMock()
        mock_item.item_id = "item_001"

        with patch(
            f"{_RUN_SERVICE_MODULE}.BenchmarkRunItem",
        ) as mock_model:
            # The first filter (source_trace_id) hits
            qs1 = MagicMock()
            qs1.first = AsyncMock(return_value=mock_item)
            mock_model.filter.return_value = qs1

            result = await service.resolve_item_id_by_job_id(
                run_id="run_1",
                job_id="judge_job_123",
            )

        assert result == "item_001"
        mock_model.filter.assert_called_once_with(
            run_id="run_1",
            source_trace_id="judge_job_123",
        )

    @pytest.mark.asyncio
    async def test_judge_falls_back_to_attribution_job_id(
        self,
        service: BenchmarkService,
    ) -> None:
        """Judge callback: source_trace_id misses → look up attribution_job_id."""
        mock_item = MagicMock()
        mock_item.item_id = "item_002"

        with patch(
            f"{_RUN_SERVICE_MODULE}.BenchmarkRunItem",
        ) as mock_model:
            # The first filter (source_trace_id) misses
            qs1 = MagicMock()
            qs1.first = AsyncMock(return_value=None)
            # The second filter (attribution_job_id) hits
            qs2 = MagicMock()
            qs2.first = AsyncMock(return_value=mock_item)
            mock_model.filter.return_value = qs1
            # The second call returns qs2
            mock_model.filter.side_effect = [qs1, qs2]

            result = await service.resolve_item_id_by_job_id(
                run_id="run_1",
                job_id="attr_task_456",
            )

        assert result == "item_002"
        assert mock_model.filter.call_count == 2

    @pytest.mark.asyncio
    async def test_judge_returns_none_when_both_miss(
        self,
        service: BenchmarkService,
    ) -> None:
        """Judge callback: both fields miss → None."""
        with patch(
            f"{_RUN_SERVICE_MODULE}.BenchmarkRunItem",
        ) as mock_model:
            qs = MagicMock()
            qs.first = AsyncMock(return_value=None)
            mock_model.filter.return_value = qs

            result = await service.resolve_item_id_by_job_id(
                run_id="run_1",
                job_id="unknown_job",
            )

        assert result is None

    # ── attribution callback (kind="attribution")────────────────────

    @pytest.mark.asyncio
    async def test_attr_resolves_by_attribution_job_id_first(
        self,
        service: BenchmarkService,
    ) -> None:
        """Attribution callback: look up attribution_job_id first (the attribution task_id).

        Before the fix the old implementation looked up source_trace_id first, and an attribution callback's
        payload.job_id = params.job_id echoed back = the judge source_trace_id,
        so it always hit step-1 and step-2 (the attribution_job_id lookup) was dead code.
        """
        mock_item = MagicMock()
        mock_item.item_id = "item_010"

        with patch(
            f"{_RUN_SERVICE_MODULE}.BenchmarkRunItem",
        ) as mock_model:
            # The first filter (attribution_job_id) hits
            qs1 = MagicMock()
            qs1.first = AsyncMock(return_value=mock_item)
            mock_model.filter.return_value = qs1

            result = await service.resolve_item_id_by_job_id(
                run_id="run_1",
                job_id="attr_task_789",
                kind="attribution",
            )

        assert result == "item_010"
        mock_model.filter.assert_called_once_with(
            run_id="run_1",
            attribution_job_id="attr_task_789",
        )

    @pytest.mark.asyncio
    async def test_attr_falls_back_to_source_trace_id(
        self,
        service: BenchmarkService,
    ) -> None:
        """Attribution callback: attribution_job_id misses → fall back to source_trace_id.

        Case: an early callback race — PG has not written attribution_job_id yet,
        but source_trace_id was already written during judge."""
        mock_item = MagicMock()
        mock_item.item_id = "item_011"

        with patch(
            f"{_RUN_SERVICE_MODULE}.BenchmarkRunItem",
        ) as mock_model:
            # The first filter (attribution_job_id) misses
            qs1 = MagicMock()
            qs1.first = AsyncMock(return_value=None)
            # The second filter (source_trace_id) hits
            qs2 = MagicMock()
            qs2.first = AsyncMock(return_value=mock_item)
            mock_model.filter.side_effect = [qs1, qs2]

            result = await service.resolve_item_id_by_job_id(
                run_id="run_1",
                job_id="judge_job_abc",
                kind="attribution",
            )

        assert result == "item_011"
        assert mock_model.filter.call_count == 2
        # The first lookup must query attribution_job_id
        first_call = mock_model.filter.call_args_list[0]
        assert first_call.kwargs == {
            "run_id": "run_1",
            "attribution_job_id": "judge_job_abc",
        }
        # The second lookup must query source_trace_id
        second_call = mock_model.filter.call_args_list[1]
        assert second_call.kwargs == {
            "run_id": "run_1",
            "source_trace_id": "judge_job_abc",
        }

    @pytest.mark.asyncio
    async def test_attr_returns_none_when_both_miss(
        self,
        service: BenchmarkService,
    ) -> None:
        """Attribution callback: both fields miss → None."""
        with patch(
            f"{_RUN_SERVICE_MODULE}.BenchmarkRunItem",
        ) as mock_model:
            qs = MagicMock()
            qs.first = AsyncMock(return_value=None)
            mock_model.filter.return_value = qs

            result = await service.resolve_item_id_by_job_id(
                run_id="run_1",
                job_id="unknown_job",
                kind="attribution",
            )

        assert result is None


class TestUpdateAttributionJobId:
    """update_attribution_job_id — update attribution_job_id after attribution is submitted."""

    @pytest.mark.asyncio
    async def test_updates_attribution_job_id(
        self,
        service: BenchmarkService,
    ) -> None:
        """Update the attribution_job_id field."""
        with patch(
            f"{_RUN_SERVICE_MODULE}.BenchmarkRunItem",
        ) as mock_model:
            qs = MagicMock()
            qs.update = AsyncMock(return_value=1)
            mock_model.filter.return_value = qs

            await service.update_attribution_job_id(
                run_id="run_1",
                item_id="item_001",
                attribution_job_id="attr_task_789",
            )

        mock_model.filter.assert_called_once_with(
            run_id="run_1",
            item_id="item_001",
            criterion_id="fused",
        )
        qs.update.assert_awaited_once()
        # Check that the kwargs passed in include attribution_job_id
        _, kwargs = qs.update.call_args
        assert kwargs["attribution_job_id"] == "attr_task_789"


class TestIncrementAttributionReceivedOnce:
    """Idempotently increment attribution_received_count per item."""

    @pytest.mark.asyncio
    async def test_first_claim_increments(
        self,
        service: BenchmarkService,
    ) -> None:
        run = MagicMock()
        run.config_json = {"attribution_received_count": 0}
        run.save = AsyncMock()
        qs = MagicMock()
        qs.select_for_update.return_value = qs
        qs.using_db.return_value = qs
        qs.first = AsyncMock(return_value=run)
        tx = MagicMock()
        tx.__aenter__ = AsyncMock(return_value=MagicMock())
        tx.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(f"{_RUN_SERVICE_MODULE}.in_transaction", return_value=tx),
            patch(f"{_RUN_SERVICE_MODULE}.BenchmarkRun.filter", return_value=qs),
            patch(f"{_RUN_SERVICE_MODULE}._stamp_bucket", side_effect=lambda x: x),
        ):
            count, config, claimed = await service.increment_attribution_received_once(
                "run_1",
                "item_001",
            )

        assert claimed is True
        assert count == 1
        assert config["attribution_received_item_ids"] == ["item_001"]
        run.save.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_second_claim_is_idempotent(
        self,
        service: BenchmarkService,
    ) -> None:
        run = MagicMock()
        run.config_json = {
            "attribution_received_count": 1,
            "attribution_received_item_ids": ["item_001"],
        }
        run.save = AsyncMock()
        qs = MagicMock()
        qs.select_for_update.return_value = qs
        qs.using_db.return_value = qs
        qs.first = AsyncMock(return_value=run)
        tx = MagicMock()
        tx.__aenter__ = AsyncMock(return_value=MagicMock())
        tx.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(f"{_RUN_SERVICE_MODULE}.in_transaction", return_value=tx),
            patch(f"{_RUN_SERVICE_MODULE}.BenchmarkRun.filter", return_value=qs),
        ):
            count, _, claimed = await service.increment_attribution_received_once(
                "run_1",
                "item_001",
            )

        assert claimed is False
        assert count == 1
        run.save.assert_not_awaited()


class TestIncrementAttributionExpectedOnce:
    """Idempotently increment attribution_expected_count per item (guard against an inflated 136/101)."""

    @pytest.mark.asyncio
    async def test_first_claim_increments(
        self,
        service: BenchmarkService,
    ) -> None:
        run = MagicMock()
        run.config_json = {"attribution_expected_count": 0}
        run.save = AsyncMock()
        qs = MagicMock()
        qs.select_for_update.return_value = qs
        qs.using_db.return_value = qs
        qs.first = AsyncMock(return_value=run)
        tx = MagicMock()
        tx.__aenter__ = AsyncMock(return_value=MagicMock())
        tx.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(f"{_RUN_SERVICE_MODULE}.in_transaction", return_value=tx),
            patch(f"{_RUN_SERVICE_MODULE}.BenchmarkRun.filter", return_value=qs),
            patch(f"{_RUN_SERVICE_MODULE}._stamp_bucket", side_effect=lambda x: x),
        ):
            count, config, claimed = await service.increment_attribution_expected_once(
                "run_1",
                "item_001",
            )

        assert claimed is True
        assert count == 1
        assert config["attribution_expected_item_ids"] == ["item_001"]
        assert config["attribution_expected_count"] == 1
        run.save.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_second_claim_is_idempotent(
        self,
        service: BenchmarkService,
    ) -> None:
        run = MagicMock()
        run.config_json = {
            "attribution_expected_count": 1,
            "attribution_expected_item_ids": ["item_001"],
        }
        run.save = AsyncMock()
        qs = MagicMock()
        qs.select_for_update.return_value = qs
        qs.using_db.return_value = qs
        qs.first = AsyncMock(return_value=run)
        tx = MagicMock()
        tx.__aenter__ = AsyncMock(return_value=MagicMock())
        tx.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(f"{_RUN_SERVICE_MODULE}.in_transaction", return_value=tx),
            patch(f"{_RUN_SERVICE_MODULE}.BenchmarkRun.filter", return_value=qs),
        ):
            count, _, claimed = await service.increment_attribution_expected_once(
                "run_1",
                "item_001",
            )

        assert claimed is False
        assert count == 1
        run.save.assert_not_awaited()


class TestIsExternalJudgeItemTerminal:
    """is_external_judge_item_terminal — the healer distinguishes a normal completed from a lost callback."""

    @pytest.mark.asyncio
    async def test_true_when_success(
        self,
        service: BenchmarkService,
    ) -> None:
        item = MagicMock()
        item.status = "success"
        with patch.object(
            service,
            "get_run_item",
            new_callable=AsyncMock,
            return_value=item,
        ):
            assert await service.is_external_judge_item_terminal("r1", "i1") is True

    @pytest.mark.asyncio
    async def test_false_when_running(
        self,
        service: BenchmarkService,
    ) -> None:
        item = MagicMock()
        item.status = "running"
        with patch.object(
            service,
            "get_run_item",
            new_callable=AsyncMock,
            return_value=item,
        ):
            assert await service.is_external_judge_item_terminal("r1", "i1") is False

    @pytest.mark.asyncio
    async def test_false_when_missing(
        self,
        service: BenchmarkService,
    ) -> None:
        with patch.object(
            service,
            "get_run_item",
            new_callable=AsyncMock,
            return_value=None,
        ):
            assert await service.is_external_judge_item_terminal("r1", "i1") is False
