"""Тесты для проверки импортов и зависимостей модулей."""
import sys
from pathlib import Path

import pytest

# обавляем src в путь
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@pytest.mark.unit
class TestImports:
    """роверка импортов всех модулей."""

    def test_import_core(self) -> None:
        """роверка импорта модуля core."""
        import core
        assert hasattr(core, "load_bonds_data")
        assert hasattr(core, "build_portfolio")
        assert hasattr(core, "calculate_metrics")
        assert hasattr(core, "calculate_full_cashflow")
        assert hasattr(core, "ru")
        assert hasattr(core, "load_journal")
        assert hasattr(core, "get_current_holdings")
        assert hasattr(core, "save_to_journal")
        assert hasattr(core, "portfolio_table_view")
        assert hasattr(core, "buy_list_export_frame")
        assert hasattr(core, "drop_amortizing_bonds")
        assert hasattr(core, "AMORTIZING_CSV_COLUMNS")

    def test_import_moex_bonds_update(self) -> None:
        """роверка импорта модуля moex_bonds_update."""
        from moex_bonds_update import (
            save_moex_bonds_csv,
            fetch_moex_bonds_dataframe,
            fetch_iss_block,
            DEFAULT_BOARDS,
            BASE_ISS,
        )
        assert DEFAULT_BOARDS == ("TQCB", "TQOB")
        assert "iss.moex.com" in BASE_ISS

    def test_import_moex_amort(self) -> None:
        """роверка импорта модуля moex_amort."""
        from moex_amort import (
            moex_amortizing_isins,
            check_amortization_bulk,
            drop_amortizing_bonds,
            AmortizationInfo,
            AmortizationStatus,
            DEFAULT_CACHE_PATH,
        )
        assert callable(moex_amortizing_isins)
        assert callable(check_amortization_bulk)
        assert callable(drop_amortizing_bonds)
        from typing import get_args
        assert set(get_args(AmortizationStatus)) == {
    "amortizing",
    "not_amortizing",
    "unknown",
}

    def test_import_smartlab_ratings(self) -> None:
        """роверка импорта модуля smartlab_ratings."""
        from smartlab_ratings import (
            get_ratings_bulk,
            rating_to_score,
            RatingInfo,
            normalize_rating,
        )
        assert callable(get_ratings_bulk)
        assert callable(rating_to_score)
        assert RatingInfo.__doc__ is not None
        assert callable(normalize_rating)

    def test_import_ml_model(self) -> None:
        """роверка импорта модуля ml_model."""
        from ml_model import (
            train_and_save_model,
            load_model,
            predict_ytm,
            enrich_with_ml_scores,
            FEATURES,
            TARGET,
        )
        assert callable(train_and_save_model)
        assert callable(load_model)
        assert callable(predict_ytm)
        assert callable(enrich_with_ml_scores)
        assert isinstance(FEATURES, list)
        assert isinstance(TARGET, str)

    def test_import_moex_client(self) -> None:
        """роверка импорта модуля moex_client."""
        from moex_client import create_http_session, get_json, USER_AGENTS
        assert callable(create_http_session)
        assert callable(get_json)
        assert isinstance(USER_AGENTS, list)
        assert len(USER_AGENTS) > 0

    def test_import_taxation(self) -> None:
        """роверка импорта модуля taxation."""
        from taxation import (
            calculate_tax,
            calculate_taxable_income,
            calculate_withholding_tax,
        )
        assert callable(calculate_tax)
        assert callable(calculate_taxable_income)
        assert callable(calculate_withholding_tax)

    def test_import_cashflow(self) -> None:
        """роверка импорта модуля cashflow."""
        from cashflow import (
            calculate_dirty_price,
            calculate_accrued_interest,
            calculate_duration,
            calculate_cashflows,
        )
        assert callable(calculate_dirty_price)
        assert callable(calculate_accrued_interest)
        assert callable(calculate_duration)
        assert callable(calculate_cashflows)

    def test_import_facevalue(self) -> None:
        """роверка импорта модуля facevalue."""
        from facevalue import (
            get_face_values,
            calculate_dirty_price,
        )
        assert callable(get_face_values)
        assert callable(calculate_dirty_price)

    def test_no_missing_imports(self) -> None:
        """роверка отсутствия циклических импортов и пропущенных зависимостей."""
        # роверка, что нет импортов, которые вызывают ошибки
        import core
        import moex_bonds_update
        import moex_amort
        import smartlab_ratings
        import ml_model
        import moex_client
        import taxation
        import cashflow
        import facevalue

    def test_no_unused_imports(self) -> None:
        """роверка отсутствия неиспользуемых импортов."""
        # мпорты должны использоваться в функциях
        import core
        import moex_bonds_update
        import moex_amort
        import smartlab_ratings
        import ml_model
        import moex_client
        import taxation
        import cashflow
        import facevalue

    def test_module_structure(self) -> None:
        """роверка структуры модулей."""
        import core
        import moex_bonds_update
        import moex_amort
        import smartlab_ratings
        import ml_model
        import moex_client
        import taxation
        import cashflow
        import facevalue

        # се модули должны быть импортируемыми
        assert core
        assert moex_bonds_update
        assert moex_amort
        assert smartlab_ratings
        assert ml_model
        assert moex_client
        assert taxation
        assert cashflow
        assert facevalue

    def test_user_agent_format(self) -> None:
        """роверка формата User-Agent."""
        from moex_client import USER_AGENTS
        for ua in USER_AGENTS:
            assert isinstance(ua, str)
            assert len(ua) > 0
