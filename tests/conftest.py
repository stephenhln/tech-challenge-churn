"""Configuração compartilhada dos testes.

Garante que ``src/`` esteja no ``sys.path`` mesmo sem instalação editável e
fixa as seeds antes de qualquer teste rodar.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from churn.config import set_seeds  # noqa: E402

set_seeds()
