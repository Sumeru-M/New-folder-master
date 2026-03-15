"""
milestone6/security/ — Adaptive PQC Immune Defense Layer
=========================================================
Extends the M6 crypto layer with a biologically-inspired adaptive security
system.  The existing crypto_layer.py is treated as a read-only black box;
all M6 public APIs remain unchanged.

Public entry point:
    from milestone6.security import SecurityEngine
    engine = SecurityEngine()
    report = engine.process_transaction_security(tx_dict)
"""
from milestone6.security.security_engine import SecurityEngine, SecurityReport

__all__ = ["SecurityEngine", "SecurityReport"]
