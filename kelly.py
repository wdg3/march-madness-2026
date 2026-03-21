"""Shared Kelly Criterion calculations for bet sizing."""


def kelly_fraction(p, cost):
    """Kelly fraction for a binary contract.

    Args:
        p: Estimated probability of winning.
        cost: Price to buy one contract (0 to 1). Pays $1 on win.

    Returns:
        Fraction of bankroll to wager (0 if no edge).
    """
    if cost <= 0 or cost >= 1:
        return 0.0
    b = (1.0 / cost) - 1  # net payout per dollar wagered
    if b <= 0:
        return 0.0
    edge = p / cost - 1
    if edge <= 0:
        return 0.0
    return edge / b
