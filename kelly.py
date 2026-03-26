"""Shared Kelly Criterion calculations for bet sizing."""


def kelly_fraction(p, cost, fee=0.0):
    """Kelly fraction for a binary contract.

    Args:
        p: Estimated probability of winning.
        cost: Price to buy one contract (0 to 1). Pays $1 on win.
        fee: Per-contract fee (e.g. 0.02 for Kalshi/Coinbase 2¢ fee).
            Charged on buy and on winning settlement, so effective cost
            is cost + fee and effective payout is 1 - fee.

    Returns:
        Fraction of bankroll to wager (0 if no edge).
    """
    eff_cost = cost + fee
    eff_payout = 1.0 - fee
    if eff_cost <= 0 or eff_cost >= eff_payout:
        return 0.0
    net_profit = eff_payout - eff_cost  # profit per contract on win
    b = net_profit / eff_cost  # net payout per dollar wagered
    if b <= 0:
        return 0.0
    # Edge: expected value per dollar wagered
    edge = (p * eff_payout) / eff_cost - 1
    if edge <= 0:
        return 0.0
    return edge / b
