"""Visual bracket printer for NCAA tournament simulation results.

Usage:
    python bracket_viz.py output/pbp_v1/bracket.csv
    python bracket_viz.py output/v5/bracket.csv --title "V5 Ensemble"
"""

import argparse
import pandas as pd


def load_picks(bracket_csv: str) -> dict:
    b = pd.read_csv(bracket_csv)
    return {r["Slot"]: (r["TeamName"], r["WinPct"]) for _, r in b.iterrows()}


def _fmt_left(slot, picks, w=18, show_pct=True):
    n, pct = picks.get(slot, ("???", 0))
    s = f"{n} {pct:.0f}%" if show_pct else n
    return s.ljust(w)


def _fmt_right(slot, picks, w=18, show_pct=True):
    n, pct = picks.get(slot, ("???", 0))
    s = f"{pct:.0f}% {n}" if show_pct else n
    return s.rjust(w)


def _render_half(picks, left_region, right_region, col_widths, show_pct=True):
    rows = []
    for i in [1, 8, 5, 4, 3, 6, 7, 2]:
        L, R = left_region, right_region
        r1L = f"R1{L}{i}"
        r1R = f"R1{R}{i}"

        # Which R2/R3/R4 slot does this seed feed into?
        r2_idx = (i + 1) // 2
        r3_idx = (i + 3) // 4
        r2L = f"R2{L}{r2_idx}"
        r2R = f"R2{R}{r2_idx}"
        r3L = f"R3{L}{r3_idx}"
        r3R = f"R3{R}{r3_idx}"
        r4L = f"R4{L}1"
        r4R = f"R4{R}1"

        rows.append((r1L, r2L, r3L, r4L, r4R, r3R, r2R, r1R, i))

    # Now output in bracket order: pairs of seeds that play each other
    matchup_order = [(1, 8), (5, 4), (3, 6), (7, 2)]
    lines = []
    fl = lambda slot: _fmt_left(slot, picks, show_pct=show_pct)
    fr = lambda slot: _fmt_right(slot, picks, show_pct=show_pct)

    for mi, (s_top, s_bot) in enumerate(matchup_order):
        top_row = next(r for r in rows if r[8] == s_top)
        bot_row = next(r for r in rows if r[8] == s_bot)

        # Top seed line
        c = [""] * 8
        c[0] = fl(top_row[0])
        c[7] = fr(top_row[7])
        lines.append(c[:])

        # R2 line
        c = [""] * 8
        c[1] = fl(top_row[1])
        c[6] = fr(top_row[6])
        lines.append(c[:])

        # Bottom seed line
        c = [""] * 8
        c[0] = fl(bot_row[0])
        c[7] = fr(bot_row[7])
        lines.append(c[:])

        # S16 / E8 lines (only on certain matchup boundaries)
        if mi == 0:
            c = [""] * 8
            c[2] = fl(top_row[2])
            c[5] = fr(top_row[5])
            lines.append(c[:])
        elif mi == 1:
            c = [""] * 8
            c[3] = fl(top_row[3])
            c[4] = fr(top_row[4])
            lines.append(c[:])
        elif mi == 2:
            c = [""] * 8
            r3_2L = f"R3{left_region}2"
            r3_2R = f"R3{right_region}2"
            c[2] = fl(r3_2L)
            c[5] = fr(r3_2R)
            lines.append(c[:])
        else:
            lines.append([""] * 8)

    output = []
    for cells in lines:
        parts = []
        for i, cell in enumerate(cells):
            w = col_widths[i]
            if cell:
                parts.append(cell.ljust(w) if i < 4 else cell.rjust(w))
            else:
                parts.append(" " * w)
        output.append("".join(parts).rstrip())
    return output


def print_bracket(bracket_csv: str, title: str = "PBP Deep Model",
                  show_pct: bool = True):
    picks = load_picks(bracket_csv)
    col_widths = [18, 18, 18, 18, 18, 18, 18, 18]

    print()
    header = f"2026 NCAA TOURNAMENT — {title}"
    print(header)
    print("=" * len(header))

    # Top half: W (left) vs X (right)
    print()
    print("         WEST (W)" + " " * 80 + "EAST (X)")
    print()
    for line in _render_half(picks, "W", "X", col_widths, show_pct):
        print(line)

    # Final Four
    n_wx, p_wx = picks.get("R5WX", ("???", 0))
    n_yz, p_yz = picks.get("R5YZ", ("???", 0))
    n_ch, p_ch = picks.get("R6CH", ("???", 0))
    print()
    if show_pct:
        ff = f"{'':>54}{n_wx} {p_wx:.0f}%  >>>  {n_ch} {p_ch:.0f}%  <<<  {p_yz:.0f}% {n_yz}"
    else:
        ff = f"{'':>54}{n_wx}  >>>  {n_ch}  <<<  {n_yz}"
    print(ff)
    print()

    # Bottom half: Y (left) vs Z (right)
    print("         SOUTH (Y)" + " " * 78 + "MIDWEST (Z)")
    print()
    for line in _render_half(picks, "Y", "Z", col_widths, show_pct):
        print(line)
    print()


def main():
    parser = argparse.ArgumentParser(description="Print visual NCAA bracket")
    parser.add_argument("bracket_csv", help="Path to bracket.csv")
    parser.add_argument("--title", default="PBP Deep Model",
                        help="Title for the bracket header")
    parser.add_argument("--no-pct", action="store_true",
                        help="Hide percentages (for greedy/deterministic brackets)")
    args = parser.parse_args()
    print_bracket(args.bracket_csv, args.title, show_pct=not args.no_pct)


if __name__ == "__main__":
    main()
