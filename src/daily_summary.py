"""
Sends a daily win-rate summary to Discord based on trades.json.
Run once a day (see .github/workflows/daily_summary.yml).
"""
import os
import json
import requests

TRADES_PATH = os.path.join(os.path.dirname(__file__), "..", "trades.json")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")


def load_trades():
    if os.path.exists(TRADES_PATH):
        with open(TRADES_PATH, "r") as f:
            return json.load(f)
    return []


def send_discord(embed):
    if not DISCORD_WEBHOOK_URL:
        print("summary: DISCORD_WEBHOOK_URL not set")
        print(embed)
        return
    resp = requests.post(DISCORD_WEBHOOK_URL, json={"embeds": [embed]}, timeout=15)
    if resp.status_code >= 300:
        print("summary: discord send failed", resp.status_code, resp.text)
    else:
        print("summary: discord sent")


def summarize(trades):
    closed = [t for t in trades if t.get("status") == "closed" and t.get("result") in ("WIN", "LOSS")]
    wins = [t for t in closed if t["result"] == "WIN"]
    losses = [t for t in closed if t["result"] == "LOSS"]
    timeouts = [t for t in trades if t.get("result") == "TIMEOUT"]
    open_trades = [t for t in trades if t.get("status") == "open"]

    total_closed = len(closed)
    win_rate = round(len(wins) / total_closed * 100, 1) if total_closed > 0 else None

    by_combo = {}
    for t in closed:
        combo = t["asset"] + " " + t["timeframe"]
        if combo not in by_combo:
            by_combo[combo] = {"wins": 0, "losses": 0}
        if t["result"] == "WIN":
            by_combo[combo]["wins"] = by_combo[combo]["wins"] + 1
        else:
            by_combo[combo]["losses"] = by_combo[combo]["losses"] + 1

    lines = ""
    for combo in by_combo:
        stat = by_combo[combo]
        c_total = stat["wins"] + stat["losses"]
        c_rate = round(stat["wins"] / c_total * 100, 1) if c_total > 0 else 0
        lines = lines + combo + ": " + str(stat["wins"]) + "W " + str(stat["losses"]) + "L (" + str(c_rate) + "%)\n"
    if lines == "":
        lines = "no closed trades yet"

    win_rate_txt = (str(win_rate) + "%") if win_rate is not None else "n/a"

    return {
        "title": "Daily performance summary",
        "color": 3447003,
        "fields": [
            {"name": "total signals", "value": str(len(trades)), "inline": True},
            {"name": "closed trades", "value": str(total_closed), "inline": True},
            {"name": "open trades", "value": str(len(open_trades)), "inline": True},
            {"name": "overall win rate", "value": win_rate_txt, "inline": True},
            {"name": "wins / losses", "value": str(len(wins)) + " / " + str(len(losses)), "inline": True},
            {"name": "timeouts", "value": str(len(timeouts)), "inline": True},
            {"name": "by asset+timeframe", "value": lines, "inline": False},
        ],
    }


def main():
    trades = load_trades()
    embed = summarize(trades)
    send_discord(embed)


if __name__ == "__main__":
    main()
