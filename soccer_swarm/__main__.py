import argparse
import json
import logging
import os
import sys

from soccer_swarm.config import API_KEY, DB_PATH, LEAGUES, CURRENT_SEASON


def setup_logging(verbose: bool = False, quiet: bool = False):
    level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_fetch(args):
    if not API_KEY:
        print("Error: API_FOOTBALL_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    from soccer_swarm.data.db import get_connection, create_tables
    from soccer_swarm.data.client import ApiClient

    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = get_connection(DB_PATH)
    create_tables(conn)
    client = ApiClient(api_key=API_KEY, conn=conn)

    leagues = args.leagues.split(",") if args.leagues else list(LEAGUES.keys())
    season = args.season or CURRENT_SEASON

    for league_name in leagues:
        league_id = LEAGUES.get(league_name)
        if not league_id:
            print(f"Unknown league: {league_name}", file=sys.stderr)
            continue
        print(f"Fetching {league_name} (id={league_id}) season {season}...")
        try:
            data = client.get("fixtures", {"league": league_id, "season": season})
            fixtures = data.get("response", [])
            print(f"  Got {len(fixtures)} fixtures")
            for fx in fixtures:
                fixture_data = fx.get("fixture", {})
                teams = fx.get("teams", {})
                goals = fx.get("goals", {})
                status = fixture_data.get("status", {}).get("short", "NS")

                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO leagues VALUES (?, ?, ?, ?, ?)",
                        (league_id, league_name, "", season, league_id),
                    )
                    for side in ("home", "away"):
                        team = teams.get(side, {})
                        conn.execute(
                            "INSERT OR IGNORE INTO teams VALUES (?, ?, ?, ?)",
                            (team.get("id"), team.get("name", ""), league_id, team.get("id")),
                        )

                    conn.execute(
                        "INSERT OR IGNORE INTO fixtures (league_id, home_team_id, away_team_id, date, status, home_goals, away_goals, api_id) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            league_id,
                            teams.get("home", {}).get("id"),
                            teams.get("away", {}).get("id"),
                            fixture_data.get("date", ""),
                            status,
                            goals.get("home"),
                            goals.get("away"),
                            fixture_data.get("id"),
                        ),
                    )
                except Exception as e:
                    logging.warning("Skipping fixture %s: %s", fixture_data.get("id"), e)

            conn.commit()
        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)

    conn.close()
    print("Fetch complete.")


def cmd_train(args):
    from soccer_swarm.data.db import get_connection
    from soccer_swarm.agents.poisson import PoissonAgent
    from soccer_swarm.agents.elo import EloAgent
    from soccer_swarm.agents.xgboost_agent import XGBoostAgent

    conn = get_connection(DB_PATH)
    fixtures = [dict(row) for row in conn.execute("SELECT * FROM fixtures WHERE status IN ('FT','AET','PEN')").fetchall()]
    standings_rows = conn.execute("SELECT * FROM standings").fetchall()
    standings = {}
    for row in standings_rows:
        standings[row["team_id"]] = dict(row)
    conn.close()

    print(f"Training on {len(fixtures)} fixtures...")
    os.makedirs("models", exist_ok=True)

    poisson = PoissonAgent()
    poisson.train(fixtures, [])
    poisson.save("models/poisson.json")
    print("  Poisson agent trained")

    elo = EloAgent()
    elo.train(fixtures, [])
    elo.save("models/elo.json")
    print("  ELO agent trained")

    xgb_agent = XGBoostAgent()
    xgb_agent.train(fixtures, [], standings=standings)
    xgb_agent.save("models")
    print("  XGBoost agent trained")

    print("Training complete. Models saved to models/")


def cmd_optimize(args):
    print("Running MOPSO optimization...")
    from soccer_swarm.data.db import get_connection
    from soccer_swarm.agents.poisson import PoissonAgent
    from soccer_swarm.agents.elo import EloAgent
    from soccer_swarm.agents.xgboost_agent import XGBoostAgent
    from soccer_swarm.agents.odds_implied import OddsImpliedAgent
    from soccer_swarm.backtest.engine import BacktestEngine

    conn = get_connection(DB_PATH)
    fixtures = [dict(row) for row in conn.execute("SELECT * FROM fixtures WHERE status IN ('FT','AET','PEN')").fetchall()]
    standings_rows = conn.execute("SELECT * FROM standings").fetchall()
    standings = {row["team_id"]: dict(row) for row in standings_rows}
    conn.close()

    agents = [PoissonAgent(), EloAgent(), XGBoostAgent(), OddsImpliedAgent()]
    engine = BacktestEngine(agents=agents, mopso_pop=args.pop_size, mopso_gen=args.n_gen)
    results = engine.run(fixtures, standings)

    os.makedirs("results", exist_ok=True)
    with open("results/optimize_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Optimization complete. Results: {json.dumps(results['global'], indent=2)}")


def cmd_backtest(args):
    print("Running backtest...")
    cmd_optimize(args)


def cmd_predict(args):
    from soccer_swarm.data.db import get_connection
    from soccer_swarm.agents.poisson import PoissonAgent
    from soccer_swarm.agents.elo import EloAgent
    from soccer_swarm.agents.xgboost_agent import XGBoostAgent
    from soccer_swarm.agents.odds_implied import OddsImpliedAgent
    from soccer_swarm.output.formatter import format_predictions_json

    import numpy as np

    conn = get_connection(DB_PATH)
    upcoming = [dict(row) for row in conn.execute(
        "SELECT * FROM fixtures WHERE status = 'NS' AND date >= ? ORDER BY date",
        (args.date or "",),
    ).fetchall()]
    history = [dict(row) for row in conn.execute("SELECT * FROM fixtures WHERE status IN ('FT','AET','PEN')").fetchall()]
    standings_rows = conn.execute("SELECT * FROM standings").fetchall()
    standings = {row["team_id"]: dict(row) for row in standings_rows}
    conn.close()

    if not upcoming:
        print("No upcoming fixtures found.")
        return

    agents = []
    poisson = PoissonAgent()
    poisson.load("models/poisson.json")
    agents.append(poisson)

    elo = EloAgent()
    elo.load("models/elo.json")
    agents.append(elo)

    xgb_agent = XGBoostAgent()
    xgb_agent.load("models")
    agents.append(xgb_agent)

    agents.append(OddsImpliedAgent())

    try:
        from soccer_swarm.optimizer.mopso import load_pareto, select_from_pareto
        X, F = load_pareto("models/pareto_front.npz")
        idx = select_from_pareto(F, X, profile=args.profile)
        best_x = X[idx]
        weights_raw = best_x[:len(agents)]
        weights = weights_raw / weights_raw.sum()
    except FileNotFoundError:
        weights = np.array([0.25, 0.25, 0.25, 0.25])

    predictions = []
    for f in upcoming:
        preds = []
        for agent in agents:
            try:
                p = agent.predict(f)
            except TypeError:
                p = agent.predict(f, history=history, standings=standings)
            preds.append(p)

        ensemble = np.zeros(7)
        w_total = 0
        for i, p in enumerate(preds):
            if p is not None:
                ensemble += weights[i] * p.as_array()
                w_total += weights[i]
        if w_total > 0:
            ensemble /= w_total

        predictions.append({
            "fixture_id": f["id"],
            "date": f["date"],
            "home_team": str(f["home_team_id"]),
            "away_team": str(f["away_team_id"]),
            "match_1x2": tuple(ensemble[0:3]),
            "over_under_25": tuple(ensemble[3:5]),
            "btts": tuple(ensemble[5:7]),
        })

    output = format_predictions_json(predictions)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Predictions written to {args.output}")
    else:
        print(output)


def cmd_fetch_csv(args):
    """Fetch data from football-data.co.uk (free, no API key needed)."""
    from soccer_swarm.data.db import get_connection, create_tables
    from soccer_swarm.data.csv_import import import_season, import_upcoming, generate_remaining_fixtures

    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = get_connection(DB_PATH)
    create_tables(conn)

    total = 0

    if args.season:
        print(f"Importing season {args.season} from football-data.co.uk...")
        total += import_season(conn, args.season)

    if args.upcoming:
        print("Importing upcoming fixtures from football-data.co.uk...")
        total += import_upcoming(conn)

    if args.generate_remaining:
        print("Generating remaining season fixtures...")
        total += generate_remaining_fixtures(conn)

    if not args.season and not args.upcoming and not args.generate_remaining:
        # Default: import current season + generate remaining
        print("Importing season 2526 from football-data.co.uk...")
        total += import_season(conn, "2526")
        print("Generating remaining season fixtures...")
        total += generate_remaining_fixtures(conn)

    conn.close()
    print(f"Import complete. {total} new fixtures added.")


def cmd_pareto(args):
    from soccer_swarm.optimizer.mopso import load_pareto, select_from_pareto
    import numpy as np

    X, F = load_pareto("models/pareto_front.npz")
    print(f"Pareto front: {len(F)} solutions")
    print(f"Objectives: log_loss | -ROI | max_drawdown")
    for i, (x, f) in enumerate(zip(X, F)):
        print(f"  [{i}] log_loss={f[0]:.4f} ROI={-f[1]:.4f} drawdown={f[2]:.4f}")

    if args.profile:
        idx = select_from_pareto(F, X, profile=args.profile)
        print(f"\nSelected ({args.profile}): solution [{idx}]")
        print(f"  Weights: {X[idx][:4] / X[idx][:4].sum()}")
        print(f"  Thresholds: 1x2={X[idx][4]:.3f} ou={X[idx][5]:.3f} btts={X[idx][6]:.3f}")


def main():
    parser = argparse.ArgumentParser(prog="soccer_swarm", description="MOPSO Soccer Prediction System")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # fetch
    p_fetch = subparsers.add_parser("fetch", help="Fetch data from API-Football")
    p_fetch.add_argument("--leagues", type=str, default=None, help="Comma-separated league names")
    p_fetch.add_argument("--season", type=int, default=None)

    # train
    subparsers.add_parser("train", help="Train prediction agents")

    # optimize
    p_opt = subparsers.add_parser("optimize", help="Run MOPSO optimization")
    p_opt.add_argument("--pop-size", type=int, default=100)
    p_opt.add_argument("--n-gen", type=int, default=300)

    # backtest
    p_bt = subparsers.add_parser("backtest", help="Run walk-forward backtest")
    p_bt.add_argument("--output", type=str, default=None)
    p_bt.add_argument("--pop-size", type=int, default=100)
    p_bt.add_argument("--n-gen", type=int, default=300)

    # predict
    p_pred = subparsers.add_parser("predict", help="Generate predictions")
    p_pred.add_argument("--date", type=str, default=None)
    p_pred.add_argument("--output", type=str, default=None)
    p_pred.add_argument("--profile", type=str, default="balanced")

    # fetch-csv
    p_csv = subparsers.add_parser("fetch-csv", help="Fetch data from football-data.co.uk (free)")
    p_csv.add_argument("--season", type=str, default=None, help="Season code e.g. 2526 for 2025/26")
    p_csv.add_argument("--upcoming", action="store_true", help="Fetch upcoming fixtures")
    p_csv.add_argument("--generate-remaining", action="store_true", help="Generate remaining season fixtures")

    # pareto
    p_pareto = subparsers.add_parser("pareto", help="Inspect Pareto front")
    p_pareto.add_argument("--profile", type=str, default="balanced")
    p_pareto.add_argument("--weights", type=str, default=None)

    args = parser.parse_args()
    setup_logging(args.verbose, args.quiet)

    commands = {
        "fetch": cmd_fetch,
        "fetch-csv": cmd_fetch_csv,
        "train": cmd_train,
        "optimize": cmd_optimize,
        "backtest": cmd_backtest,
        "predict": cmd_predict,
        "pareto": cmd_pareto,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
