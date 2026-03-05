import requests
import pandas as pd
import time

BASE_URL = "https://api.opendota.com/api"

TARGET_MATCHES = 1000
MIN_DURATION = 1200
REQUEST_DELAY = 1.5

OUTPUT_FILE = "dota_ml_current_patch.csv"


def safe_get(url, max_retries=5):
    for _ in range(max_retries):
        try:
            r = requests.get(url, timeout=15)

            if r.status_code == 200:
                return r.json()

            if r.status_code == 429:
                print("Rate limit... ждем 15 сек")
                time.sleep(15)
                continue

            print("Ошибка:", r.status_code)
            return None

        except requests.exceptions.RequestException:
            print("Ошибка соединения, повтор...")
            time.sleep(5)

    print("Превышено число попыток.")
    return None


def get_current_patch():
    patches = safe_get(f"{BASE_URL}/constants/patch")
    return patches[-1]["id"]


def assign_positions(players, prefix):
    sorted_players = sorted(
        players,
        key=lambda x: x.get("gold_per_min") or 0,
        reverse=True
    )

    return {
        f"{prefix}_pos1": sorted_players[0]["hero_id"],
        f"{prefix}_pos2": sorted_players[1]["hero_id"],
        f"{prefix}_pos3": sorted_players[2]["hero_id"],
        f"{prefix}_pos4": sorted_players[3]["hero_id"],
        f"{prefix}_pos5": sorted_players[4]["hero_id"],
    }


def collect_dataset():
    print("Определяем текущий патч...")
    CURRENT_PATCH = get_current_patch()
    print("Текущий патч:", CURRENT_PATCH)

    dataset = []
    seen_ids = set()

    pro_matches = safe_get(f"{BASE_URL}/proMatches")
    if not pro_matches:
        print("Не удалось получить список pro матчей")
        return

    for match in pro_matches:

        if len(dataset) >= TARGET_MATCHES:
            break

        match_id = match["match_id"]

        if match_id in seen_ids:
            continue

        details = safe_get(f"{BASE_URL}/matches/{match_id}")
        if not details:
            continue

        if details.get("patch") != CURRENT_PATCH:
            continue

        if details.get("duration", 0) < MIN_DURATION:
            continue

        players = details.get("players")
        if not players or len(players) != 10:
            continue

        radiant = [p for p in players if p.get("isRadiant")]
        dire = [p for p in players if not p.get("isRadiant")]

        if len(radiant) != 5 or len(dire) != 5:
            continue

        row = {
            "match_id": match_id,
            "radiant_win": int(details.get("radiant_win", 0))
        }

        row.update(assign_positions(radiant, "r"))
        row.update(assign_positions(dire, "d"))

        dataset.append(row)
        seen_ids.add(match_id)

        print(f"Собрано: {len(dataset)}")

        time.sleep(REQUEST_DELAY)

    pd.DataFrame(dataset).to_csv(OUTPUT_FILE, index=False)
    print("\nГотово! Файл сохранен:", OUTPUT_FILE)


if __name__ == "__main__":
    collect_dataset()