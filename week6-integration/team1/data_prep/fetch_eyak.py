import csv
import os
import sys
import time
from urllib.parse import unquote

import requests
from dotenv import load_dotenv

load_dotenv()

URL = "https://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList"
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eyakeunyo.csv")
NUM_OF_ROWS = 100

FIELDS = [
    "itemSeq",
    "itemName",
    "entpName",
    "efcyQesitm",
    "useMethodQesitm",
    "atpnWarnQesitm",
    "atpnQesitm",
    "intrcQesitm",
    "seQesitm",
    "depositMethodQesitm",
    "openDe",
    "updateDe",
    "itemImage",
    "bizrno",
]


def get_page(session, key, page):
    params = {"serviceKey": key, "pageNo": page, "numOfRows": NUM_OF_ROWS, "type": "json"}
    for attempt in range(5):
        try:
            res = session.get(URL, params=params, timeout=30)
            res.raise_for_status()
            data = res.json()
            body = data.get("body") or data.get("response", {}).get("body")
            if body is None:
                raise RuntimeError(f"unexpected response: {str(data)[:300]}")
            return body
        except Exception as e:
            if attempt == 4:
                raise
            print(f"page {page} retry {attempt + 1}: {e}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))


def main():
    key = os.getenv("DATA_GO_KR_API_KEY")
    if not key:
        sys.exit("DATA_GO_KR_API_KEY 가 .env 에 없습니다.")
    key = unquote(key)  # Encoding 키를 넣어도 이중 인코딩되지 않게 디코딩

    session = requests.Session()
    first = get_page(session, key, 1)
    total = int(first["totalCount"])
    pages = -(-total // NUM_OF_ROWS)
    print(f"totalCount={total}, pages={pages}")

    rows = list(first.get("items") or [])
    for page in range(2, pages + 1):
        body = get_page(session, key, page)
        rows.extend(body.get("items") or [])
        if page % 10 == 0 or page == pages:
            print(f"{page}/{pages} pages, {len(rows)} rows")

    with open(OUT_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {len(rows)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
