import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EYAK_PATH = os.path.join(HERE, "eyakeunyo.csv")
PILL_PATH = os.path.join(HERE, "pill_info.csv")
OUT_PATH = os.path.join(HERE, "master.parquet")

# e약은요에서 pill_info와 겹치는 식별 컬럼은 pill_info 쪽을 사용
EYAK_DROP = ["itemSeq", "itemName", "entpName", "itemImage", "bizrno"]

EYAK_RENAME = {
    "efcyQesitm": "efficacy",
    "useMethodQesitm": "use_method",
    "atpnWarnQesitm": "warning",
    "atpnQesitm": "precaution",
    "intrcQesitm": "interaction",
    "seQesitm": "side_effect",
    "depositMethodQesitm": "storage",
    "openDe": "open_date",
    "updateDe": "update_date",
}


def build_master(eyak_path=EYAK_PATH, pill_path=PILL_PATH, out_path=OUT_PATH):
    eyak = pd.read_csv(eyak_path, dtype=str)
    pill = pd.read_csv(pill_path, dtype=str)

    # e약은요는 itemImage만 다른 중복 행이 있어 첫 행만 유지.
    # pill_info 중복은 함량별로 모양/각인/이미지가 다른 별개 낱알이라 그대로 유지.
    eyak = eyak.drop_duplicates("itemSeq", keep="first")

    master = pill.merge(
        eyak.rename(columns={"itemSeq": "ITEM_SEQ"}).drop(
            columns=[c for c in EYAK_DROP if c != "itemSeq"]
        ),
        on="ITEM_SEQ",
        how="inner",
    )
    n_joined = len(master)

    # 일반의약품만 (일반의약품 + 일반(안전상비))
    master = master[master["ETC_OTC_NAME"].str.startswith("일반", na=False)]
    master = master.reset_index(drop=True)
    master = master.rename(columns=EYAK_RENAME)
    master.columns = [c.lower() for c in master.columns]

    print(f"e약은요 unique ITEM_SEQ : {eyak['itemSeq'].nunique()}")
    print(f"pill_info unique ITEM_SEQ: {pill['ITEM_SEQ'].nunique()}")
    print(f"inner join rows         : {n_joined}")
    print(f"일반의약품 filter rows   : {len(master)} (unique ITEM_SEQ {master['item_seq'].nunique()})")

    master.to_parquet(out_path, index=False)
    print(f"saved -> {out_path}")
    return master


if __name__ == "__main__":
    build_master(*sys.argv[1:4])
