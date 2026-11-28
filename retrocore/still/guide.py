"""Still App の説明の文（ブラウザ版の「？」と CLI の --help で共通）。

項目（`convert.ITEMS`）と初めの設定（`convert.MODES`）ごとに、3 行の決まった形で書く:

- `lead`: 「オンにすると」（印を付ける項目）・「選ぶと」（選ぶ項目と初めの設定）・「入れると」（数を入れる項目）
- `care`: 「気を付けること」
- `suits`: 「向いている絵」

ブラウザ版は `convert.spec()` の `guide` から読み（tools/build_still_web.py が埋め込む）、CLI は `cli_help()` で
`lead` の文を --help に使う。出力（PNG・JSON）には関わらない。
"""

from __future__ import annotations

from .colors import MAX_COLORS, MIN_COLORS

# 1 行目の見出し（項目の種類ごと）
LEAD_LABELS = {"check": "オンにすると", "select": "選ぶと", "value": "入れると"}
LINE_LABELS = {"care": "気を付けること", "suits": "向いている絵"}

# 初めの設定（--mode）。1 行目の見出しは「選ぶと」
MODES = {
    "pc98": {
        "lead": "16 色・4096 色に丸める・640×400 までの、98 風のドット絵にそろえる",
        "care": "結果は元の絵の作風で決まる（98 らしさを足すのではない）",
        "suits": "AI に 98 風で描かせた絵。写真の 98 風は、写真の下書き＋16 色＋4096 色に丸める",
    },
    "modern": {
        "lead": "32 色・丸めない・1920×1080 まで・似た色をまとめる。今のドット絵の手直しの出発点に",
        "care": "元が 98 風の絵だと、98 風とほとんど変わらない",
        "suits": "AI に今風で描かせた絵",
    },
    "photo": {
        "lead": "向きを直し、横 320 マス・色の塊をまとめる・孤立したドットをそろえるで、塗り直しの下絵にする",
        "care": "幅 640 くらいだとドット絵らしく、1920 だと実写に近い（試しの機能）",
        "suits": "自分で撮った写真（部屋・建物など）",
    },
}

# 項目（工程の順。convert.ITEMS と同じ並び）。kind は 1 行目の見出しの種類、cli は CLI のオプション
ITEMS = {
    "orient": {
        "kind": "check", "cli": "--orient・--no-orient",
        "lead": "写真に記録された向き（EXIF）を見て、正しい向きに回してから読む",
        "care": "向きの記録が無い画像では何も変わらない",
        "suits": "スマホやカメラで撮った写真（写真の下書きでは初めから入）",
    },
    "find_grid": {
        "kind": "select", "cli": "--grid・--width",
        "lead": "「マス目を探す」は AI のドットのマス目を自動で見つける。「横のマス数で分ける」は横を決まった数に分ける",
        "care": "細かいドットの絵ではマス目が見つからないことがある（そのときは横のマス数で分ける）",
        "suits": "マス目を探す＝マス目がはっきりした AI のドット絵。横のマス数＝写真・見つからない絵",
    },
    "width": {
        "kind": "value", "cli": "--width",
        "lead": "マス目を探さずに、横をこの数のマスに分ける（縦は縦横の比から）",
        "care": "「マス目を探す」のときは使わない。写真は 640 くらいでドット絵らしく、1920 だと実写に近い",
        "suits": "写真の下書き（初めは 320）、マス目が見つからない AI の絵",
    },
    "aspect": {
        "kind": "select", "cli": "--aspect",
        "lead": "拡大・縮小はせず、切り抜くだけでこの比率にする",
        "care": "切り落とした所は戻らない。マス目を探したときはマス単位で切るので、ちょうどの比にならないことがある",
        "suits": "16:9 は Godot の背景、16:10 は 98 の 640×400、数値で見出し画像（1200:630）なども",
    },
    "anchor": {
        "kind": "select", "cli": "--anchor",
        "lead": "切り抜くときにどこを残すか（横・縦それぞれ 0〜100 の%。50・50 は中央）",
        "care": "比率が「元のまま」のときは使わない",
        "suits": "上を残したい写真は縦 0〜20、足元を残したい絵は縦 100",
    },
    "max_size": {
        "kind": "value", "cli": "--max-size",
        "lead": "大きさ（ドット数）の上限。超えたら出力せずに、収まる横のマス数の候補を出す",
        "care": "縮めて収めることはしない（候補のボタンで選び直す）",
        "suits": "98 の 640×400 や、ゲームの画面の大きさに収めたいとき",
    },
    "smooth": {
        "kind": "select", "cli": "--smooth",
        "lead": "境目は残したまま、似た色の近所をならして色の塊にし、写真のムラを平らにする（強いほど塊が大きい）",
        "care": "AI の絵では 1 ドットの細い線が途切れることがある",
        "suits": "写真（写真の下書きでは初めから「中」）",
    },
    "colors": {
        "kind": "value", "cli": "--colors",
        "lead": f"パレットの色数の上限（{MIN_COLORS}〜{MAX_COLORS}）",
        "care": "上限なので、似た色をまとめたり 4096 色に丸めたりすると少なくなることがある",
        "suits": "98 風は 16、今風・写真は 32 から。細かい色の絵は 48〜128",
    },
    "four_bit": {
        "kind": "check", "cli": "--4bit・--no-4bit",
        "lead": "色を RGB 各 16 段（4096 色）の中から選び、98 らしい色にする",
        "care": "色みが少しずれ、重なって色数が減ることがある",
        "suits": "98 風にしたい絵・写真（98 風では初めから入）",
    },
    "keep_darkest": {
        "kind": "check", "cli": "--keep-darkest・--no-keep-darkest",
        "lead": "パレットの 1 色をいちばん暗い色にし、暗い線が中間の色に吸われて消えないようにする",
        "care": "1 色をその色に使うので、ほかに使える色が 1 つ減る",
        "suits": "黒っぽい輪郭や影のある絵（どの初めの設定でも入）",
    },
    "merge_similar": {
        "kind": "check", "cli": "--merge-similar・--no-merge-similar",
        "lead": "見た目でほぼ同じ色どうしをまとめ、AI のにじみの色を減らす",
        "care": "写真や色数が少ないときは、空などに要る段まで消えることがある",
        "suits": "AI に今風で描かせた絵、色数を多くしたとき（今風では初めから入）",
    },
    "dither": {
        "kind": "select", "cli": "--dither",
        "lead": "なめらかな所（空など）を、近い 2 色の決まった網目でつなぐ（新しい色は作らない）",
        "care": "平らな壁にも少し入ることがある。手直しはしにくくなる",
        "suits": "空などのある写真。2×2 は幅 640 くらいで 98 らしく、4×4 は 1280 以上でなめらか",
    },
    "tidy_dots": {
        "kind": "check", "cli": "--tidy-dots・--no-tidy-dots",
        "lead": "まわりと色の違う、散らばった 1 ドットの点を、まわりの色にそろえて減らす",
        "care": "わざと置いた 1 ドットの光が消えることがある",
        "suits": "にじみの点が多い AI の絵・写真（写真の下書きでは初めから入）",
    },
    "tidy_pattern": {
        "kind": "check", "cli": "--tidy-pattern",
        "lead": "AI の崩れたディザ（ばらばらの点）を、2 色の規則正しい網目に整える",
        "care": "写真ではほとんど効かない",
        "suits": "AI がディザ風に描いた絵",
    },
    "outline": {
        "kind": "check", "cli": "--outline",
        "lead": "明るさの差が大きい色の境目に、いちばん暗い色で 1 ドットの線を足す",
        "care": "線がガタつきやすい",
        "suits": "境目をはっきりさせたい写真",
    },
    "canvas": {
        "kind": "value", "cli": "--canvas",
        "lead": "絵を真ん中に置き、いちばん暗い色の余白を足してこの大きさにする",
        "care": "絵より小さい大きさは入れられない（そのときは止める）",
        "suits": "切らずに決まった大きさ（640×400 など）にそろえたいとき",
    },
}


def lines(entry: dict, lead_label: str) -> list[list[str]]:
    """3 行（[見出し, 文] の組）。"""
    return [[lead_label, entry["lead"]], [LINE_LABELS["care"], entry["care"]], [LINE_LABELS["suits"], entry["suits"]]]


def spec() -> dict:
    """ブラウザ版に埋め込む説明（項目・初めの設定ごとの 3 行と CLI のオプション）。"""
    return {"modes": {k: {"lines": lines(v, LEAD_LABELS["select"])} for k, v in MODES.items()},
            "mode_order": list(MODES),   # 埋め込む JSON はキーを並べ替えるので、並びは別に持つ
            "items": {k: {"lines": lines(v, LEAD_LABELS[v["kind"]]), "cli": v["cli"]} for k, v in ITEMS.items()}}


def cli_help(item: str, note: str = "") -> str:
    """CLI の --help の説明（1 行目の文のあとに、CLI だけの補足 note を足す）。"""
    text = ITEMS[item]["lead"]
    return (f"{text}。{note}" if note else text).replace("%", "%%")   # argparse は % を書式に使う


def mode_help(mode: str) -> str:
    """CLI の --mode の説明に使う、初めの設定の 1 行目の文。"""
    return MODES[mode]["lead"].replace("%", "%%")
