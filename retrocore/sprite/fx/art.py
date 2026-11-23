"""重ね（漫画的エフェクト）の絵と動き。ここだけを直せば、入れておく絵を描き直せる（画面と CLI で同じものを使う）。

絵は「部品」（文字で書いたドットの表）で描き、動きは「コマ」（どの部品を、どこに置くか・何ミリ秒）で書く。

- 部品の表の 1 文字が 1 ドット。`.` は透明、`o` は輪郭（キャラのいちばん暗い輪郭の色に合わせる）、ほかの文字は `COLORS` の色。
  行の長さはそろえる（そろっていなければ読み込むときにエラー）
- コマは `(置く部品の一覧, ミリ秒)`。置く部品は `(部品の名前, x, y)`（x・y はドット。右と下が +）。部品の一覧が空のコマは
  何も見えない（点滅の「消える」コマ）
- エフェクトの 1 コマの大きさは、全部のコマの部品が入るいちばん小さな四角（自動で決まる）。動きは、その四角の中で部品を
  ずらす・切り替える・消すだけで作る（拡大・縮小はしない。ドットがにじむため）
- `anchor` は置く場所: `"top"` = 頭の上（真ん中）、`"side"` = 頭の横（右。キャラを左右反転したら左）、
  `"face"` = 顔（右の目の下のあたり）、`"upper_face"` = 顔の上半分（真ん中）、`"around"` = 体のまわり
- `"around"` のコマでは、置く部品を `(部品の名前, 基準, x, y)` とも書ける。基準は絵全体の範囲（1 コマ目の不透明な所を囲む四角）の
  角・辺の真ん中（`nw` `n` `ne` `w` `e` `sw` `s` `se`）で、x・y はそこから部品の左上までのずれ（キャラの大きさで間が変わる）
- `loop` が False のものは 1 回で終わる（最後のコマで止めて残すか、消すかを選べる）
- 時間は 50 ミリ秒の倍数にしておく（.piskel は速さを 1 つしか持てないので、キャラと重ねた .piskel を作りやすいように）

新しいエフェクトを足すときは `EFFECTS` に 1 つ足すだけ（名前は英小文字と `_`。ソフトハウス名・作品名は使わない）。
"""

from __future__ import annotations

# 輪郭の文字（色はキャラのいちばん暗い輪郭の色。見つからなければ OUTLINE_FALLBACK）
OUTLINE = "o"
TRANSPARENT = "."
OUTLINE_FALLBACK = (32, 24, 40)

# 決まった色（エフェクトはキャラと別のスプライトなので、キャラのパレットに無い色でよい）
COLORS = {
    "Y": (255, 214, 64),    # 黄（！）
    "R": (232, 56, 80),     # 赤（ハート）
    "B": (96, 196, 240),    # 水色（汗）
    "W": (255, 255, 255),   # 白（光の点・Z・？・ぐるぐる・湯気）
    "P": (255, 128, 176),   # 桃（音符）
    "G": (160, 160, 176),   # 灰（電球の口金）
    "L": (200, 208, 224),   # 薄い灰（湯気の影）
    "N": (80, 96, 216),     # 青紫（青ざめの縦線。輪郭なし）
    "U": (128, 120, 160),   # くすんだ紫（もやもや）
    "K": (88, 80, 120),     # 暗い紫（もやもやの渦）
}

EFFECTS = {
    # ！（驚き）: 頭の上。下から出てきて 2 ドット跳ね、1 ドット戻って止まる。1 回で終わる
    "surprise": {
        "label": "！（驚き）",
        "anchor": "top",
        "loop": False,
        "parts": {
            "mark": """
                .oooo.
                oYYYYo
                oYYYYo
                oYYYYo
                .oYYo.
                .oYYo.
                ..oo..
                ......
                ..oo..
                .oYYo.
                .oYYo.
                ..oo..
            """,
        },
        "frames": [
            ([("mark", 0, 2)], 50),
            ([("mark", 0, -2)], 100),
            ([("mark", 0, 0)], 100),
            ([("mark", 0, -1)], 100),
            ([("mark", 0, 0)], 600),
        ],
    },
    # 汗（たらり）: 頭の横。上から 3 ドット下へ流れて止まる。1 回で終わる
    "sweat": {
        "label": "汗（たらり）",
        "anchor": "side",
        "loop": False,
        "parts": {
            "drop": """
                ...o...
                ..oBo..
                ..oBo..
                .oBBBo.
                oBBBBBo
                oWBBBBo
                oWBBBBo
                .oBBBo.
                ..ooo..
            """,
        },
        "frames": [
            ([("drop", 0, 0)], 200),
            ([("drop", 0, 1)], 150),
            ([("drop", 0, 2)], 150),
            ([("drop", 0, 3)], 600),
        ],
    },
    # ハート: 頭の上。左右に 1 ドットずつ揺れながら上へ浮かび、消えてまた下から。くり返し
    "heart": {
        "label": "ハート",
        "anchor": "top",
        "loop": True,
        "parts": {
            "heart": """
                .ooo.ooo.
                oRRRoRRRo
                oRWRRRRRo
                oRRRRRRRo
                .oRRRRRo.
                ..oRRRo..
                ...oRo...
                ....o....
            """,
        },
        "frames": [
            ([("heart", 1, 5)], 150),
            ([("heart", 2, 4)], 150),
            ([("heart", 2, 3)], 150),
            ([("heart", 1, 2)], 150),
            ([("heart", 0, 1)], 150),
            ([("heart", 0, 0)], 150),
            ([], 100),
        ],
    },
    # Zzz: 頭の横。小さな Z から 1 つずつ増えながら上へ。全部出たら消えて、また 1 つ目から。くり返し
    "sleep": {
        "label": "Zzz",
        "anchor": "side",
        "loop": True,
        "parts": {
            "z_small": """
                .oooo.
                oWWWWo
                .ooWWo
                oWWoo.
                oWWWWo
                .oooo.
            """,
            "z_mid": """
                .ooooo.
                oWWWWWo
                .ooWWo.
                .oWWoo.
                oWWWWWo
                .ooooo.
            """,
            "z_big": """
                .oooooo.
                oWWWWWWo
                .oooWWo.
                ..oWWo..
                .oWWooo.
                oWWWWWWo
                .oooooo.
            """,
        },
        "frames": [
            ([("z_small", 0, 12)], 350),
            ([("z_small", 0, 12), ("z_mid", 5, 6)], 350),
            ([("z_small", 0, 12), ("z_mid", 5, 6), ("z_big", 11, 0)], 500),
            ([], 300),
        ],
    },
    # ？（疑問）: 頭の上。左右に 1 ドットずつゆっくり揺れる。くり返し
    "question": {
        "label": "？（疑問）",
        "anchor": "top",
        "loop": True,
        "parts": {
            "mark": """
                ..oooo..
                .oWWWWo.
                oWWooWWo
                .oo.oWWo
                ...oWWo.
                ..oWWo..
                ..oWWo..
                ...oo...
                ........
                ...oo...
                ..oWWo..
                ..oWWo..
                ...oo...
            """,
        },
        "frames": [
            ([("mark", 1, 0)], 300),
            ([("mark", 2, 0)], 300),
            ([("mark", 1, 0)], 300),
            ([("mark", 0, 0)], 300),
        ],
    },
    # 怒りマーク: 頭の横。4 つの L 字（角が真ん中を向く）。小さい絵と大きい絵を交互にして、ドクドクと脈打つ。くり返し
    "anger": {
        "label": "怒りマーク",
        "anchor": "side",
        "loop": True,
        "parts": {
            "big": """
                .....o...o.....
                ....oRo.oRo....
                ....oRo.oRo....
                ....oRo.oRo....
                .ooooRo.oRoooo.
                oRRRRRo.oRRRRRo
                .ooooo...ooooo.
                ...............
                .ooooo...ooooo.
                oRRRRRo.oRRRRRo
                .ooooRo.oRoooo.
                ....oRo.oRo....
                ....oRo.oRo....
                ....oRo.oRo....
                .....o...o.....
            """,
            "small": """
                ...o...o...
                ..oRo.oRo..
                .ooRo.oRoo.
                oRRRo.oRRRo
                .ooo...ooo.
                ...........
                .ooo...ooo.
                oRRRo.oRRRo
                .ooRo.oRoo.
                ..oRo.oRo..
                ...o...o...
            """,
        },
        "frames": [
            ([("small", 2, 2)], 250),
            ([("big", 0, 0)], 150),
            ([("small", 2, 2)], 150),
            ([("big", 0, 0)], 350),
        ],
    },
    # 音符: 頭の横。左右に揺れながら上へ浮かび、消えてまた下から。くり返し
    "music": {
        "label": "音符",
        "anchor": "side",
        "loop": True,
        "parts": {
            "note": """
                ...oo...
                ..oPPo..
                ..oPPPo.
                ..oPoPPo
                ..oPooPo
                ..oPo.o.
                .ooPo...
                oPPPo...
                oPPPPo..
                oPPPPo..
                .oPPo...
                ..oo....
            """,
        },
        "frames": [
            ([("note", 0, 5)], 150),
            ([("note", 1, 4)], 150),
            ([("note", 2, 3)], 150),
            ([("note", 2, 2)], 150),
            ([("note", 1, 1)], 150),
            ([("note", 0, 0)], 150),
            ([], 100),
        ],
    },
    # キラキラ: 体のまわり（絵全体の四角の角と辺）。2 か所ずつ、大きい星・小さい星・消えるをくり返して、ちかちか光る。くり返し
    "sparkle": {
        "label": "キラキラ",
        "anchor": "around",
        "loop": True,
        "parts": {
            "big": """
                ....o....
                ...oYo...
                ...oYo...
                .ooYWYoo.
                oYYWWWYYo
                .ooYWYoo.
                ...oYo...
                ...oYo...
                ....o....
            """,
            "small": """
                ..o..
                .oYo.
                oYWYo
                .oYo.
                ..o..
            """,
        },
        "frames": [
            ([("big", "ne", -5, -4), ("small", "sw", -4, -9)], 200),
            ([("small", "ne", -3, -2), ("big", "w", -9, -10)], 200),
            ([("small", "w", -7, -8), ("big", "e", 0, 2)], 200),
            ([("big", "sw", -6, -11), ("small", "e", 2, 4)], 200),
        ],
    },
    # 涙: 顔（右の目の下のあたり）。目のふちにたまった涙から、しずくが下へ落ちる。くり返し
    "tears": {
        "label": "涙",
        "anchor": "face",
        "loop": True,
        "parts": {
            "pool": """
                .ooo.
                oBBBo
                .ooo.
            """,
            "drop": """
                ..o..
                .oBo.
                oBWBo
                oBBBo
                .oBo.
                ..o..
            """,
        },
        "frames": [
            ([("pool", 0, 0), ("drop", 0, 1)], 150),
            ([("pool", 0, 0), ("drop", 0, 3)], 150),
            ([("pool", 0, 0), ("drop", 0, 5)], 150),
            ([("pool", 0, 0), ("drop", 0, 7)], 150),
            ([("pool", 0, 0)], 200),
        ],
    },
    # 青ざめの縦線: 顔の上半分（目の上）。顔に重なる縦線（細い線なので輪郭なし）。下の 2 行がゆっくり消えて戻る。くり返し
    "pale": {
        "label": "青ざめ（縦線）",
        "anchor": "upper_face",
        "loop": True,
        "parts": {
            "lines": """
                N.N.N.N.N
                N.N.N.N.N
                N.N.N.N.N
                N.N.N.N.N
                N.N.N.N.N
                ..N.N.N..
                ..N.N.N..
                ....N....
            """,
            "short": """
                N.N.N.N.N
                N.N.N.N.N
                N.N.N.N.N
                N.N.N.N.N
                ..N.N.N..
                ....N....
                .........
                .........
            """,
        },
        "frames": [
            ([("lines", 0, 0)], 900),
            ([("short", 0, 0)], 300),
        ],
    },
    # 電球（ひらめき）: 頭の上。下からポンと出て、光の線が 2 回点滅し、光ったまま止まる。1 回で終わる
    "idea": {
        "label": "電球（ひらめき）",
        "anchor": "top",
        "loop": False,
        "parts": {
            "bulb": """
                ...ooo...
                ..oYYYo..
                .oYWYYYo.
                oYWYYYYYo
                oYWYYYYYo
                oYYYYYYYo
                .oYYYYYo.
                ..oYYYo..
                ..oGGGo..
                ..oGGGo..
                ...oGo...
                ....o....
            """,
            "ray_up": """
                .o.
                oYo
                oYo
                .o.
            """,
            "ray_side": """
                .oo.
                oYYo
                .oo.
            """,
            "ray_nw": """
                .o..
                oYo.
                .oYo
                ..o.
            """,
            "ray_ne": """
                ..o.
                .oYo
                oYo.
                .o..
            """,
        },
        "frames": [
            ([("bulb", 5, 8)], 50),
            ([("bulb", 5, 4)], 100),
            ([("bulb", 5, 5)], 100),
            ([("bulb", 5, 5), ("ray_up", 8, 0), ("ray_nw", 2, 2), ("ray_ne", 13, 2), ("ray_side", 0, 8),
              ("ray_side", 15, 8)], 200),
            ([("bulb", 5, 5)], 150),
            ([("bulb", 5, 5), ("ray_up", 8, 0), ("ray_nw", 2, 2), ("ray_ne", 13, 2), ("ray_side", 0, 8),
              ("ray_side", 15, 8)], 600),
        ],
    },
    # ぐるぐる（混乱）: 頭の上。うずまきを 90 度ずつ回す（4 枚を順に）。くり返し
    "dizzy": {
        "label": "ぐるぐる（混乱）",
        "anchor": "top",
        "loop": True,
        "parts": {
            "a": """
                ...ooooo...
                ..oWWWWWo..
                .oWoooooWo.
                oWooWWWooWo
                oWoWoooWoWo
                oWoWoWWWoWo
                oWoWoooooWo
                oWooWWWWWo.
                .oWoooooo..
                ..oWWWWo...
                ...oooo....
            """,
            "b": """
                ...ooooo...
                ..oWWWWWo..
                .oWoooooWo.
                oWooWWWooWo
                oWoWoooWoWo
                oWoWoWoWoWo
                oWoWoWoWoWo
                .ooWoWWooWo
                ..oWooooWo.
                ...oWWWWo..
                ....oooo...
            """,
            "c": """
                ....oooo...
                ...oWWWWo..
                ..ooooooWo.
                .oWWWWWooWo
                oWoooooWoWo
                oWoWWWoWoWo
                oWoWoooWoWo
                oWooWWWooWo
                .oWoooooWo.
                ..oWWWWWo..
                ...ooooo...
            """,
            "d": """
                ...oooo....
                ..oWWWWo...
                .oWooooWo..
                oWooWWoWoo.
                oWoWoWoWoWo
                oWoWoWoWoWo
                oWoWoooWoWo
                oWooWWWooWo
                .oWoooooWo.
                ..oWWWWWo..
                ...ooooo...
            """,
        },
        "frames": [
            ([("a", 0, 0)], 150),
            ([("b", 0, 0)], 150),
            ([("c", 0, 0)], 150),
            ([("d", 0, 0)], 150),
        ],
    },
    # 湯気（怒り・暑さ）: 頭の上。左右 2 本の湯気が、小さい→中→大きいと上がって消える（左右で順番をずらす）。くり返し
    "steam": {
        "label": "湯気（怒り・暑さ）",
        "anchor": "top",
        "loop": True,
        "parts": {
            "s": """
                ..oo..
                .oWWo.
                oWWWLo
                .oLLo.
                ..oo..
            """,
            "m": """
                ..oo....
                .oWWoo..
                oWWWWWo.
                oWWWWWLo
                .oWWLLo.
                ..oooo..
            """,
            "l": """
                ...oo.oo..
                ..oWWoWWo.
                .oWWWWWWWo
                oWWWWWWWWo
                oWWWWWWLLo
                .oLLWLLLo.
                ..oooooo..
            """,
        },
        "frames": [
            ([("s", 2, 12), ("l", 10, 0)], 200),
            ([("m", 1, 6)], 200),
            ([("l", 0, 0), ("s", 12, 12)], 200),
            ([("m", 11, 6)], 200),
        ],
    },
    # …（沈黙）: 頭の横。輪郭の色の点が 1 つずつ増えて、3 つで止まる。1 回で終わる
    "silence": {
        "label": "…（沈黙）",
        "anchor": "side",
        "loop": False,
        "parts": {
            "dot": """
                oo
                oo
            """,
        },
        "frames": [
            ([("dot", 0, 0)], 350),
            ([("dot", 0, 0), ("dot", 4, 0)], 350),
            ([("dot", 0, 0), ("dot", 4, 0), ("dot", 8, 0)], 800),
        ],
    },
    # もやもや（落ち込み）: 頭の上。渦の入った暗い雲が、左右に 1 ドットずつゆっくり揺れる。くり返し
    "gloom": {
        "label": "もやもや（落ち込み）",
        "anchor": "top",
        "loop": True,
        "parts": {
            "cloud": """
                .....ooo........
                ...ooUUUooooo...
                ..oUUUUUUUUUUo..
                .oUUUKKUUUUUUUo.
                oUUUKUUKUUKKUUUo
                oUUUUKKUUKUUKUUo
                .oUUUUUUUUKKUUo.
                ..oUUUoUUUUUUo..
                ...ooo.oooooo...
            """,
        },
        "frames": [
            ([("cloud", 1, 0)], 400),
            ([("cloud", 2, 0)], 400),
            ([("cloud", 1, 0)], 400),
            ([("cloud", 0, 0)], 400),
        ],
    },
    # 大きな汗（あせあせ）: 頭の横。上の大きい粒と下の小さい粒が、交互に外へ飛ぶ。くり返し
    "panic": {
        "label": "大きな汗（あせあせ）",
        "anchor": "side",
        "loop": True,
        "parts": {
            "big": """
                ....ooo..
                ...oBBBo.
                ..oBBBBBo
                ..oBWBBBo
                .oBWBBBBo
                .oBBBBBo.
                oBBBooo..
                oBoo.....
                .o.......
            """,
            "small": """
                ...oo.
                ..oBBo
                .oWBBo
                oBBBo.
                oBoo..
                .o....
            """,
        },
        "frames": [
            ([("big", 0, 4)], 150),
            ([("big", 3, 1)], 150),
            ([("small", 0, 14)], 150),
            ([("small", 3, 12)], 150),
        ],
    },
}
