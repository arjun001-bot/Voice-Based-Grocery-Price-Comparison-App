
import re
from typing import Dict, List

import pandas as pd
import speech_recognition as sr
import streamlit as st
from rapidfuzz import fuzz, process

FRESHCO_CSV = "freshco.csv"
NOFRILLS_CSV = "nofrills.csv"
WALMART_CSV = "walmart.csv"
DEBUG = False  

st.set_page_config(page_title="Grocery List Price Compare", layout="wide")
st.title("Grocery List — Multi-store Price Comparison")




def _read_csv_with_fallback(path: str) -> pd.DataFrame:
    """Load CSV, normalize expected columns, and return a cleaned DF."""
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        df = pd.read_csv(path, encoding="cp1252", errors="replace")
    df.columns = [c.strip().lower() for c in df.columns]

   
    if "name" not in df.columns:
        for cand in ("product", "title", "item"):
            if cand in df.columns:
                df = df.rename(columns={cand: "name"})
                break

    if "price" not in df.columns:
        for cand in ("cost", "amount", "value"):
            if cand in df.columns:
                df = df.rename(columns={cand: "price"})
                break

    if "name" not in df.columns:
        raise RuntimeError(f"{path} missing a product name column")

    if "price" not in df.columns:
        df["price"] = None

    df["name"] = df["name"].astype(str).str.strip()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    if "available" in df.columns:
        df["available"] = pd.to_numeric(df["available"], errors="coerce").fillna(0).astype(int)
    else:
        df["available"] = 1

    if "store" not in df.columns:
        lc = path.lower()
        if "freshco" in lc:
            df["store"] = "Freshco"
        elif "nofrills" in lc:
            df["store"] = "Nofrills"
        elif "walmart" in lc:
            df["store"] = "Walmart"
        else:
            df["store"] = path
    else:
        df["store"] = df["store"].astype(str)

    return df


def load_all_store_dfs() -> Dict[str, pd.DataFrame]:
    return {
        "Freshco": _read_csv_with_fallback(FRESHCO_CSV),
        "Nofrills": _read_csv_with_fallback(NOFRILLS_CSV),
        "Walmart": _read_csv_with_fallback(WALMART_CSV),
    }


def currency(v):
    return f"${v:,.2f}" if pd.notnull(v) else "—"



def listen_once(timeout: int = 3, phrase_time_limit: int = 8) -> str:
    r = sr.Recognizer()
    try:
        with sr.Microphone() as mic:
            audio = r.listen(mic, timeout=timeout, phrase_time_limit=phrase_time_limit)
    except sr.WaitTimeoutError:
        return ""
    except Exception as e:
        return f"[STT error] microphone error: {e}"

    try:
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        return f"[STT error] {e}"
    except Exception as e:
        return f"[STT error] {e}"



def split_by_separators(raw: str, max_items: int) -> List[str]:
    if not raw:
        return []
    text = raw.strip().replace("\r", "\n")
    
    text = re.sub(r"\b(and|then|also|,|comma)\b", "\n", text, flags=re.IGNORECASE)
    parts = [p.strip() for p in text.splitlines() if p.strip()]
    seen = set()
    out = []
    for p in parts:
        k = p.lower()
        if k not in seen:
            seen.add(k)
            out.append(p)
        if len(out) >= max_items:
            break
    return out


def greedy_extract_from_transcript(transcript: str, choices: List[str], max_items: int, score_cutoff: int = 78) -> List[str]:
    if not transcript:
        return []
    
    canon = {c.lower(): c for c in choices}
    text = transcript.lower()
    found = []

    
    for key in sorted(canon.keys(), key=len, reverse=True):
        if key in text:
            found.append(canon[key])
            text = text.replace(key, " ")
            if len(found) >= max_items:
                return found

 
    tokens = re.findall(r"\w+", transcript.lower())
    n_tokens = len(tokens)
    for window in range(min(6, n_tokens), 0, -1):
        i = 0
        while i + window <= n_tokens and len(found) < max_items:
            candidate = " ".join(tokens[i : i + window])
            res = process.extractOne(candidate, choices, scorer=fuzz.WRatio)
            if res and res[1] >= score_cutoff:
                matched = res[0]
                if matched not in found:
                    found.append(matched)
                i += window
                continue
            i += 1
    return found[:max_items]



def top_match_in_store(name: str, df: pd.DataFrame, score_cutoff: int = 60):
    if df is None or df.empty:
        return None, None
    name_l = name.strip().lower()
    exact = df[df["name"].str.lower() == name_l]
    if not exact.empty:
        avail = exact[exact["available"].astype(int) == 1] if "available" in exact.columns else exact
        chosen = avail if not avail.empty else exact
        price = chosen["price"].min()
        return chosen.iloc[0]["name"], float(price) if pd.notnull(price) else None

    choices = df["name"].dropna().astype(str).tolist()
    res = process.extractOne(name, choices, scorer=fuzz.WRatio, score_cutoff=score_cutoff)
    if res:
        matched = res[0]
        rows = df[df["name"] == matched]
        if not rows.empty and rows["price"].notnull().any():
            price = rows["price"].min()
            return matched, float(price)
        return matched, None
    return None, None


def build_comparison(requested_items: List[str], store_dfs: Dict[str, pd.DataFrame]):
    out = {}
    for s, df in store_dfs.items():
        rows = []
        for item in requested_items:
            matched, price = top_match_in_store(item, df)
            rows.append({"requested_item": item, "matched_name": matched, "price": price})
        out[s] = pd.DataFrame(rows)
    return out



left, right = st.columns([3, 1])
with left:
    st.write("Enter a grocery list ")
    raw_text = st.text_area("Grocery list", height=140, placeholder="e.g. milk, eggs, bread and butter")
with right:
    rec_seconds = st.number_input("Record seconds", min_value=3, max_value=20, value=6, step=1)
    max_products = st.number_input("Max products", min_value=1, max_value=30, value=10, step=1)
    gst_percent = st.number_input("GST %", min_value=0.0, max_value=100.0, value=13.0, step=0.1)

if st.button("Record voice list"):
    st.info("Recording... please start speaking your list now.")
    transcript = listen_once(timeout=3, phrase_time_limit=rec_seconds)
    if isinstance(transcript, str) and transcript.startswith("[STT error]"):
        st.error(transcript)
    else:
        st.success("Recording finished. Transcript below.")
        st.write("Transcript:", transcript)
        raw_text = (raw_text + "\n" + transcript).strip() if raw_text.strip() else (transcript or "")


items = split_by_separators(raw_text, int(max_products))


if len(items) <= 1:
    try:
        store_dfs = load_all_store_dfs()
        
        choices_map = {}
        for df in store_dfs.values():
            for n in df["name"].dropna().unique().tolist():
                key = n.strip().lower()
                if key not in choices_map:
                    choices_map[key] = n.strip()
        choices = list(choices_map.values())

        greedy = greedy_extract_from_transcript(raw_text or "", choices, int(max_products), score_cutoff=72)

        if DEBUG:
            st.markdown("**Debug — parsing**")
            st.write("Transcript (raw):", raw_text)
            st.write("Separator-split result:", items)
            st.write("Greedy extractor result:", greedy)

        if greedy and len(greedy) > len(items):
            items = greedy
        else:
            tokens = [t.strip() for t in re.split(r"\s+|,", (raw_text or "").strip()) if t.strip()]
            if len(tokens) > 1:
                stopwords = {"and", "please", "need", "want", "i", "would", "also"}
                final_tokens = []
                seen = set()
                for tk in tokens:
                    kk = tk.lower()
                    if kk in stopwords:
                        continue
                    if kk not in seen:
                        seen.add(kk)
                        final_tokens.append(tk)
                    if len(final_tokens) >= int(max_products):
                        break
                merged = []
                i = 0
                while i < len(final_tokens):
                    if i + 1 < len(final_tokens):
                        pair = f"{final_tokens[i]} {final_tokens[i+1]}"
                        res = process.extractOne(pair, choices, scorer=fuzz.WRatio)
                        if res and res[1] >= 78:
                            merged.append(res[0])
                            i += 2
                            continue
                    merged.append(final_tokens[i])
                    i += 1
                items = merged[: int(max_products)]
                if DEBUG:
                    st.write("Fallback token split used:", items)
    except Exception as ex:
        if DEBUG:
            st.write("Parsing fallback error:", ex)
        if raw_text:
            items = [p.strip() for p in raw_text.split() if p.strip()][: int(max_products)]


seen = set()
final_items = []
for it in items:
    k = it.strip().lower()
    if k and k not in seen:
        seen.add(k)
        final_items.append(it.strip())
    if len(final_items) >= int(max_products):
        break
items = final_items

if not items:
    st.info("No items to search. Type or record a list then press Record or enter text.")
    st.stop()

st.markdown(f"**Searching {len(items)} item(s)** (up to {int(max_products)}).")
st.write("Items:", ", ".join(items))

store_dfs = load_all_store_dfs()
comparison = build_comparison(items, store_dfs)

st.markdown("## Per-store results")
cols = st.columns(len(store_dfs))
store_names = list(store_dfs.keys())
subtotals = {}

for col, s in zip(cols, store_names):
    with col:
        st.subheader(s)
        df_display = comparison[s].copy()
        df_display["matched_name"] = df_display["matched_name"].fillna("No match")
        df_display["price_display"] = df_display["price"].apply(lambda v: currency(v))
        display_table = df_display[["requested_item", "matched_name", "price_display"]].rename(
            columns={"requested_item": "Requested item", "matched_name": "Matched product", "price_display": "Price"}
        )
        st.table(display_table)
        subtotal = float(comparison[s]["price"].dropna().astype(float).sum())
        subtotals[s] = subtotal
        st.write(f"**Subtotal ({s})**: {currency(subtotal)}")

gst_rate = float(gst_percent) / 100.0
summary_rows = []
for s in store_names:
    subtotal = subtotals.get(s, 0.0)
    gst_amt = subtotal * gst_rate
    grand = subtotal + gst_amt
    summary_rows.append({"Platform": s, "Subtotal": subtotal, "GST": gst_amt, "Grand total": grand})

summary_df = pd.DataFrame(summary_rows)
summary_fmt = summary_df.copy()
summary_fmt["Subtotal"] = summary_fmt["Subtotal"].apply(currency)
summary_fmt["GST"] = summary_fmt["GST"].apply(currency)
summary_fmt["Grand total"] = summary_fmt["Grand total"].apply(currency)

st.markdown("## Comparison summary")
st.table(summary_fmt)

cheapest_row = min(summary_rows, key=lambda r: r["Grand total"])
st.success(f"Cheapest platform for your list: **{cheapest_row['Platform']}** — Grand total {currency(cheapest_row['Grand total'])}")


combined = []
for s in store_names:
    temp = comparison[s].copy()
    temp["store"] = s
    combined.append(temp)
combined_df = pd.concat(combined, ignore_index=True)
combined_df = combined_df[["requested_item", "matched_name", "store", "price"]]
combined_df.columns = ["Requested item", "Matched product", "Store", "Price"]
csv_bytes = combined_df.to_csv(index=False).encode("utf-8")
st.download_button("Download full comparison CSV", data=csv_bytes, file_name="grocery_price_comparison.csv", mime="text/csv")

st.markdown(
    """
    **Notes**
    - Each store table shows how your requested item was matched to a product in that store (if any), and the price used.
    - Store Subtotal is the sum of the found prices for that store only (items not found are excluded).
    - GST is applied to the subtotal and Grand total = Subtotal + GST.
    - The cheapest platform is chosen by comparing the Grand totals.
    """
)
